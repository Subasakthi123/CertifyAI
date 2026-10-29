"""
tamper_detection.py - AI-assisted image tampering analysis (PROTOTYPE).

IDEA IN ONE PARAGRAPH
  When a JPEG image is edited (text replaced, region pasted) and saved again,
  the edited area usually has a different compression history from the rest
  of the picture. We measure this with ERROR LEVEL ANALYSIS (ELA):
  re-save the image once at a known JPEG quality and look at how much each
  pixel changes. Untouched areas change in a consistent way; edited areas
  often react differently. We turn this (plus edge / noise statistics) into a
  FEATURE VECTOR and then either
     (a) feed it to a trained scikit-learn model  (models/tamper_model.pkl), or
     (b) if no trained model exists, compute a transparent HEURISTIC risk score.

  The heuristic is a prototype: its thresholds were chosen by hand and are
  NOT validated on a large forged-certificate dataset. It never "proves"
  forgery - it only highlights potentially suspicious regions.

FEATURES (all computed from the image, nothing is random or hard-coded)
  ela_mean / ela_std / ela_p99 : overall size and spread of ELA differences
  ela_hot_fraction             : share of pixels with unusually large ELA values
  suspicious_block_ratio       : share of 32x32 text/edge blocks whose ELA is
                                 out of line with their local edge strength
  max_block_zscore             : the single most unusual block (robust z-score)
  noise_std / noise_block_cv   : sensor/scanner noise level and how unevenly it
                                 varies over flat background (patched areas often
                                 have different noise)
  edge_density                 : fraction of Canny edge pixels
  laplacian_var_log            : sharpness statistic (log of Laplacian variance)
  gradient_mean                : average Sobel gradient strength
"""

import cv2
import joblib
import numpy as np

from utils import MODEL_PATH

BLOCK = 32                                   # block size (pixels) for local statistics
ELA_QUALITY = 90                             # JPEG quality used for the ELA re-save
Z_FLAG = 4.5                           # robust z-score above which a block is called suspicious

FEATURE_NAMES = ["ela_mean", "ela_std", "ela_p99", "ela_hot_fraction", "suspicious_block_ratio",
                 "max_block_zscore", "noise_std", "noise_block_cv", "edge_density",
                 "laplacian_var_log", "gradient_mean"]

# Risk thresholds (0-100). Documented in the README as prototype values.
LOW_RISK_BELOW = 40
HIGH_RISK_FROM = 70

STATUS_OK = "No obvious tampering detected"
STATUS_MID = "Analysis inconclusive"
STATUS_HIGH = "Potential tampering detected"


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------
def compute_ela(image_bgr: np.ndarray, quality: int = ELA_QUALITY) -> np.ndarray:
    """Error Level Analysis: |original - re-saved JPEG|, strongest channel per pixel."""
    ok, encoded = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG re-save failed")
    resaved = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    diff = cv2.absdiff(image_bgr, resaved)
    return diff.max(axis=2).astype(np.float32)


def _block_mean(array: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """Average an image over non-overlapping block x block tiles."""
    h = (array.shape[0] // block) * block
    w = (array.shape[1] // block) * block
    tiles = array[:h, :w].reshape(h // block, block, w // block, block)
    return tiles.mean(axis=(1, 3))


def extract_features(image_bgr: np.ndarray):
    """
    Compute the feature dictionary.
    Returns (features, ela_map, flagged_mask) where flagged_mask is a boolean
    grid (one cell per block) marking potentially suspicious blocks.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    ela = compute_ela(image_bgr)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)

    # noise residual = what remains after removing the smooth image content
    residual = np.abs(gray.astype(np.float32) - cv2.medianBlur(gray, 3).astype(np.float32))

    ela_b = _block_mean(ela)
    grad_b = _block_mean(gradient)
    noise_b = _block_mean(residual)

    # --- suspicious blocks -------------------------------------------------
    # In a consistently-saved image the ELA of a block is roughly proportional to
    # how much text/edge detail the block holds. We therefore divide each block's
    # ELA by its edge density and look for blocks that break the pattern, using a
    # robust z-score (median / MAD, so a few outliers do not distort the baseline).
    edge_b = _block_mean((cv2.Canny(gray, 100, 200) > 0).astype(np.float32))
    content = edge_b > 0.03                    # blocks that contain text / edges
    flagged = np.zeros_like(content)
    suspicious_ratio, max_z = 0.0, 0.0
    if content.sum() >= 12:
        ratio = (ela_b + 0.1) / (edge_b + 0.01)
        values = ratio[content]
        median = np.median(values)
        mad = np.median(np.abs(values - median)) * 1.4826 + 1e-3
        z = (ratio - median) / mad
        flagged = content & (z > Z_FLAG) & (ela_b > ela.mean())
        suspicious_ratio = float(flagged.sum() / content.sum())
        max_z = float(min(z[content].max(), 20.0))

    # --- background noise consistency --------------------------------------
    flat = grad_b <= 15
    noise_cv = 0.0
    if flat.sum() >= 12 and noise_b[flat].mean() > 0.05:
        noise_cv = float(noise_b[flat].std() / noise_b[flat].mean())

    ela_mean, ela_std = float(ela.mean()), float(ela.std())
    features = {
        "ela_mean": ela_mean,
        "ela_std": ela_std,
        "ela_p99": float(np.percentile(ela, 99)),
        "ela_hot_fraction": float((ela > ela_mean + 3 * ela_std).mean()),
        "suspicious_block_ratio": suspicious_ratio,
        "max_block_zscore": max_z,
        "noise_std": float(residual.mean()),
        "noise_block_cv": noise_cv,
        "edge_density": float((cv2.Canny(gray, 100, 200) > 0).mean()),
        "laplacian_var_log": float(np.log1p(cv2.Laplacian(gray, cv2.CV_64F).var())),
        "gradient_mean": float(gradient.mean()),
    }
    return features, ela, flagged


def feature_vector(features: dict) -> np.ndarray:
    """Features in the fixed order expected by the ML model."""
    return np.array([[features[name] for name in FEATURE_NAMES]], dtype=np.float64)


# --------------------------------------------------------------------------
# Scoring: heuristic (no trained model) or trained model
# --------------------------------------------------------------------------
def heuristic_risk(features: dict):
    """
    Transparent rule-based score from 0 to 100.
    Conservative prototype heuristic.
    """

    parts = {
        "suspicious_block_ratio": float(
            np.clip(features["suspicious_block_ratio"] / 0.05, 0, 1)
        ),
        "max_block_zscore": float(
            np.clip((features["max_block_zscore"] - 6.0) / 10.0, 0, 1)
        ),
        "ela_hot_fraction": float(
            np.clip(features["ela_hot_fraction"] / 0.03, 0, 1)
        ),
        "noise_block_cv": float(
            np.clip((features["noise_block_cv"] - 0.8) / 1.2, 0, 1)
        ),
    }

    weights = {
        "suspicious_block_ratio": 0.45,
        "max_block_zscore": 0.20,
        "ela_hot_fraction": 0.15,
        "noise_block_cv": 0.20,
    }

    score = 100.0 * sum(
        parts[k] * weights[k] for k in parts
    )

    strong_signals = sum(
        1 for value in parts.values()
        if value >= 0.65
    )

    if strong_signals <= 1:
        score = min(score, 39.0)
    elif strong_signals == 2:
        score = min(score, 65.0)
    else:
        score = min(score, 100.0)

    return score, parts


def load_model():
    """Load models/tamper_model.pkl. Returns (payload_or_None, message)."""
    if not MODEL_PATH.exists():
        return None, "No trained model file found (models/tamper_model.pkl) - using heuristic analysis."
    try:
        payload = joblib.load(MODEL_PATH)
        if not isinstance(payload, dict) or "model" not in payload:
            return None, "Model file has an unexpected format - using heuristic analysis."
        if list(payload.get("feature_names", [])) != FEATURE_NAMES:
            return None, "Model was trained with different features - using heuristic analysis."
        return payload, "Trained model loaded."
    except Exception:
        return None, "Model file could not be loaded - using heuristic analysis."


def _status_from_score(score: float) -> str:
    if score < LOW_RISK_BELOW:
        return STATUS_OK
    if score >= HIGH_RISK_FROM:
        return STATUS_HIGH
    return STATUS_MID


# --------------------------------------------------------------------------
# Visualisation
# --------------------------------------------------------------------------
def build_heatmap(image_bgr: np.ndarray, ela: np.ndarray, flagged: np.ndarray) -> np.ndarray:
    """
    Colour map of ELA values blended over the certificate, with boxes around
    'potentially suspicious regions'. Returns an RGB array for display.
    """
    scale = max(float(np.percentile(ela, 99.5)), 1.0)
    ela_u8 = np.clip(ela / scale * 255.0, 0, 255).astype(np.uint8)
    colour = cv2.applyColorMap(ela_u8, cv2.COLORMAP_INFERNO)
    overlay = cv2.addWeighted(image_bgr, 0.45, colour, 0.55, 0)
    for by, bx in zip(*np.nonzero(flagged)):
        top_left = (int(bx * BLOCK), int(by * BLOCK))
        bottom_right = (int((bx + 1) * BLOCK), int((by + 1) * BLOCK))
        cv2.rectangle(overlay, top_left, bottom_right, (255, 0, 255), 2)   # magenta box (BGR)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def analyze_tampering(image_bgr: np.ndarray) -> dict:
    """
    Run the tamper analysis. Never raises.

    Returns dict(status, risk_score, mode, mode_label, indicators, features,
                 heatmap_rgb, flagged_blocks, message)
    risk_score is None whenever no real score could be computed.
    """
    out = {"status": STATUS_MID, "risk_score": None, "mode": "none", "mode_label": "",
           "indicators": [], "features": {}, "heatmap_rgb": None, "flagged_blocks": 0, "message": ""}
    try:
        h, w = image_bgr.shape[:2]
        if min(h, w) < 200:
            out["message"] = "The image is too small for a meaningful forensic analysis."
            return out

        features, ela, flagged = extract_features(image_bgr)
        out["features"] = features
        out["flagged_blocks"] = int(flagged.sum())
        out["heatmap_rgb"] = build_heatmap(image_bgr, ela, flagged)

        model_payload, model_message = load_model()
        parts = None
        if model_payload is not None:
            try:
                prob = model_payload["model"].predict_proba(feature_vector(features))[0]
                classes = list(model_payload["model"].classes_)
                score = float(prob[classes.index(1)] * 100.0)
                out["mode"] = "model"
                out["mode_label"] = ("Trained scikit-learn model (" + str(model_payload.get("description", "tamper_model.pkl")) + ")")
            except Exception:
                model_payload = None
                model_message = "The trained model failed to run - using heuristic analysis."
        if model_payload is None:
            score, parts = heuristic_risk(features)
            out["mode"] = "heuristic"
            out["mode_label"] = "Prototype heuristic analysis (no trained model available)"
            out["message"] = model_message

        out["risk_score"] = round(score, 1)
        out["status"] = _status_from_score(score)
        out["indicators"] = _describe_indicators(features, flagged, parts)
        return out
    except Exception:
        out.update(status=STATUS_MID, risk_score=None, mode="none",
                   message="The tampering analysis could not be completed for this image.")
        return out


def _describe_indicators(features: dict, flagged: np.ndarray, parts) -> list:
    """Plain-language explanation of what the numbers mean."""
    notes = []
    n_flagged = int(flagged.sum())
    if n_flagged:
        notes.append(f"{n_flagged} image block(s) show compression/error-level behaviour that differs from their surroundings (boxed in the analysis image).")
    if features["noise_block_cv"] > 1.2:
        notes.append("Background noise level varies unevenly across the page, which can happen when areas were patched or pasted.")
    if features["ela_hot_fraction"] > 0.02:
        notes.append("A noticeable share of pixels show unusually strong error-level differences.")
    if not notes:
        notes.append("No block showed unusual compression, edge or noise behaviour compared with the rest of the image.")
    if parts is None:
        notes.append("Score produced by the model from the same forensic features.")
    return notes
