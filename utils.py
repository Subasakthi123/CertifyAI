"""
utils.py - shared helpers for CertifyAI.

Contents
  1. Paths and constants
  2. File loading (JPG / PNG / PDF -> OpenCV image)
  3. Text normalisation helpers (used by database + QR comparison)
  4. The FINAL DECISION ENGINE that combines all four checks

Everything here is plain Python so it is easy to explain in a viva.
"""

import hashlib
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

# --------------------------------------------------------------------------
# 1. Paths and constants
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
ASSETS_DIR = BASE_DIR / "assets"
SAMPLE_DIR = BASE_DIR / "sample_certificates"
MODEL_PATH = BASE_DIR / "models" / "tamper_model.pkl"
DB_PATH = BASE_DIR / "certificates.db"

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf"}
MAX_FILE_MB = 15                 # reject very large uploads to protect memory
MAX_IMAGE_SIDE = 2400            # large scans are shrunk to this many pixels

# Final status labels (the three primary outcomes required by the project)
FINAL_VERIFIED = "GENUINE / VERIFIED"
FINAL_SUSPICIOUS = "SUSPICIOUS / POTENTIALLY TAMPERED"
FINAL_UNVERIFIED = "UNVERIFIED / VERIFICATION FAILED"

DISCLAIMER = (
    "This is an AI-assisted prototype analysis and does not independently "
    "prove document authenticity."
)


# --------------------------------------------------------------------------
# 2. File loading
# --------------------------------------------------------------------------
@dataclass
class LoadedCertificate:
    """Result of trying to open an uploaded file."""
    ok: bool
    image_bgr: Optional[np.ndarray] = None   # OpenCV image (BGR) for analysis
    image_rgb: Optional[Image.Image] = None  # PIL image for preview
    is_pdf: bool = False
    pdf_text: str = ""                       # text layer of PDF page 1 (if any)
    page_count: int = 1
    error: str = ""                          # user friendly message when ok=False
    notes: list = field(default_factory=list)


def format_file_size(num_bytes: int) -> str:
    """Turn 183204 into '178.9 KB'."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def file_hash(data: bytes) -> str:
    """Short fingerprint used to detect that a *new* file was uploaded."""
    return hashlib.sha256(data).hexdigest()[:16]


def safe_filename(name: str) -> str:
    """Remove risky characters from an uploaded file name."""
    name = Path(name).name
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)[:80] or "upload"


def _shrink_if_large(image_bgr: np.ndarray) -> np.ndarray:
    """Downscale very large images so analysis stays fast."""
    h, w = image_bgr.shape[:2]
    longest = max(h, w)
    if longest <= MAX_IMAGE_SIDE:
        return image_bgr
    scale = MAX_IMAGE_SIDE / longest
    return cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _pil_to_bgr(pil_image: Image.Image) -> np.ndarray:
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def load_certificate(data: bytes, filename: str) -> LoadedCertificate:
    """
    Open an uploaded certificate (JPG, PNG or the first page of a PDF).
    Never raises: problems are returned as ok=False with a friendly message.
    """
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return LoadedCertificate(False, error="Unsupported file type. Please upload a JPG, JPEG, PNG or PDF file.")
    if not data:
        return LoadedCertificate(False, error="The uploaded file is empty.")
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        return LoadedCertificate(False, error=f"File is larger than {MAX_FILE_MB} MB. Please upload a smaller file.")

    if ext == ".pdf":
        return _load_pdf(data)
    return _load_image(data)


def _load_image(data: bytes) -> LoadedCertificate:
    try:
        import io
        pil = Image.open(io.BytesIO(data))
        pil.load()                                   # forces decoding -> catches corrupted files
        pil = pil.convert("RGB")
        bgr = _shrink_if_large(_pil_to_bgr(pil))
        preview = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return LoadedCertificate(True, image_bgr=bgr, image_rgb=preview)
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        return LoadedCertificate(False, error="The image could not be opened. The file may be corrupted or not a real image.")
    except Exception:
        return LoadedCertificate(False, error="Unexpected problem while reading the image.")


def _load_pdf(data: bytes) -> LoadedCertificate:
    """Render page 1 of a PDF to an image using PyMuPDF (pure pip install, no system tools)."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return LoadedCertificate(False, is_pdf=True, error="PDF support needs the PyMuPDF package (pip install PyMuPDF).")
    try:
        doc = fitz.open(stream=data, filetype="pdf")
        if doc.page_count == 0:
            return LoadedCertificate(False, is_pdf=True, error="The PDF has no pages.")
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=200)               # 200 dpi is a good OCR / analysis size
        pil = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        bgr = _shrink_if_large(_pil_to_bgr(pil))
        preview = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        text = page.get_text("text") or ""
        notes = []
        if doc.page_count > 1:
            notes.append(f"PDF has {doc.page_count} pages - only the first page is analysed.")
        result = LoadedCertificate(True, image_bgr=bgr, image_rgb=preview, is_pdf=True,
                                   pdf_text=text, page_count=doc.page_count, notes=notes)
        doc.close()
        return result
    except Exception:
        return LoadedCertificate(False, is_pdf=True, error="The PDF could not be converted. It may be corrupted or password protected.")


# --------------------------------------------------------------------------
# 3. Text normalisation helpers
# --------------------------------------------------------------------------
# OCR often confuses look-alike characters. For IDs we map them to one symbol
# so that "DEM0-CERT-2O25-OO1" and "DEMO-CERT-2025-001" still compare equal.
# (Only these specific pairs are merged - this is NOT a loose fuzzy match.)
_CONFUSABLE = str.maketrans({"O": "0", "I": "1", "L": "1", "S": "5", "B": "8"})


def normalize_id(value: str) -> str:
    """Upper-case, keep only letters/digits, merge OCR look-alike characters."""
    if not value:
        return ""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", str(value)).upper()
    return cleaned.translate(_CONFUSABLE)


def normalize_text(value: str) -> str:
    """Lower-case, drop punctuation, collapse spaces. Used for names/institution/course."""
    if not value:
        return ""
    value = re.sub(r"[^a-z0-9\s]", " ", str(value).lower())
    return re.sub(r"\s+", " ", value).strip()


_TITLES = {"mr", "mrs", "ms", "miss", "dr", "shri", "smt", "kumari", "prof"}


def normalize_name(value: str) -> str:
    """Normalise a person name: remove titles, sort tokens (so 'KUMAR ARUN' == 'Arun Kumar')."""
    tokens = [t for t in normalize_text(value).split() if t not in _TITLES]
    return " ".join(sorted(tokens))


def similarity(a: str, b: str) -> float:
    """0..1 similarity between two already-normalised strings."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def tokens_found_fraction(value: str, text: str) -> float:
    """
    What fraction of the words in `value` also appear (almost exactly) in `text`?
    A word counts as found when some word in the text is >= 85% similar.
    Used to check e.g. whether the database course name occurs in the OCR text.
    """
    value_tokens = [t for t in normalize_text(value).split() if len(t) > 1]
    text_tokens = set(normalize_text(text).split())
    if not value_tokens or not text_tokens:
        return 0.0
    found = 0
    for token in value_tokens:
        if token in text_tokens or any(similarity(token, t) >= 0.85 for t in text_tokens if abs(len(t) - len(token)) <= 2):
            found += 1
    return found / len(value_tokens)


# --------------------------------------------------------------------------
# 4. FINAL DECISION ENGINE
# --------------------------------------------------------------------------
def decide_final_status(ocr: dict, db: dict, qr: dict, tamper: dict) -> dict:
    """
    Combine the four evidence sources into one transparent decision.

    Evidence is first turned into three kinds of signal:

      STRONG POSITIVE : database record found AND its fields match the certificate
      STRONG NEGATIVE : database identity mismatch, QR conflicting with the
                        record/certificate, or significant tampering indicators
      NEUTRAL         : missing QR, unreadable QR, no database record,
                        inconclusive tamper analysis ("we simply do not know")

    Rules (checked in this order):
      1. Any STRONG NEGATIVE signal              -> SUSPICIOUS / POTENTIALLY TAMPERED
      2. DB verified and nothing negative        -> GENUINE / VERIFIED
      3. Everything else (not enough evidence)   -> UNVERIFIED / VERIFICATION FAILED

    A missing QR or a missing database record is NEVER treated as a forgery
    on its own - it only means we lack evidence.
    """
    reasons = []          # list of (level, text) with level in ok / warn / bad / info
    negatives = []
    positives = []

    ocr_status = ocr.get("status", "OCR Failed")
    db_status = db.get("status", "Insufficient Information")
    qr_status = qr.get("status", "QR Not Found")
    tamper_status = tamper.get("status", "Analysis inconclusive")

    # ---- OCR -------------------------------------------------------------
    if ocr_status == "OCR Failed":
        reasons.append(("bad", "OCR could not read usable text from the certificate, so identity details could not be checked."))
    elif ocr_status == "OCR Partially Completed":
        reasons.append(("warn", "OCR read only part of the certificate information; some fields were not detected."))
    else:
        reasons.append(("ok", "OCR extracted the certificate text and key details."))

    # ---- Database --------------------------------------------------------
    if db_status == "Verified":
        positives.append("database")
        reasons.append(("ok", "The certificate details match a record in the certificate database (field-by-field comparison)."))
    elif db_status == "Mismatch":
        negatives.append("database")
        bad_fields = ", ".join(db.get("mismatched_fields", [])) or "identity fields"
        reasons.append(("bad", f"A database record was found but these fields do not match the certificate: {bad_fields}."))
    elif db_status == "No Record Found":
        reasons.append(("warn", "No matching record exists in the database. This does NOT mean the certificate is fake - the institution may simply not be in this demo database."))
    else:
        reasons.append(("warn", "Not enough identifying details (certificate ID / register number) were detected to search the database."))

    # ---- QR --------------------------------------------------------------
    if qr_status == "QR Verified":
        positives.append("qr")
        reasons.append(("ok", "The QR code content is consistent with the certificate details."))
    elif qr_status == "QR Mismatch":
        negatives.append("qr")
        reasons.append(("bad", "The QR code contains information that conflicts with the certificate / database record."))
    elif qr_status == "QR Not Found":
        reasons.append(("info", "No QR code was found. Older certificates may legitimately have none, so this is not treated as evidence of forgery."))
    elif qr_status == "QR Unreadable":
        reasons.append(("warn", "A QR-like pattern was detected but could not be decoded (blur, low resolution or damage). Treated as neutral."))
    else:
        reasons.append(("info", "QR data was read but cannot be fully verified offline (for example a web link only)."))

    # ---- Tampering -------------------------------------------------------
    if tamper_status == "Potential tampering detected":
        negatives.append("tamper")
        reasons.append(("bad", "The AI-assisted image analysis found significant indicators of possible editing (see highlighted regions)."))
    elif tamper_status == "No obvious tampering detected":
        reasons.append(("ok", "The image analysis found no obvious signs of editing (this does not prove authenticity)."))
    else:
        reasons.append(("info", "The image analysis was inconclusive, so it is treated as neutral."))

    # ---- Final rule ------------------------------------------------------
    if negatives:
        final = FINAL_SUSPICIOUS
        level = "bad"
        summary = ("Contradicting evidence was found: " + ", ".join(negatives).replace("database", "database mismatch")
                   .replace("qr", "QR conflict").replace("tamper", "tampering indicators")
                   + ". The certificate should be checked manually with the issuing institution.")
    elif "database" in positives:
        final = FINAL_VERIFIED
        level = "ok"
        extra = " QR evidence also agrees." if "qr" in positives else ""
        summary = ("The certificate details match a database record and no strong contradicting evidence was found." + extra
                   + " Verification evidence supports the certificate, but this is not legal proof of authenticity.")
    else:
        final = FINAL_UNVERIFIED
        level = "warn"
        summary = ("There is not enough reliable evidence to verify this certificate. "
                   "No strong evidence of tampering was found either, so it is NOT declared fake - "
                   "please verify it with the issuing institution.")

    return {"final_status": final, "level": level, "summary": summary,
            "reasons": reasons, "positives": positives, "negatives": negatives}
