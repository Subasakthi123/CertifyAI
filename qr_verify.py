"""
qr_verify.py - QR code detection, decoding and comparison.

HOW IT WORKS
  1. DETECT/DECODE with cv2.QRCodeDetector(). If the first try fails we retry
     on pre-processed versions (grayscale, enlarged, sharpened, thresholded)
     because certificate QR codes are often small or slightly blurry.
  2. PARSE the decoded text. It may be JSON, "key: value" text, a URL with
     query parameters, or just a plain ID.
  3. COMPARE the fields found in the QR with
        - the database record (if one was found), otherwise
        - the information read by OCR.
     Any conflicting field -> 'QR Mismatch'.

A certificate with NO QR is reported as 'QR Not Found' - never as fake.
"""

import json
import re
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from utils import normalize_id, normalize_name, tokens_found_fraction

# Names a QR payload might use for each field
_KEY_ALIASES = {
    "certificate_id": {"certificate_id", "certid", "cert_id", "certificate", "cid", "id", "certificateid"},
    "register_number": {"register_number", "registernumber", "regno", "reg_no", "reg", "register", "roll", "rollno", "registration"},
    "student_name": {"student_name", "name", "studentname", "student", "candidate"},
    "certificate_number": {"certificate_number", "certno", "cert_no", "certificatenumber"},
}


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------
def _candidate_images(image_bgr: np.ndarray):
    """Yield differently prepared versions of the image for QR detection."""
    yield image_bgr
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    yield gray
    for scale in (2.0, 0.5):                       # bigger helps tiny codes, smaller helps huge ones
        yield cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    yield clahe
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield otsu
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    yield cv2.addWeighted(gray, 1.8, blur, -0.8, 0)  # unsharp mask
    # QR codes usually sit in a corner: try the four quarter-crops at 2x
    h, w = gray.shape
    for ys, xs in ((0, 0), (0, w // 2), (h // 2, 0), (h // 2, w // 2)):
        crop = gray[ys:ys + h // 2, xs:xs + w // 2]
        yield cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)


def detect_qr(image_bgr: np.ndarray):
    """Return (decoded_text, detected_flag). decoded_text is '' when it could not be decoded."""
    detector = cv2.QRCodeDetector()
    detected = False
    for candidate in _candidate_images(image_bgr):
        try:
            data, points, _ = detector.detectAndDecode(candidate)
        except cv2.error:
            continue
        if points is not None:
            detected = True
        if data:
            return data, True
    return "", detected


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------
def _map_keys(raw: dict) -> dict:
    """Rename arbitrary keys (e.g. 'RegNo') to our standard field names."""
    mapped = {}
    for key, value in raw.items():
        simple = re.sub(r"[^a-z_]", "", str(key).lower())
        for field, aliases in _KEY_ALIASES.items():
            if simple in aliases and value not in (None, ""):
                mapped[field] = str(value).strip()
    return mapped


def parse_qr_payload(data: str) -> dict:
    """Return {'kind': ..., 'fields': {...}, 'url': ...} for the decoded QR text."""
    data = data.strip()
    info = {"kind": "text", "fields": {}, "url": None, "raw": data}

    # JSON
    if data.startswith("{"):
        try:
            parsed = json.loads(data)
            if isinstance(parsed, dict):
                info.update(kind="json", fields=_map_keys(parsed))
                return info
        except json.JSONDecodeError:
            pass

    # URL (possibly with ?id=...&reg=...)
    if re.match(r"^https?://", data, re.IGNORECASE):
        parsed_url = urlparse(data)
        query = {k: v[0] for k, v in parse_qs(parsed_url.query).items()}
        fields = _map_keys(query)
        # also accept a certificate id as the last part of the path:  /verify/DEMO-CERT-2025-001
        last_part = parsed_url.path.rstrip("/").split("/")[-1]
        if "certificate_id" not in fields and re.search(r"\d", last_part) and len(last_part) >= 6:
            fields["certificate_id"] = last_part
        info.update(kind="url", fields=fields, url=data)
        return info

    # "Key: value" lines (or key=value; key=value)
    pairs = re.findall(r"([A-Za-z_ ]{2,25})\s*[:=]\s*([^\n;|,]+)", data)
    if pairs:
        fields = _map_keys({k.strip(): v.strip() for k, v in pairs})
        if fields:
            info.update(kind="key-value text", fields=fields)
            return info

    # A bare token that looks like an identifier
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9\-/_.]{5,}", data) and re.search(r"\d", data):
        info.update(kind="plain ID", fields={"certificate_id": data})
    return info


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------
def _same(field: str, qr_value: str, other_value: str, rescue_text: str = "") -> bool:
    """
    Compare one QR field with the reference value.
    rescue_text (raw OCR text) is only used when comparing with OCR output: if the
    OCR field extractor grabbed the wrong text, the QR value may still be in the page.
    """
    if field == "student_name":
        if normalize_name(qr_value) == normalize_name(other_value):
            return True
        return bool(rescue_text) and tokens_found_fraction(qr_value, rescue_text) == 1.0
    if normalize_id(qr_value) == normalize_id(other_value):
        return True
    key = normalize_id(qr_value)
    return bool(rescue_text) and len(key) >= 6 and key in normalize_id(rescue_text)


def verify_qr(image_bgr: np.ndarray, ocr_fields: dict, db_record: dict = None, ocr_text: str = "") -> dict:
    """
    Detect, decode and compare the QR code.

    Returns dict(status, raw, kind, url, qr_fields, comparisons, message)
    """
    out = {"status": "QR Not Found", "raw": "", "kind": "", "url": None, "qr_fields": {},
           "comparisons": [], "message": "No QR code was detected on this certificate."}
    try:
        data, detected = detect_qr(image_bgr)
    except Exception:
        out.update(status="QR Unreadable", message="QR detection failed because of an image processing problem.")
        return out

    if not data:
        if detected:
            out.update(status="QR Unreadable",
                       message="A QR code was detected but could not be decoded (blur, low resolution or damage).")
        return out

    info = parse_qr_payload(data)
    out.update(raw=data, kind=info["kind"], url=info["url"], qr_fields=info["fields"])

    # compare against the database record when we have one, otherwise against OCR
    source_name = "database" if db_record else "certificate text (OCR)"
    reference = db_record if db_record else ocr_fields
    comparisons = []
    for field, qr_value in info["fields"].items():
        ref_value = reference.get(field)
        if not ref_value:
            # OCR did not isolate this field. We can still confirm the QR value if it
            # appears anywhere in the page text; otherwise there is nothing to compare.
            if not db_record and ocr_text and _same(field, qr_value, "", ocr_text):
                comparisons.append({"field": field, "qr_value": qr_value, "reference_value": "found in page text",
                                    "result": "Match"})
            continue
        comparisons.append({"field": field, "qr_value": qr_value, "reference_value": str(ref_value),
                            "result": "Match" if _same(field, qr_value, str(ref_value), "" if db_record else ocr_text) else "Mismatch"})
    out["comparisons"] = comparisons

    if any(c["result"] == "Mismatch" for c in comparisons):
        out.update(status="QR Mismatch",
                   message=f"QR information conflicts with the {source_name}.")
    elif comparisons:
        out.update(status="QR Verified", message=f"QR information matches the {source_name}.")
    else:
        out.update(status="QR Data Detected - Unable to Fully Verify",
                   message="The QR code was read, but it holds no fields that can be compared "
                           "(e.g. a web link only). Online verification would be needed.")
    return out
