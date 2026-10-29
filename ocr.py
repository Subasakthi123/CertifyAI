"""
ocr.py - text extraction and certificate field detection.

Pipeline
  image -> preprocessing -> EasyOCR -> text lines -> regex/keyword field extraction

WHY PREPROCESSING?
  Scanned or photographed certificates are often low-contrast and noisy.
    * grayscale        : colour is not needed to read text, and it is faster
    * upscaling        : OCR works best when characters are ~20-40 px tall
    * CLAHE contrast   : boosts faint text without blowing out the background
    * light denoising  : removes scanner grain that confuses character shapes
    * thresholding     : turns text black / paper white. It can HELP on faded
                         scans but can also DESTROY thin letters, so we only
                         use it as a second attempt if the first pass read
                         very little text.
  We deliberately keep preprocessing mild so OCR quality is not damaged.

HOW FIELDS ARE EXTRACTED
  Certificates have no fixed layout, so we do not rely on one template.
  For every field we try, in order:
    1. "Label : value" on the same line   (e.g. "Register No: ABC123")
    2. the value on the line AFTER the label
    3. keyword / sentence patterns        (e.g. "This is to certify that <NAME>")
  Anything we cannot find is reported as "Not detected" - nothing is invented.
"""

import re
from functools import lru_cache

import cv2
import numpy as np

# Fields we try to find (keys are used by database.py and app.py)
FIELD_KEYS = ["student_name", "register_number", "certificate_id", "certificate_number",
              "institution", "course", "year"]

# The fields that matter most for deciding OCR "completeness"
CORE_FIELDS = ["student_name", "register_number", "certificate_id", "institution", "course", "year"]


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
def preprocess_for_ocr(image_bgr: np.ndarray, threshold: bool = False) -> np.ndarray:
    """Return a grayscale image prepared for OCR (see module docstring for the reasons)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # upscale small images so characters are large enough; cap size for speed
    h, w = gray.shape
    longest = max(h, w)
    if longest < 1400:
        scale = 1400 / longest
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    elif longest > 2400:
        scale = 2400 / longest
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # local contrast improvement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # gentle denoising (small h keeps the letter shapes)
    gray = cv2.fastNlMeansDenoising(gray, None, h=7, templateWindowSize=7, searchWindowSize=21)

    if threshold:
        # adaptive threshold copes with uneven lighting across the page
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 15)
    return gray


# --------------------------------------------------------------------------
# EasyOCR (loaded lazily and only once)
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_reader():
    """Create the EasyOCR reader once (loading the model takes several seconds)."""
    import easyocr  # imported here so the app still starts if EasyOCR is missing
    return easyocr.Reader(["en"], gpu=False, verbose=False)


def _group_into_lines(results) -> list:
    """
    EasyOCR returns separate word boxes. Boxes whose vertical centres are close
    belong to the same text line, so we group them and sort left-to-right.
    """
    items = []
    for box, text, conf in results:
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        items.append({"text": text.strip(), "x": min(xs), "yc": (min(ys) + max(ys)) / 2,
                      "h": max(ys) - min(ys), "conf": float(conf)})
    items = [i for i in items if i["text"]]
    if not items:
        return []
    items.sort(key=lambda i: i["yc"])
    lines, current = [], [items[0]]
    for item in items[1:]:
        ref_h = max(np.mean([c["h"] for c in current]), 1)
        if abs(item["yc"] - np.mean([c["yc"] for c in current])) <= 0.6 * ref_h:
            current.append(item)
        else:
            lines.append(current)
            current = [item]
    lines.append(current)
    return [" ".join(i["text"] for i in sorted(line, key=lambda i: i["x"])) for line in lines]


def _run_easyocr(image_bgr: np.ndarray):
    """Run OCR (and a thresholded second attempt when the first read is poor)."""
    reader = get_reader()
    best_lines, best_conf = [], 0.0
    for use_threshold in (False, True):
        prepared = preprocess_for_ocr(image_bgr, threshold=use_threshold)
        results = reader.readtext(prepared, detail=1, paragraph=False)
        lines = _group_into_lines(results)
        conf = float(np.mean([r[2] for r in results])) if results else 0.0
        chars = sum(len(l) for l in lines)
        if chars > sum(len(l) for l in best_lines):
            best_lines, best_conf = lines, conf
        if chars >= 150:          # first attempt already read plenty of text
            break
    return best_lines, best_conf


# --------------------------------------------------------------------------
# Field extraction
# --------------------------------------------------------------------------
_ID_VALUE = r"([A-Za-z0-9][A-Za-z0-9\-/_.]{3,})"   # a token that looks like an identifier

_LABEL_PATTERNS = {
    "certificate_id": r"(?:certificate|cert\.?)\s*(?:id|i\.d\.?)\b",
    "certificate_number": r"(?:certificate|cert\.?)\s*(?:no\.?|number|num\.?)\b",
    "register_number": r"(?:register(?:ed)?|registration|reg\.?|roll|enrol+ment|hall\s*ticket)\s*(?:no\.?|number|num\.?|id)?\b",
}


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" :-.,;|")


def _value_after_label(lines, label_regex, value_regex=_ID_VALUE):
    """Find 'LABEL [: -] value' on one line, or the value on the next line."""
    pattern = re.compile(label_regex + r"\s*[:\-#.]*\s*" + value_regex, re.IGNORECASE)
    label_only = re.compile(label_regex + r"\s*[:\-#.]*\s*$", re.IGNORECASE)
    for idx, line in enumerate(lines):
        match = pattern.search(line)
        if match:
            return _clean(match.group(match.lastindex))
        if label_only.search(line) and idx + 1 < len(lines):
            nxt = re.match(value_regex, lines[idx + 1].strip())
            if nxt:
                return _clean(nxt.group(1))
    return None


def _looks_like_id(value: str) -> bool:
    """IDs normally contain at least one digit (filters out words like 'Number')."""
    return bool(value) and any(ch.isdigit() for ch in value)


def _extract_ids(lines) -> dict:
    found = {}
    # certificate number / id are matched first so "Certificate No" is not mistaken for a register number
    for key in ("certificate_id", "certificate_number", "register_number"):
        value = _value_after_label(lines, _LABEL_PATTERNS[key])
        if value and _looks_like_id(value):
            found[key] = value
    return found


_NAME_STOP = re.compile(r"(?:\bregister\b|\breg\b|\broll\b|\bcertificate\b|\bhas\b|\bhaving\b|\bbearing\b|\bwith\b|\bsuccessfully\b|\bof\b)", re.IGNORECASE)


def _clean_name(raw: str):
    """Keep only a plausible person name: 2-5 alphabetic words."""
    raw = _NAME_STOP.split(raw)[0]
    raw = re.sub(r"[^A-Za-z.\s'-]", " ", raw)
    raw = _clean(raw)
    words = raw.split()
    if 2 <= len(words) <= 5 and all(len(w) > 1 for w in words):
        return raw.title() if raw.isupper() else raw
    return None


def _extract_name(lines):
    # 1) explicit labels: "Name of the Student : X", "Student Name : X", "Candidate Name : X"
    label_regex = r"(?:name\s+of\s+the\s+(?:student|candidate)|(?:student|candidate)(?:'s)?\s+name|\bname\b)"
    value = _value_after_label(lines, label_regex, r"([A-Za-z][A-Za-z.\s'-]{3,})")
    name = _clean_name(value) if value else None
    if name:
        return name
    # 2) sentence patterns: "This is to certify that X", "awarded to X", "presented to X"
    trigger = re.compile(r"(?:certify\s+that|awarded\s+to|presented\s+to|conferred\s+(?:up)?on|granted\s+to|issued\s+to)\s*[:\-]?\s*(.*)$",
                         re.IGNORECASE)
    for idx, line in enumerate(lines):
        match = trigger.search(line)
        if match:
            candidate = _clean_name(match.group(1)) if match.group(1).strip() else None
            if candidate:
                return candidate
            if idx + 1 < len(lines):               # the name is often alone on the next line
                candidate = _clean_name(lines[idx + 1])
                if candidate:
                    return candidate
    return None


_INSTITUTION_WORDS = re.compile(r"\b(university|institute|institution|college|school|academy|board|polytechnic)\b", re.IGNORECASE)


def _extract_institution(lines):
    """Pick the first headline-like line that contains an institution keyword."""
    for line in lines[:max(len(lines), 1)]:
        low = line.lower()
        if _INSTITUTION_WORDS.search(line) and not re.search(r"certify|conferred|awarded|this is", low):
            cleaned = _clean(re.sub(r"^(?:name\s+of\s+the\s+)?(?:institution|institute|university|college)\s*[:\-]\s*", "", line, flags=re.IGNORECASE))
            if len(cleaned) >= 6:
                return cleaned
    return None


# Long degree words are case-insensitive; very short forms (B.E, M.A ...) must be written with dots
# so ordinary words such as "be" or "me" are not mistaken for degrees.
_COURSE_KEYWORDS = (r"(?:(?i:\b(?:bachelor|master|diploma|b\.?\s?tech|m\.?\s?tech|b\.?\s?sc|m\.?\s?sc|"
                    r"b\.?\s?com|m\.?\s?com|mba|bca|mca)\b)|\b(?:B\.E|M\.E|B\.A|M\.A)\b)")


def _extract_course(lines):
    # 1) labels
    value = _value_after_label(lines, r"(?:course|degree|programme|program|branch|qualification)(?:\s*name)?",
                               r"([A-Za-z][A-Za-z.,&()\s-]{5,})")
    if value:
        return _clean(value)
    # 2) any line that starts a degree name
    for line in lines:
        match = re.search(_COURSE_KEYWORDS + r"[A-Za-z.,&()\s-]*", line)
        if match and len(match.group(0).strip()) >= 5:
            text = _clean(line[match.start():])
            text = re.split(r"\b(?:in the year|during|held in|with)\b", text, flags=re.IGNORECASE)[0]
            return _clean(text)
    return None


_YEAR = r"((?:19[89]\d|20[0-3]\d))"


def _extract_year(lines):
    # 1) labelled: "Year of Passing : 2025", "Passing Year", "Year : 2025"
    value = _value_after_label(
        lines, r"(?:year\s+of\s+(?:passing|pass(?:ed)?\s*out|completion|graduation)|passing\s+year|pass\s*out\s+year|year)",
        _YEAR)
    if value:
        return value
    # 2) lines that talk about passing / examination / session / convocation
    for line in lines:
        if re.search(r"passing|passed|examination|session|convocation|completed|graduat", line, re.IGNORECASE):
            match = re.search(r"\b" + _YEAR + r"\b", line)
            if match:
                return match.group(1)
    return None


def extract_fields(text: str) -> dict:
    """Run every extractor and return a dict with value-or-None for each field."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    fields = {key: None for key in FIELD_KEYS}
    if not lines:
        return fields
    fields.update(_extract_ids(lines))
    fields["student_name"] = _extract_name(lines)
    fields["institution"] = _extract_institution(lines)
    fields["course"] = _extract_course(lines)
    fields["year"] = _extract_year(lines)
    return fields


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def run_ocr(image_bgr: np.ndarray, pdf_text: str = "") -> dict:
    """
    Extract text and certificate fields. Never raises.

    Returns dict(status, text, lines, fields, confidence, method, message)
      status: 'OCR Completed' | 'OCR Partially Completed' | 'OCR Failed'
    """
    out = {"status": "OCR Failed", "text": "", "lines": [], "fields": {k: None for k in FIELD_KEYS},
           "confidence": None, "method": "", "message": ""}
    try:
        if pdf_text and len(pdf_text.strip()) >= 30:
            # Digital PDFs already contain real text - reading it directly is exact and fast.
            lines = [l.strip() for l in pdf_text.splitlines() if l.strip()]
            out.update(lines=lines, method="PDF text layer")
        else:
            lines, conf = _run_easyocr(image_bgr)
            out.update(lines=lines, confidence=conf, method="EasyOCR")
    except ImportError:
        out["message"] = "EasyOCR is not installed. Run: pip install easyocr"
        return out
    except Exception as exc:  # model download problems, memory errors, etc.
        out["message"] = f"OCR engine error: {type(exc).__name__}. Please try again or use a clearer image."
        return out

    text = "\n".join(out["lines"])
    out["text"] = text
    if len(text.strip()) < 15:
        out["message"] = "No readable text was detected. Try a sharper, well-lit image."
        return out

    out["fields"] = extract_fields(text)
    detected = sum(1 for k in CORE_FIELDS if out["fields"].get(k))
    if detected >= 4:
        out["status"] = "OCR Completed"
    elif detected >= 1 or len(text.strip()) >= 60:
        out["status"] = "OCR Partially Completed"
        out["message"] = "Some certificate fields could not be detected."
    else:
        out["status"] = "OCR Failed"
        out["message"] = "Text was found but no certificate details could be identified."
    return out
