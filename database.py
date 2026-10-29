"""
database.py - SQLite storage for CertifyAI.

Two tables
  certificates          : the "genuine records" we verify against
  verification_history  : one row for every completed verification

HOW DATABASE MATCHING WORKS
  1. FIND a candidate record using identifiers (certificate ID, register
     number, certificate number). IDs are normalised first so harmless OCR
     differences (case, dashes, O/0, I/1) do not stop the search.
  2. COMPARE every field one by one (name, register number, institution,
     course, year ...). Names/institutions use a similarity score with
     strict thresholds - we never accept a match just because "something"
     looks similar.
  3. Turn the field results into one status:
     Verified / Mismatch / No Record Found / Insufficient Information.
     "No Record Found" only means the database has no entry - NOT that the
     certificate is fake.
"""

import sqlite3
from datetime import datetime

import pandas as pd

from utils import (DB_PATH, normalize_id, normalize_name, normalize_text,
                   similarity, tokens_found_fraction)

# --------------------------------------------------------------------------
# DEMO data - clearly NOT real university records (used for project review)
# --------------------------------------------------------------------------
DEMO_REMARK = "DEMO RECORD - sample data for project demonstration, not a real university record"

DEMO_RECORDS = [
    # certificate_id, student_name, register_number, institution, course, year, certificate_number
    ("DEMO-CERT-2025-001", "Aarav Sharma", "DEMO21AD001", "Demo Institute of Technology",
     "B.Tech Artificial Intelligence and Data Science", 2025, "CN-DEMO-0001"),
    ("DEMO-CERT-2025-002", "Meera Nair", "DEMO21CS002", "Demo Institute of Technology",
     "B.Tech Computer Science and Engineering", 2025, "CN-DEMO-0002"),
    ("DEMO-CERT-2024-003", "Rohan Verma", "DEMO20EC003", "Demo Institute of Technology",
     "B.E. Electronics and Communication Engineering", 2024, "CN-DEMO-0003"),
    ("DEMO-CERT-2024-004", "Priya Iyer", "DEMO20ME004", "Sample State University",
     "B.Tech Mechanical Engineering", 2024, "CN-DEMO-0004"),
    ("DEMO-CERT-2023-005", "Karthik Raj", "DEMO19CE005", "Sample State University",
     "B.Tech Civil Engineering", 2023, "CN-DEMO-0005"),
]

RECORD_COLUMNS = ["certificate_id", "student_name", "register_number", "institution",
                  "course", "year", "certificate_number"]


def get_connection() -> sqlite3.Connection:
    """Open a new connection (one per call keeps Streamlit reruns safe)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# --------------------------------------------------------------------------
# Initialisation
# --------------------------------------------------------------------------
def initialize_database():
    """Create tables (if missing) and add the demo records. Returns (ok, message)."""
    try:
        with get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS certificates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    certificate_id TEXT UNIQUE NOT NULL,
                    student_name TEXT NOT NULL,
                    register_number TEXT,
                    institution TEXT,
                    course TEXT,
                    year INTEGER,
                    certificate_number TEXT,
                    certificate_id_key TEXT,      -- normalised copies used for searching
                    register_number_key TEXT,
                    certificate_number_key TEXT,
                    remarks TEXT
                )""")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS verification_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_name TEXT,
                    certificate_id TEXT,
                    student_name TEXT,
                    timestamp TEXT,
                    ocr_status TEXT,
                    database_status TEXT,
                    qr_status TEXT,
                    tamper_status TEXT,
                    tamper_risk_score REAL,
                    final_status TEXT
                )""")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cert_key ON certificates(certificate_id_key)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reg_key ON certificates(register_number_key)")
        for rec in DEMO_RECORDS:
            insert_certificate(dict(zip(RECORD_COLUMNS, rec)), remarks=DEMO_REMARK)
        return True, "Database ready."
    except sqlite3.Error as exc:
        return False, f"Database could not be initialised: {exc}"


def insert_certificate(record: dict, remarks: str = "") -> bool:
    """Insert one genuine certificate record. Existing certificate IDs are skipped."""
    try:
        with get_connection() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO certificates
                   (certificate_id, student_name, register_number, institution, course, year,
                    certificate_number, certificate_id_key, register_number_key,
                    certificate_number_key, remarks)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (record["certificate_id"], record["student_name"], record.get("register_number"),
                 record.get("institution"), record.get("course"), record.get("year"),
                 record.get("certificate_number"),
                 normalize_id(record["certificate_id"]),
                 normalize_id(record.get("register_number", "")),
                 normalize_id(record.get("certificate_number", "")),
                 remarks))
        return True
    except (sqlite3.Error, KeyError):
        return False


# --------------------------------------------------------------------------
# Lookups
# --------------------------------------------------------------------------
def _fetch_one(column: str, key: str):
    if not key:
        return None
    try:
        with get_connection() as conn:
            row = conn.execute(f"SELECT * FROM certificates WHERE {column} = ?", (key,)).fetchone()
        return dict(row) if row else None
    except sqlite3.Error:
        return None


def get_certificate_by_id(certificate_id: str):
    """Search using certificate ID (normalised)."""
    return _fetch_one("certificate_id_key", normalize_id(certificate_id))


def get_certificate_by_register_number(register_number: str):
    """Search using register number (normalised)."""
    return _fetch_one("register_number_key", normalize_id(register_number))


def get_certificate_by_certificate_number(certificate_number: str):
    return _fetch_one("certificate_number_key", normalize_id(certificate_number))


def find_record_in_text(full_text: str):
    """
    Fallback when regex could not isolate an ID: check whether any database
    certificate ID / register number appears inside the raw OCR text.
    Keys shorter than 8 characters are ignored to avoid accidental hits.
    """
    text_key = normalize_id(full_text)
    if not text_key:
        return None, ""
    try:
        with get_connection() as conn:
            rows = conn.execute("SELECT * FROM certificates").fetchall()
    except sqlite3.Error:
        return None, ""
    for row in rows:
        for column, label in (("certificate_id_key", "certificate ID found in text"),
                              ("register_number_key", "register number found in text")):
            key = row[column] or ""
            if len(key) >= 8 and key in text_key:
                return dict(row), label
    return None, ""


def get_all_certificates() -> pd.DataFrame:
    try:
        with get_connection() as conn:
            return pd.read_sql_query(
                "SELECT certificate_id, student_name, register_number, institution, course, year, "
                "certificate_number, remarks FROM certificates", conn)
    except Exception:
        return pd.DataFrame()


# --------------------------------------------------------------------------
# Comparison: OCR information  vs  database record
# --------------------------------------------------------------------------
ID_FIELDS = ("certificate_id", "register_number", "certificate_number")
CRITICAL_FIELDS = ("certificate_id", "student_name", "register_number", "certificate_number", "year")
SOFT_FIELDS = ("institution", "course")
FIELD_LABELS = {
    "certificate_id": "Certificate ID", "student_name": "Student Name",
    "register_number": "Register Number", "certificate_number": "Certificate Number",
    "institution": "Institution", "course": "Course / Degree", "year": "Year of Passing",
}


def _compare_id(extracted: str, db_value: str, text: str) -> str:
    """IDs must be equal after normalisation (case / punctuation / O-0 / I-1 ignored)."""
    db_key = normalize_id(db_value)
    if extracted and normalize_id(extracted) == db_key:
        return "Match"
    # the regex may have grabbed extra characters - accept if the exact ID is in the raw text
    if db_key and len(db_key) >= 6 and db_key in normalize_id(text):
        return "Match"
    return "Mismatch" if extracted else "Not detected"


def _compare_name(extracted: str, db_value: str, text: str) -> str:
    """Names: order-insensitive similarity with strict thresholds."""
    if extracted:
        score = similarity(normalize_name(extracted), normalize_name(db_value))
        if score >= 0.93:
            return "Match"
        if score >= 0.82:
            return "Partial Match"
    # every word of the database name must appear in the OCR text
    if tokens_found_fraction(db_value, text) == 1.0:
        return "Match"
    return "Mismatch" if extracted else "Not detected"


def _compare_descriptive(extracted: str, db_value: str, text: str) -> str:
    """Institution / course: similarity of the field, or most words found in the text."""
    if extracted:
        score = similarity(normalize_text(extracted), normalize_text(db_value))
        if score >= 0.85:
            return "Match"
        if score >= 0.70:
            return "Partial Match"
    fraction = tokens_found_fraction(db_value, text)
    if fraction >= 0.9:
        return "Match"
    if fraction >= 0.7:
        return "Partial Match"
    return "Mismatch" if extracted else "Not detected"


def _compare_year(extracted: str, db_value) -> str:
    """Year: compare the detected year only (years also appear inside IDs, so no text search)."""
    if not extracted:
        return "Not detected"
    return "Match" if str(extracted).strip() == str(db_value).strip() else "Mismatch"


def compare_with_database(fields: dict, full_text: str) -> dict:
    """
    Main database verification function.

    fields    : dict from ocr.extract_fields() (values or None)
    full_text : the complete OCR text (used as a fallback source of evidence)
    """
    result = {"status": "Insufficient Information", "record": None, "fields": [],
              "mismatched_fields": [], "lookup_method": "", "message": ""}

    # ---- Step 1: find a candidate record ---------------------------------
    record, method = None, ""
    for field_name, getter, label in (
            ("certificate_id", get_certificate_by_id, "certificate ID"),
            ("register_number", get_certificate_by_register_number, "register number"),
            ("certificate_number", get_certificate_by_certificate_number, "certificate number")):
        value = fields.get(field_name)
        if value:
            record = getter(value)
            if record:
                method = f"Searched by {label}"
                break
    if record is None:
        record, found_by = find_record_in_text(full_text)
        method = f"Searched raw OCR text ({found_by})" if record else ""

    has_identifier = any(fields.get(f) for f in ID_FIELDS)
    if record is None:
        if has_identifier:
            result.update(status="No Record Found",
                          message="No database record matches the detected identifiers. "
                                  "This is not proof of a fake certificate.")
        else:
            result.update(message="No certificate ID, register number or certificate number was detected, "
                                  "so the database could not be searched.")
        return result

    # ---- Step 2: compare field by field ----------------------------------
    rows = []
    for field_name in RECORD_COLUMNS:
        extracted = fields.get(field_name)
        db_value = record.get(field_name)
        if db_value in (None, ""):
            outcome = "Not in database"
        elif field_name in ID_FIELDS:
            outcome = _compare_id(extracted, db_value, full_text)
        elif field_name == "student_name":
            outcome = _compare_name(extracted, db_value, full_text)
        elif field_name == "year":
            outcome = _compare_year(extracted, db_value)
        else:
            outcome = _compare_descriptive(extracted, db_value, full_text)
        rows.append({"field": field_name, "label": FIELD_LABELS[field_name],
                     "certificate_value": extracted or "Not detected",
                     "database_value": "" if db_value is None else str(db_value),
                     "result": outcome})

    # ---- Step 3: decide the database status ------------------------------
    outcome_of = {r["field"]: r["result"] for r in rows}
    critical_bad = [f for f in CRITICAL_FIELDS if outcome_of.get(f) == "Mismatch"]
    soft_bad = [f for f in SOFT_FIELDS if outcome_of.get(f) == "Mismatch"]
    confirmed = sum(1 for r in rows if r["result"] in ("Match", "Partial Match"))
    name_confirmed = outcome_of.get("student_name") in ("Match", "Partial Match")

    mismatched = critical_bad + (soft_bad if len(soft_bad) == len(SOFT_FIELDS) else [])
    if mismatched:
        status = "Mismatch"
        message = "A record was found but some fields conflict with the certificate."
    elif name_confirmed and confirmed >= 3:
        status = "Verified"
        message = "The certificate details agree with the database record."
        if soft_bad:
            message += f" (Minor difference in {FIELD_LABELS[soft_bad[0]].lower()} - possibly an OCR reading error.)"
    else:
        status = "Insufficient Information"
        message = "A record was found, but too few fields (or the student name) could be confirmed."

    result.update(status=status, record=record, fields=rows, message=message, lookup_method=method,
                  mismatched_fields=[FIELD_LABELS[f] for f in mismatched])
    return result


# --------------------------------------------------------------------------
# Verification history
# --------------------------------------------------------------------------
HISTORY_COLUMNS = ["id", "file_name", "certificate_id", "student_name", "timestamp", "ocr_status",
                   "database_status", "qr_status", "tamper_status", "tamper_risk_score", "final_status"]


def add_history(file_name, certificate_id, student_name, ocr_status, database_status,
                qr_status, tamper_status, tamper_risk_score, final_status):
    """Save one completed verification. Returns True when saved."""
    try:
        with get_connection() as conn:
            conn.execute(
                """INSERT INTO verification_history
                   (file_name, certificate_id, student_name, timestamp, ocr_status, database_status,
                    qr_status, tamper_status, tamper_risk_score, final_status)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (file_name, certificate_id or "Not detected", student_name or "Not detected",
                 datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ocr_status, database_status,
                 qr_status, tamper_status, tamper_risk_score, final_status))
        return True
    except sqlite3.Error:
        return False


def get_history() -> pd.DataFrame:
    """All history rows, newest first (empty DataFrame when none / on error)."""
    try:
        with get_connection() as conn:
            df = pd.read_sql_query("SELECT * FROM verification_history ORDER BY id DESC", conn)
        return df
    except Exception:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
