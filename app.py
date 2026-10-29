"""
app.py - CertifyAI: AI-Based Fake Certificate Detection and Verification System

Run with:  streamlit run app.py      (or: python -m streamlit run app.py)

This file only contains the user interface and the glue code. The real work
is done in the other modules:
    ocr.py              -> read text and certificate fields
    database.py         -> SQLite records, comparison, history
    qr_verify.py        -> QR detection and comparison
    tamper_detection.py -> AI-assisted image tampering analysis
    utils.py            -> file loading + the final decision engine
"""

import html
from datetime import datetime

import streamlit as st

from database import add_history, compare_with_database, get_history, initialize_database
from ocr import run_ocr
from qr_verify import verify_qr
from tamper_detection import analyze_tampering
from utils import (ASSETS_DIR, DISCLAIMER, FINAL_SUSPICIOUS, FINAL_UNVERIFIED, FINAL_VERIFIED,
                   UPLOAD_DIR, decide_final_status, file_hash, format_file_size, load_certificate,
                   safe_filename)

st.set_page_config(page_title="CertifyAI - Certificate Verification", page_icon="🛡️", layout="wide")

PAGES = ["Home", "Verify Certificate", "Verification History", "About System"]
FIELD_LABELS = {"student_name": "Student Name", "register_number": "Register Number",
                "certificate_id": "Certificate ID", "certificate_number": "Certificate Number",
                "institution": "Institution Name", "course": "Course / Degree", "year": "Year of Passing"}

# --------------------------------------------------------------------------
# Small HTML helpers (icons are inline SVG so no extra packages are needed)
# --------------------------------------------------------------------------
ICONS = {
    "ocr": '<path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/><path d="M7 8h8"/><path d="M7 12h10"/><path d="M7 16h6"/>',
    "qr": '<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><path d="M14 14h3v3h-3z"/><path d="M20 14v3"/><path d="M14 20h3"/><path d="M20 20h1"/>',
    "db": '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
    "ai": '<rect x="5" y="5" width="14" height="14" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3"/>',
    "shield": '<path d="M12 3l8 3v6c0 4.5-3.2 7.8-8 9-4.8-1.2-8-4.5-8-9V6z"/><path d="M9 12l2 2 4-4"/>',
}


def icon(name: str) -> str:
    return f'<div class="icon-badge"><svg viewBox="0 0 24 24">{ICONS[name]}</svg></div>'


def show_html(markup: str):
    """Render HTML. Lines are joined so Markdown never mistakes indented tags for code."""
    st.markdown("".join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True)


def esc(value) -> str:
    return html.escape("" if value is None else str(value))


def chip(text: str, level: str) -> str:
    return f'<span class="chip {level}">{esc(text)}</span>'


def table_html(headers: list, rows: list) -> str:
    """rows: list of lists of ready-made HTML strings."""
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f'<div class="table-wrap"><table class="cmp"><tr>{head}</tr>{body}</table></div>'


def section(title: str):
    show_html(f'<div class="section-title">{esc(title)}</div>')


# --------------------------------------------------------------------------
# Status -> colour level
# --------------------------------------------------------------------------
def level_ocr(status): return {"OCR Completed": "ok", "OCR Partially Completed": "warn"}.get(status, "bad")
def level_db(status): return {"Verified": "ok", "Mismatch": "bad"}.get(status, "warn")
def level_qr(status): return {"QR Verified": "ok", "QR Mismatch": "bad", "QR Unreadable": "warn"}.get(status, "info")
def level_tamper(status): return {"No obvious tampering detected": "ok", "Potential tampering detected": "bad"}.get(status, "info")
def level_match(result): return {"Match": "ok", "Partial Match": "warn", "Mismatch": "bad"}.get(result, "info")


# --------------------------------------------------------------------------
# Setup: CSS, database, navigation state
# --------------------------------------------------------------------------
def load_css():
    css_file = ASSETS_DIR / "style.css"
    if css_file.exists():
        st.markdown(f"<style>{css_file.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def setup_database():
    """Create tables + demo records once per browser session."""
    if "db_ok" not in st.session_state:
        ok, message = initialize_database()
        st.session_state["db_ok"], st.session_state["db_message"] = ok, message
    if not st.session_state["db_ok"]:
        st.warning(f"{st.session_state['db_message']} Verification history may not be saved.")


def go_to(page: str):
    """Button callback: switch page (callbacks may change the nav widget's state)."""
    st.session_state["nav_page"] = page


# --------------------------------------------------------------------------
# HOME
# --------------------------------------------------------------------------
def page_home():
    show_html("""
    <div class="hero">
      <div class="brand-pill">🛡️ CertifyAI</div>
      <h1><span class="gradient-text">AI-Based Fake Certificate Detection<br>and Verification System</span></h1>
      <div class="subtitle">AI-Powered Certificate Intelligence</div>
      <p class="lead">CertifyAI performs multi-layer verification of educational certificates. Instead of trusting a
      single test, it combines OCR text extraction, a certificate database check, QR-code validation and an
      AI-assisted image tampering analysis, then explains exactly why it reached its verdict.</p>
    </div>""")

    cards = [("ocr", "OCR Extraction", "Reads the certificate text and finds name, register number, certificate ID, institution, course and year."),
             ("qr", "QR Verification", "Detects and decodes QR codes and compares their content with the certificate and database."),
             ("db", "Database Verification", "Compares every extracted field with genuine records stored in SQLite."),
             ("ai", "AI Tamper Detection", "Forensic image features (ELA, noise, edges) highlight potentially suspicious regions.")]
    cols = st.columns(4)
    for col, (ic, title, text) in zip(cols, cards):
        with col:
            show_html(f'<div class="glass feature-card hover-lift">{icon(ic)}<h4>{title}</h4><p>{text}</p></div>')

    st.write("")
    _, mid, _ = st.columns([1, 1, 1])
    with mid:
        st.button("Verify Certificate", key="home_cta", on_click=go_to, args=("Verify Certificate"))

    st.write("")
    show_html('<div class="notice"><b>Academic prototype.</b> Results are verification evidence, not legal proof of authenticity. '
              'The built-in database contains clearly labelled DEMO records only.</div>')


# --------------------------------------------------------------------------
# VERIFY: pipeline
# --------------------------------------------------------------------------
def save_upload(data: bytes, filename: str):
    """Keep a copy of the uploaded file in uploads/ (ignored by Git). Failure is harmless."""
    try:
        UPLOAD_DIR.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        (UPLOAD_DIR / f"{stamp}_{safe_filename(filename)}").write_bytes(data)
    except OSError:
        pass


def run_verification(loaded, filename: str) -> dict:
    """Run all four checks, take the final decision and store it in the history table."""
    ocr = run_ocr(loaded.image_bgr, loaded.pdf_text)

    if ocr["text"].strip():
        try:
            db = compare_with_database(ocr["fields"], ocr["text"])
        except Exception:
            db = {"status": "Insufficient Information", "record": None, "fields": [], "mismatched_fields": [],
                  "lookup_method": "", "message": "The database comparison could not be completed."}
    else:
        db = {"status": "Insufficient Information", "record": None, "fields": [], "mismatched_fields": [],
              "lookup_method": "", "message": "No text was extracted, so the database could not be searched."}

    qr = verify_qr(loaded.image_bgr, ocr["fields"], db.get("record"), ocr["text"])
    tamper = analyze_tampering(loaded.image_bgr)
    decision = decide_final_status(ocr, db, qr, tamper)

    fields = ocr["fields"]
    certificate_id = fields.get("certificate_id") or qr["qr_fields"].get("certificate_id")
    student_name = fields.get("student_name") or qr["qr_fields"].get("student_name")
    saved = add_history(filename, certificate_id, student_name, ocr["status"], db["status"], qr["status"],
                        tamper["status"], tamper["risk_score"], decision["final_status"])
    return {"ocr": ocr, "db": db, "qr": qr, "tamper": tamper, "decision": decision, "history_saved": saved}


# --------------------------------------------------------------------------
# VERIFY: result dashboard
# --------------------------------------------------------------------------
def render_status_cards(result: dict):
    items = [("ocr", "OCR", result["ocr"]["status"], level_ocr(result["ocr"]["status"])),
             ("db", "Database", result["db"]["status"], level_db(result["db"]["status"])),
             ("qr", "QR Code", result["qr"]["status"], level_qr(result["qr"]["status"])),
             ("ai", "AI Tamper Analysis", result["tamper"]["status"], level_tamper(result["tamper"]["status"]))]
    for col, (ic, tag, status, level) in zip(st.columns(4), items):
        with col:
            show_html(f'<div class="glass status-card {level}">{icon(ic)}<div class="tag">{tag}</div>'
                      f'<div class="value">{esc(status)}</div></div>')


def render_final_card(decision: dict):
    show_html(f'<div class="final-card {decision["level"]}"><div class="small">Final verification result</div>'
              f'<div class="headline">{esc(decision["final_status"])}</div>'
              f'<div class="summary">{esc(decision["summary"])}</div></div>')


def render_reasons(decision: dict):
    symbols = {"ok": "✓", "warn": "!", "bad": "✕", "info": "i"}
    rows = "".join(f'<div class="reason {lvl}"><span class="dot">{symbols[lvl]}</span><span>{esc(text)}</span></div>'
                   for lvl, text in decision["reasons"])
    show_html(f'<div class="glass">{rows}</div>')


def render_extracted(ocr: dict):
    rows = []
    for key in ("student_name", "register_number", "certificate_id", "certificate_number", "institution", "course", "year"):
        value = ocr["fields"].get(key)
        rows.append([esc(FIELD_LABELS[key]), esc(value) if value else '<span class="muted">Not detected</span>'])
    show_html(f'<div class="glass">{table_html(["Field", "Detected value"], rows)}</div>')
    method = ocr["method"] or "n/a"
    conf = f" | average OCR confidence {ocr['confidence'] * 100:.0f}%" if ocr["confidence"] is not None else ""
    st.caption(f"Extraction method: {method}{conf}. {ocr['message']}")
    with st.expander("Extracted Text (raw OCR output)"):
        st.text_area("Extracted Text", ocr["text"] or "(no text extracted)", height=220, disabled=True, label_visibility="collapsed")


def render_database(db: dict):
    status = db["status"]
    head = f'{chip(status, level_db(status))} <span class="muted">&nbsp;{esc(db["message"])}</span>'
    body = ""
    if db["fields"]:
        rows = [[esc(r["label"]), esc(r["certificate_value"]), esc(r["database_value"]), chip(r["result"], level_match(r["result"]))]
                for r in db["fields"]]
        body = table_html(["Field", "On certificate (OCR)", "Database record", "Comparison"], rows)
        remark = db["record"].get("remarks") if db["record"] else ""
        body += f'<p class="muted" style="margin:.8rem 0 0">{esc(db["lookup_method"])}. {esc(remark)}</p>'
    show_html(f'<div class="glass"><p style="margin-top:0">{head}</p>{body}</div>')


def render_qr(qr: dict):
    status = qr["status"]
    show_html(f'<div class="glass"><p style="margin:0">{chip(status, level_qr(status))} '
              f'<span class="muted">&nbsp;{esc(qr["message"])}</span></p></div>')
    if qr["raw"]:
        st.write("")
        st.caption(f"QR content type: {qr['kind']}")
        st.code(qr["raw"], language=None)
        if qr["comparisons"]:
            rows = [[esc(FIELD_LABELS.get(c["field"], c["field"])), esc(c["qr_value"]), esc(c["reference_value"]),
                     chip(c["result"], level_match(c["result"]))] for c in qr["comparisons"]]
            show_html(f'<div class="glass">{table_html(["Field", "In QR code", "Compared with", "Result"], rows)}</div>')


def render_tamper(tamper: dict, original):
    status = tamper["status"]
    score = tamper["risk_score"]
    score_text = f"{score:.0f} / 100" if score is not None else "Not available"
    notes = "".join(f"<li>{esc(n)}</li>" for n in tamper["indicators"]) or "<li>No analysis details available.</li>"
    extra = f'<p class="muted" style="margin:.4rem 0 0">{esc(tamper["message"])}</p>' if tamper["message"] else ""
    show_html(f'<div class="glass"><p style="margin-top:0">{chip(status, level_tamper(status))}</p>'
              f'<div class="muted" style="font-size:.8rem;letter-spacing:.1em;text-transform:uppercase">Tampering Risk Score</div>'
              f'<div style="font-family:Poppins;font-weight:800;font-size:2rem">{score_text}</div>'
              f'<p class="muted" style="margin:.2rem 0 .8rem">Method: {esc(tamper["mode_label"] or "n/a")}</p>'
              f'<ul style="margin:0 0 .4rem 1.1rem;line-height:1.6">{notes}</ul>{extra}</div>')
    if tamper["heatmap_rgb"] is not None:
        st.write("")
        left, right = st.columns(2)
        with left:
            st.image(original, caption="Original Certificate")
        with right:
            st.image(tamper["heatmap_rgb"], caption="Tampering Analysis - potentially suspicious regions (magenta boxes)")
    st.caption("Highlighted areas only show unusual image statistics; they are not proof of forgery. " + DISCLAIMER)


def render_results(result: dict, original):
    show_html('<div class="section-title">Verification Dashboard</div>')
    render_status_cards(result)
    render_final_card(result["decision"])
    if not result["history_saved"]:
        st.warning("The result could not be saved to the verification history.")

    section("Verification Summary")
    show_html(f'<div class="glass">{esc(result["decision"]["summary"])}<br><br><span class="muted">{esc(DISCLAIMER)}</span></div>')
    section("Reasons for Decision")
    render_reasons(result["decision"])
    section("Extracted Certificate Details")
    render_extracted(result["ocr"])
    section("Available Database Comparison")
    render_database(result["db"])
    section("QR Details")
    render_qr(result["qr"])
    section("Tampering Analysis")
    render_tamper(result["tamper"], original)


# --------------------------------------------------------------------------
# VERIFY page
# --------------------------------------------------------------------------
def page_verify():
    show_html('<div class="section-title" style="margin-top:.4rem">Verify Certificate</div>')
    st.markdown('<span class="muted">Upload a certificate (JPG, JPEG, PNG or PDF). Nothing is judged until you press '
                '<b>Analyze Certificate</b>.</span>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload certificate", type=["jpg", "jpeg", "png", "pdf"], label_visibility="collapsed")

    if uploaded is None:
        st.session_state.pop("result", None)
        st.session_state.pop("loaded", None)
        show_html('<div class="glass" style="margin-top:1rem"><b>Tip:</b> try the demo files in the <code>sample_certificates</code> '
                  'folder of this project.</div>')
        return

    data = uploaded.getvalue()
    digest = file_hash(data)

    # open the file once per upload (keeps the app fast when widgets rerun)
    if st.session_state.get("loaded", (None, None))[0] != digest:
        st.session_state["loaded"] = (digest, load_certificate(data, uploaded.name))
        st.session_state.pop("result", None)
    loaded = st.session_state["loaded"][1]

    if not loaded.ok:
        st.error(loaded.error)
        return

    show_html(f'<div class="glass" style="margin-top:1rem"><div class="file-info"><span>File: <b>{esc(uploaded.name)}</b></span>'
              f'<span>Size: <b>{format_file_size(len(data))}</b></span>'
              f'<span>Type: <b>{"PDF (first page)" if loaded.is_pdf else "Image"}</b></span></div></div>')
    for note in loaded.notes:
        st.info(note)

    left, right = st.columns([3, 2])
    with left:
        st.image(loaded.image_rgb, caption="Certificate preview")
    with right:
        show_html('<div class="glass"><h3>Ready to analyse</h3><p class="muted">The system will run OCR, search the '
                  'certificate database, look for a QR code and run the AI-assisted tampering analysis.</p>'
                  '<p class="muted" style="font-size:.85rem">The first analysis can take a minute because the OCR model '
                  'is loaded.</p></div>')
        st.write("")
        analyze = st.button("Analyze Certificate", key="analyze")

    if analyze:
        save_upload(data, uploaded.name)
        with st.spinner("Analysing certificate: OCR, database, QR and tampering checks..."):
            try:
                st.session_state["result"] = run_verification(loaded, uploaded.name)
            except Exception:
                st.session_state.pop("result", None)
                st.error("Something went wrong while analysing this file. Please try another image or a clearer scan.")

    if "result" in st.session_state:
        render_results(st.session_state["result"], loaded.image_rgb)


# --------------------------------------------------------------------------
# HISTORY page
# --------------------------------------------------------------------------
def page_history():
    show_html('<div class="section-title" style="margin-top:.4rem">Verification History</div>')
    df = get_history()
    total = len(df)
    counts = {FINAL_VERIFIED: 0, FINAL_SUSPICIOUS: 0, FINAL_UNVERIFIED: 0}
    if total:
        for key in counts:
            counts[key] = int((df["final_status"] == key).sum())

    metrics = [("Total Verifications", total, "#ffffff"), ("Verified", counts[FINAL_VERIFIED], "var(--ok)"),
               ("Suspicious", counts[FINAL_SUSPICIOUS], "var(--bad)"), ("Unverified", counts[FINAL_UNVERIFIED], "var(--warn)")]
    for col, (label, number, colour) in zip(st.columns(4), metrics):
        with col:
            show_html(f'<div class="glass metric-card hover-lift"><div class="num" style="color:{colour}">{number}</div>'
                      f'<div class="lbl">{label}</div></div>')

    st.write("")
    if df.empty:
        st.info("No verifications yet. Analyse a certificate on the Verify Certificate page and it will appear here.")
        return

    f1, f2 = st.columns([2, 3])
    with f1:
        chosen = st.multiselect("Filter by final status", [FINAL_VERIFIED, FINAL_SUSPICIOUS, FINAL_UNVERIFIED])
    with f2:
        query = st.text_input("Search file name, certificate ID or student name")

    view = df.copy()
    if chosen:
        view = view[view["final_status"].isin(chosen)]
    if query.strip():
        q = query.strip().lower()
        mask = (view["file_name"].astype(str).str.lower().str.contains(q, regex=False)
                | view["certificate_id"].astype(str).str.lower().str.contains(q, regex=False)
                | view["student_name"].astype(str).str.lower().str.contains(q, regex=False))
        view = view[mask]

    st.dataframe(view, width="stretch", hide_index=True,
                 column_config={"tamper_risk_score": st.column_config.NumberColumn("tamper_risk_score", format="%.1f")})
    st.caption(f"Showing {len(view)} of {total} records.")
    st.download_button("Download history (CSV)", view.to_csv(index=False).encode("utf-8"), "verification_history.csv", "text/csv")


# --------------------------------------------------------------------------
# ABOUT page
# --------------------------------------------------------------------------
WORKFLOW = ["Upload Certificate", "OCR Extraction", "Information Extraction", "Database Verification", "QR Verification",
            "AI/ML Tampering Analysis", "Decision Engine", "Verification Result", "History Storage"]


def page_about():
    show_html('<div class="section-title" style="margin-top:.4rem">About System</div>')
    left, right = st.columns(2)
    with left:
        show_html('<div class="glass"><h3>Project Title</h3><p>AI-Based Fake Certificate Detection and Verification System '
                  '(CertifyAI)</p><h3>Purpose</h3><p class="muted">To help reviewers check educational certificates using several independent '
                  'pieces of evidence, and to explain the reasoning behind each result. A missing QR code or a missing database '
                  'record is never treated as proof of forgery.</p>'
                  '<h3>Technologies</h3><p>'
                  + "".join(chip(t, "info") + " " for t in ["Python 3.11+", "Streamlit", "EasyOCR", "OpenCV", "Pillow", "SQLite",
                                                            "scikit-learn", "Pandas", "NumPy", "PyMuPDF"])
                  + '</p><h3>Academic Prototype</h3><p class="muted">Developed as a third-year B.Tech Artificial Intelligence and Data '
                    'Science mini-project, inspired by the IEEE Access (2025) ElimuChain paper on blockchain-based qualification '
                    'verification. CertifyAI uses a different approach: OCR, QR, database and image-forensics checks.</p></div>')
    with right:
        steps = []
        for i, step in enumerate(WORKFLOW, 1):
            steps.append(f'<div class="flow-step"><span class="flow-num">{i}</span><span>{step}</span></div>')
        show_html('<div class="glass"><h3>Verification Workflow</h3>' + '<div class="flow-arrow">↓</div>'.join(steps) + '</div>')

    section("System Limitations")
    show_html('<div class="glass"><ul style="line-height:1.8;margin:0 0 0 1.1rem">'
              '<li>OCR accuracy depends on image quality, fonts and layout; fields may be missed or misread.</li>'
              '<li>The database holds DEMO records only - it is not connected to any real university.</li>'
              '<li>QR codes that only contain a web link cannot be verified offline.</li>'
              '<li>The tamper analysis is a prototype based on image statistics. It can miss edits and can raise '
              'false alarms, and it was not validated on a large forged-certificate dataset.</li>'
              '<li>Results are verification evidence only and are not legal proof of authenticity. Always confirm important '
              'certificates with the issuing institution.</li></ul></div>')


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    load_css()
    setup_database()
    st.session_state.setdefault("nav_page", "Home")
    st.radio("Navigation", PAGES, key="nav_page", horizontal=True, label_visibility="collapsed")
    st.write("")
    page = st.session_state["nav_page"]
    if page == "Home":
        page_home()
    elif page == "Verify Certificate":
        page_verify()
    elif page == "Verification History":
        page_history()
    else:
        page_about()


main()
