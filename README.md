# CertifyAI – AI-Based Fake Certificate Detection and Verification System

**AI-Powered Certificate Intelligence** · Third-year B.Tech Artificial Intelligence & Data Science mini-project (academic prototype)

---

## Overview
CertifyAI is a web application (Python + Streamlit) that checks an educational certificate with **four independent verification layers** and then explains *why* it reached its verdict:

1. **OCR** – reads the text and extracts name, register number, certificate ID, institution, course and year.
2. **Database verification** – compares the extracted fields with genuine records stored in SQLite.
3. **QR verification** – detects/decodes a QR code and compares it with the certificate and database.
4. **AI-assisted tamper analysis** – forensic image features (Error Level Analysis, noise, edges) → risk score + heat-map of potentially suspicious regions.

A missing QR code or a missing database record is **never** treated as proof of a fake certificate.

## Problem Statement
Forged educational certificates are common, and manual verification is slow. A single check (only a database lookup, or only a QR scan) is easy to bypass or may be unavailable for older certificates. We need a transparent system that combines several pieces of evidence.

## Objectives
- Extract certificate information automatically from JPG / PNG / PDF files.
- Verify the information against a certificate database with field-level comparison.
- Validate QR codes when present, without penalising certificates that have none.
- Provide an explainable, ML-assisted image tampering analysis.
- Combine all evidence with transparent decision logic and keep a verification history.

## Features
- Premium dark pink / purple glassmorphism UI (custom CSS), 4 pages: Home, Verify Certificate, Verification History, About System
- JPG / JPEG / PNG / PDF upload, file validation, preview (first page for PDFs)
- OCR with preprocessing (EasyOCR) and flexible regex / keyword field extraction
- SQLite certificate database with clearly labelled DEMO records
- QR detection with several preprocessing attempts; JSON / URL / key-value / plain-ID payloads
- Tamper analysis with feature vector, heuristic risk score or trained scikit-learn model, plus ELA heat-map
- Final decision engine with a written list of reasons
- History dashboard with summary cards, filters and CSV download

## Technology Stack
| Area | Technology |
|---|---|
| Language | Python 3.11+ |
| UI | Streamlit + custom CSS |
| OCR | EasyOCR (no system packages needed) |
| Image processing | OpenCV (headless), Pillow |
| QR | `cv2.QRCodeDetector` |
| Database | SQLite (`sqlite3`) |
| Machine learning | scikit-learn (RandomForest), joblib |
| Data | Pandas, NumPy |
| PDF | PyMuPDF (pure pip install) |

## System Architecture
```
Upload Certificate
      ↓
OCR Extraction  ──►  Information Extraction
      ↓
Database Verification
      ↓
QR Verification
      ↓
AI/ML Tampering Analysis
      ↓
Decision Engine  ──►  Verification Result  ──►  History Storage
```

## Project Structure
```
AI_Fake_Certificate_Detection/
├── app.py                 # Streamlit UI and page logic
├── database.py            # SQLite tables, demo records, field comparison, history
├── ocr.py                 # preprocessing, EasyOCR, field extraction
├── qr_verify.py           # QR detection, parsing, comparison
├── tamper_detection.py    # forensic features, heuristic / model risk score, heat-map
├── train_model.py         # optional: train models/tamper_model.pkl
├── utils.py               # file loading, text normalisation, final decision engine
├── requirements.txt
├── README.md
├── .gitignore
├── certificates.db        # created automatically on first run
├── .streamlit/config.toml # dark theme + upload size
├── assets/style.css       # custom theme
├── uploads/               # copies of analysed files (ignored by Git)
├── sample_certificates/   # DEMO certificates for the review
└── models/                # tamper_model.pkl appears here after training
```

## Installation
```bash
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```
No Tesseract, Poppler or other system software is needed. The first OCR run downloads the EasyOCR language model (about 100 MB) once, so the first analysis is slow.

## How to Run
```bash
streamlit run app.py
# or
python -m streamlit run app.py
```

## Demo Guide (sample certificates)
The folder `sample_certificates/` holds **fictional** DEMO certificates. The database contains matching DEMO records (`DEMO-CERT-2025-001` … `DEMO-CERT-2023-005`). Expected results when OCR reads the files correctly:

| File | What it shows | Expected final result |
|---|---|---|
| `01_genuine_with_qr.png` | DEMO record 1 with a matching QR code | GENUINE / VERIFIED (database + QR agree) |
| `02_genuine_no_qr.jpg` | DEMO record 2, no QR | GENUINE / VERIFIED (missing QR is not a problem) |
| `03_qr_mismatch.png` | Record 1 text, but the QR holds another register number | SUSPICIOUS (QR conflicts with database) |
| `04_database_mismatch.png` | Certificate ID of record 3 with a different student name | SUSPICIOUS (database mismatch) |
| `05_no_database_record.png` | Certificate that is not in the database (QR is a web link) | UNVERIFIED (no record ≠ fake) |
| `06_edited_demo.png` | Record 2 with the name pasted over | SUSPICIOUS (database name mismatch) |

## Database Explanation
Table `certificates`: `certificate_id, student_name, register_number, institution, course, year, certificate_number` (+ normalised search keys and a remarks column). Table `verification_history` stores every completed verification.

Matching works in two steps: (1) **find** a candidate record with the certificate ID, register number or certificate number (normalised: case, punctuation and look-alike characters such as O/0 or I/1 are ignored); (2) **compare field by field** – IDs must be equal after normalisation, names use an order-insensitive similarity with strict thresholds, institution/course use similarity or word coverage. Status: `Verified`, `Mismatch`, `No Record Found`, `Insufficient Information`.

## OCR Explanation
Preprocessing (grayscale → upscale → CLAHE contrast → light denoise, with an adaptive-threshold retry only when the first pass reads little) improves reading of faint scans without destroying thin letters. Lines are rebuilt from word boxes and fields are extracted with labels (`Register No :`), next-line values and sentence patterns (`This is to certify that …`). Fields that cannot be found are shown as **Not detected**. Digital PDFs with a text layer are read directly.

## QR Verification Explanation
`cv2.QRCodeDetector` is tried on the original image and on grayscale, enlarged, sharpened, thresholded and cropped versions. The payload is parsed (JSON, URL query/path, `key: value` text, plain ID) and compared with the database record (or the OCR fields if no record exists). Statuses: QR Verified, QR Mismatch, QR Not Found, QR Unreadable, QR Data Detected – Unable to Fully Verify.

## AI/ML Tamper Detection Explanation
*Error Level Analysis (ELA)* re-saves the image as JPEG and measures how much each pixel changes. We compute 11 features (ELA statistics, block-wise ELA anomalies, noise statistics, edge density, Laplacian and gradient statistics) and:
- if `models/tamper_model.pkl` exists, a **RandomForest** predicts a tamper probability (risk score);
- otherwise a transparent **heuristic score** (weighted combination of the features) is used and clearly labelled as a prototype.

Train the optional model with `python train_model.py` (needs labelled folders `dataset/genuine` and `dataset/tampered`) or `python train_model.py --synthetic` (prototype on simulated edits; its score is **not** real-world accuracy). No accuracy figure is claimed anywhere in the app.

> **Honest note:** on our own small simulated test set the heuristic found only some edits and can miss others; it is a prototype for learning and demonstration. Do not rely on it alone.

## Decision Logic
- **SUSPICIOUS / POTENTIALLY TAMPERED** – strong contradicting evidence: database mismatch, QR conflicting with the record/certificate, or significant tampering indicators (risk score ≥ 60).
- **GENUINE / VERIFIED** – the database record matches field by field and nothing contradicts it (a matching QR strengthens it, a missing QR is neutral).
- **UNVERIFIED / VERIFICATION FAILED** – not enough reliable evidence (OCR failed, no record, missing details) and no strong evidence of tampering.

## Screenshots
_Add screenshots here after running the app:_
- `docs/home.png` – Home page
- `docs/result.png` – Verification dashboard
- `docs/history.png` – History page

## Limitations
- OCR can misread low-quality images or unusual layouts.
- The database contains DEMO records only; it is not linked to any real institution.
- QR codes containing only a web link cannot be verified offline.
- The tamper analysis is a prototype and not validated on a large forged-certificate dataset; it can miss edits and give false alarms.
- Results are verification evidence, not legal proof of authenticity.

## Future Enhancements
- Train on a real labelled dataset of genuine and tampered certificates.
- Online verification of QR links and institution APIs.
- Layout/template and seal/signature checks, multi-language OCR.
- Role-based login and an admin screen to manage genuine records.
- Optional blockchain anchoring of certificate hashes (as in the base paper).

## IEEE Base Paper
This project is **inspired by**:
Said H. Said, Ramadhani S. Sinde, Efraim M. Kosia, Mussa A. Dida, *“A Comprehensive Blockchain-Based System for Educational Qualifications Management and Verification to Counter Forgery,”* **IEEE Access**, vol. 13, pp. 31562–31589, 2025. DOI: [10.1109/ACCESS.2025.3542545](https://doi.org/10.1109/ACCESS.2025.3542545)

The paper proposes **ElimuChain** (blockchain + smart contracts + IPFS). Our implementation is **different**: OCR + QR verification + SQLite database verification + AI/ML-assisted image tampering detection + Streamlit. It does **not** reproduce ElimuChain.

## Deployment Instructions (Streamlit Community Cloud)
1. Create a GitHub repository and push the project (`git init`, `git add .`, `git commit`, `git remote add origin …`, `git push`).
2. Open <https://share.streamlit.io>, sign in with GitHub and click **Create app**.
3. Select the repository, branch and set **Main file path** to `app.py`.
4. Under **Advanced settings** choose **Python 3.11**.
5. Click **Deploy**. The first start installs the packages and the first analysis downloads the OCR model.

Notes: EasyOCR depends on PyTorch, which is large. If the build or memory limit fails, add these lines at the top of `requirements.txt` to install the smaller CPU-only build:
```
--extra-index-url https://download.pytorch.org/whl/cpu
torch
torchvision
```
Streamlit Cloud storage is temporary, so the history table resets when the app is rebuilt. No credentials or API keys are used.
