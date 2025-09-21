Project Description

Resume Relevance Checker is a web-based tool that helps recruiters and hiring managers quickly assess how well candidate resumes match a given Job Description (JD). It automates the tedious process of manually reading resumes and evaluating skills, providing both hard (exact skills match) and soft (semantic similarity) scores.

Why This Project?

Recruiters often face hundreds of resumes for a single role. Manually checking each resume is time-consuming and prone to human error. This project:

Speeds up resume screening.

Standardizes evaluation with quantitative scores.

Highlights missing skills or strengths of candidates.

Works with multiple file formats including PDFs, DOCX, TXT, and images.

Key Features

Text Extraction

Extract text from resumes and job descriptions using PDF, DOCX, TXT, and image files.

Uses OCR fallback for scanned PDFs or images.

Skill Extraction

Detects technical and professional skills using spaCy and fuzzy matching.

Customizable skill list for different domains.

Scoring System

Hard Score: Measures the percentage of JD skills present in the resume.

Soft Score: Measures semantic similarity using sentence-transformers embeddings.

Final Score: Weighted average of hard and soft scores to generate an overall relevance rating.

Verdict: Categorizes candidates into “High”, “Medium”, or “Low” match.

Web Interface

Easy-to-use form to upload resumes and paste/upload a Job Description.

Displays results in a clear, sortable table showing scores, found skills, and missing skills.

Multiple File Support

Supports .pdf, .docx, .doc, .txt, .png, .jpg, .jpeg.

Technologies Used

Python 3

Flask – Web framework for UI

pdfplumber, docx2txt, Pillow, pytesseract – Text extraction

spaCy, fuzzywuzzy – Skill extraction and matching

sentence-transformers, numpy – Semantic similarity and embeddings

pandas – Tabular results handling

HTML/CSS – Simple UI

How It Works

User uploads one or more resumes and provides a Job Description.

The system extracts text from each resume and the JD.

Skills are parsed from both the JD and resumes.

Hard and soft scores are calculated.

A final weighted score is computed, and a verdict is assigned.

Results are displayed in a table highlighting matched and missing skills.

Use Cases

HR recruiters screening hundreds of resumes quickly.

Hiring managers comparing candidates objectively.

Job platforms providing automated resume scoring.

Candidates self-evaluating their resumes against a job posting.

Next Steps / Future Enhancements

Add a resume ranking dashboard with charts.

Include industry-specific skill sets dynamically.

Integrate resume parsing APIs for more file formats.

Add authentication and database support for saving results.








# Resume Relevance Checker

A Flask web application to evaluate how well candidate resumes match a Job Description (JD). The system extracts skills from both resumes and JD, computes hard and soft scores, and provides an overall relevance verdict.

---

## Features

- Extract text from PDFs, DOCX/DOC, TXT, and images (PNG, JPG, JPEG) using `pdfplumber`, `docx2txt`, and `pytesseract`.
- Detect skills using **spaCy** and fuzzy matching.
- Compute:
  - **Hard Score**: Percentage of JD skills present in the resume.
  - **Soft Score**: Semantic similarity using **sentence-transformers** embeddings.
  - **Final Score**: Weighted combination of hard and soft scores.
- Display results in a sortable table on a web interface.

---

## Folder Structure

resume_relevance/
├── app/
│ ├── init.py
│ ├── extract_text.py
│ ├── skills.py
│ ├── embeddings.py
│ └── scorer.py
├── templates/
│ └── index.html
├── web_app.py
└── requirements.txt




---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/<your-username>/resume-relevance.git
cd resume-relevance



Create a virtual environment:

Windows:

python -m venv venv
.\venv\Scripts\activate



Install dependencies:

pip install -r requirements.txt


Install system dependencies for OCR & PDF processing:

macOS: brew install tesseract poppler

Ubuntu/Debian: sudo apt install -y tesseract-ocr poppler-utils

Windows: Install Tesseract OCR (UB Mannheim recommended) and Poppler; add bin directories to PATH.

Download spaCy English model:

python -m spacy download en_core_web_sm




Step-by-Step Code

app/__init__.py
# app/__init__.py
# This file intentionally left empty to make 'app' a Python package

2. app/extract_text.py
from pathlib import Path
import pdfplumber
import docx2txt
import re
import pytesseract
from PIL import Image
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

URL_REGEX = re.compile(r'https?://\S+|www\.\S+')
REPEAT_LINE_THRESHOLD = 2

# Uncomment if Tesseract OCR is in a custom path
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_pdf_text(path: str) -> str:
    text_chunks = []
    try:
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t:
                    text_chunks.append(t)
    except Exception as e:
        logger.warning(f"pdfplumber error for {path}: {e}. Trying OCR fallback.")
        try:
            from pdf2image import convert_from_path
            pages = convert_from_path(path)
            for img in pages:
                text_chunks.append(pytesseract.image_to_string(img))
        except Exception as e2:
            logger.error(f"OCR fallback failed for {path}: {e2}")
    return "\n".join(text_chunks)

def extract_docx_text(path: str) -> str:
    try:
        return docx2txt.process(path) or ""
    except Exception as e:
        logger.error(f"docx extraction failed for {path}: {e}")
        return ""

def remove_headers_footers(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    counts = {}
    for ln in lines:
        counts[ln] = counts.get(ln, 0) + 1
    filtered = [ln for ln in lines if counts.get(ln, 0) < REPEAT_LINE_THRESHOLD]
    return "\n".join(filtered)

def extract_urls(text: str):
    return URL_REGEX.findall(text)

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r", "\n")
    text = re.sub(r'\u200b', '', text)
    text = re.sub(r'\n{2,}', '\n\n', text)
    return text.strip()

def extract_text(filepath: str) -> dict:
    p = Path(filepath)
    suffix = p.suffix.lower()
    raw = ""
    if suffix == ".pdf":
        raw = extract_pdf_text(filepath)
    elif suffix in (".docx", ".doc"):
        raw = extract_docx_text(filepath)
    elif suffix == ".txt":
        try:
            raw = p.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.error(f"Reading txt failed for {filepath}: {e}")
    elif suffix in (".png", ".jpg", ".jpeg"):
        try:
            img = Image.open(filepath)
            raw = pytesseract.image_to_string(img)
        except Exception as e:
            logger.error(f"OCR on image failed for {filepath}: {e}")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    raw = normalize_text(raw)
    raw = remove_headers_footers(raw)
    urls = extract_urls(raw)
    return {"text": raw, "urls": urls}

3. app/skills.py
import spacy
from spacy.matcher import PhraseMatcher
from fuzzywuzzy import process
import logging

logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")

DEFAULT_SKILLS = [
    "python", "java", "c++", "c#", "sql", "postgresql", "mysql",
    "react", "node.js", "docker", "kubernetes", "aws", "azure",
    "machine learning", "deep learning", "tensorflow", "pytorch",
    "nlp", "computer vision", "git", "html", "css", "javascript",
    "fastapi", "flask"
]

def build_matcher(skills_list=None):
    skills_list = skills_list or DEFAULT_SKILLS
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(s) for s in skills_list]
    matcher.add("SKILL", patterns)
    return matcher, skills_list

matcher, SKILLS = build_matcher()

def extract_skills(text, fuzzy_threshold=90):
    doc = nlp(text or "")
    found = set()
    try:
        matches = matcher(doc)
        for _, start, end in matches:
            found.add(doc[start:end].text.lower())
    except Exception as e:
        logger.error(f"PhraseMatcher error: {e}")

    words = list({tok.text.lower() for tok in doc if tok.is_alpha and len(tok.text) > 2})
    for w in words:
        res = process.extractOne(w, SKILLS)
        if res:
            match, score = res
            if score and score >= fuzzy_threshold:
                found.add(match.lower())
    return sorted(found)



