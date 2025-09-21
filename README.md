Resume Relevance Checker

A Flask web application to evaluate how well candidate resumes match a Job Description (JD). It extracts skills from both resumes and JD, calculates hard and soft scores, and provides an overall relevance verdict.

Table of Contents

Project Description

Features

Folder Structure

Installation

Step-by-Step Code Explanation

Project Description

Recruiters and hiring managers often deal with hundreds of resumes per role. This project automates the evaluation process by:

Extracting text from resumes and Job Descriptions.

Detecting skills using NLP and fuzzy matching.

Calculating hard, soft, and final scores.

Displaying results in a clear, sortable web interface.

Supports multiple file formats: .pdf, .docx, .doc, .txt, .png, .jpg, .jpeg.

Features

Text Extraction: Extracts content from PDF, DOCX, TXT, and images (OCR fallback).

Skill Detection: Uses spaCy and fuzzy matching to detect skills.

Scoring: Computes:

Hard Score: Exact match of JD skills in resume.

Soft Score: Semantic similarity using sentence-transformers embeddings.

Final Score: Weighted average of hard and soft scores.

Verdict: Categorizes resumes into "High", "Medium", or "Low" match.

Web Interface: Upload resumes, paste/upload JD, and view score




Folder Structure

resume_relevance/
├── app/
│   ├── __init__.py
│   ├── extract_text.py
│   ├── skills.py
│   ├── embeddings.py
│   └── scorer.py
├── templates/
│   └── index.html
├── web_app.py
└── requirements.txt


Installation

Clone the repository

git clone https://github.com/<your-username>/resume-relevance.git
cd resume-relevance



Create a virtual environment

Windows:
python -m venv venv
.\venv\Scripts\activate




Install dependencies

pip install -r requirements.txt


Install system dependencies for OCR & PDF

macOS: brew install tesseract poppler

Ubuntu/Debian: sudo apt install -y tesseract-ocr poppler-utils

Windows: Install Tesseract OCR (UB Mannheim recommended) and Poppler; add their bin directories to PATH.



Download spaCy English model

python -m spacy download en_core_web_sm



Run the web app

# This file is intentionally left empty
# It tells Python that 'app' is a package

Purpose:
Allows other Python files to import modules from the app folder.




2. app/extract_text.py

Handles text extraction from resumes and JD files.


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

# Uncomment if Tesseract OCR is installed in a custom path
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



Explanation:

Extracts text from PDF, DOCX, TXT, and images.

Handles OCR for scanned documents.

Cleans and normalizes text.

Removes repeated headers/footers.

Extracts URLs.





3. app/skills.py

Detects skills from text using NLP and fuzzy matching.



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
Explanation:

Uses spaCy PhraseMatcher for exact skill detection.

Uses fuzzywuzzy for approximate matching.

Returns a sorted list of found skills.



4. app/embeddings.py

Handles semantic similarity between resume and JD using embeddings.

from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
from pathlib import Path

MODEL_NAME = "all-MiniLM-L6-v2"
_cache_dir = Path(".cache_embeddings")
_cache_dir.mkdir(exist_ok=True)

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def _normalize(vec: np.ndarray):
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

def embed_text(text: str):
    """
    Embed text (truncated to 1500 chars to limit size). Returns normalized vector.
    """
    model = get_model()
    doc = text if len(text) < 1500 else text[:1500]
    vec = model.encode(doc, convert_to_numpy=True)
    return _normalize(vec)

def embed_batch(list_of_texts, cache_name=None):
    """
    Embed multiple texts at once. Optionally caches results for reuse.
    """
    if cache_name:
        cache_file = _cache_dir / f"{cache_name}.pkl"
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                return pickle.load(f)
    model = get_model()
    vecs = model.encode(list_of_texts, convert_to_numpy=True, show_progress_bar=True)
    vecs = [_normalize(v) for v in vecs]
    if cache_name:
        with open(_cache_dir / f"{cache_name}.pkl", "wb") as f:
            pickle.dump(vecs, f)
    return vecs

def cosine_sim(a, b):
    """Compute cosine similarity between two normalized vectors."""
    return float(np.dot(a, b))


Explanation:

Uses sentence-transformers to embed text into vectors.

_normalize ensures vectors have unit length.

embed_text is used for single text (JD or resume).

embed_batch can embed multiple texts efficiently and cache them.

cosine_sim calculates similarity between vectors (soft score).



5. app/scorer.py

Computes hard score, soft score, final score, and verdict.

from typing import List
from app.skills import extract_skills
from app.embeddings import embed_text, cosine_sim

def parse_jd_skills(jd_text: str) -> List[str]:
    return extract_skills(jd_text)

def hard_score_resume(jd_skills: List[str], resume_skills: List[str]) -> float:
    """Percentage of JD skills present in resume."""
    if not jd_skills:
        return 50.0
    jd_set = set(s.lower() for s in jd_skills)
    res_set = set(s.lower() for s in resume_skills)
    matched = jd_set.intersection(res_set)
    score = (len(matched) / len(jd_set)) * 100
    return round(score, 2)

def soft_score_resume(jd_text: str, resume_text: str) -> float:
    """Semantic similarity score of JD and resume."""
    if not jd_text or not resume_text:
        return 50.0
    v1 = embed_text(jd_text)
    v2 = embed_text(resume_text)
    sim = cosine_sim(v1, v2)  # in [-1,1]
    soft = max(min((sim + 1) / 2 * 100, 100), 0)
    return round(soft, 2)

def final_score(hard: float, soft: float, hard_w=0.65, soft_w=0.35) -> int:
    """Weighted combination of hard and soft scores."""
    final = hard_w * hard + soft_w * soft
    return int(round(final))

def verdict_from_score(score: int) -> str:
    """Assigns verdict based on final score."""
    if score >= 75:
        return "High"
    if score >= 50:
        return "Medium"
    return "Low"

Explanation:

Hard Score: Exact match of JD skills in resume.

Soft Score: Semantic similarity of full text (embedding-based).

Final Score: Weighted average (hard=65%, soft=35%).

Verdict: Categorizes resumes as High, Medium, Low.


6. web_app.py

The main Flask application serving the web interface.



from flask import Flask, render_template, request, flash
from pathlib import Path
import tempfile, os
import pandas as pd
import logging

from app.extract_text import extract_text
from app.skills import extract_skills
from app.scorer import parse_jd_skills, hard_score_resume, soft_score_resume, final_score, verdict_from_score

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "dev_secret_key")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    jd_text = ""

    if request.method == "POST":
        jd_text = (request.form.get("jd_text") or "").strip()
        jd_file = request.files.get("jd_file")

        # Extract JD from file if text not provided
        if jd_file and not jd_text and jd_file.filename:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(jd_file.filename).suffix)
            try:
                jd_file.save(tmp.name)
                data = extract_text(tmp.name)
                jd_text = data.get("text", "")
            finally:
                os.unlink(tmp.name)

        resumes = request.files.getlist("resumes")
        if not jd_text:
            flash("Please paste a Job Description or upload a JD file.", "warning")
        elif not resumes or all((not f or not f.filename) for f in resumes):
            flash("Please upload one or more resume files.", "warning")
        else:
            rows = []
            jd_skills = parse_jd_skills(jd_text)
            for rf in resumes:
                if not rf or not rf.filename:
                    continue
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(rf.filename).suffix)
                try:
                    rf.save(tmp.name)
                    data = extract_text(tmp.name)
                    rtext = data.get("text", "")
                finally:
                    os.unlink(tmp.name)

                res_skills = extract_skills(rtext)
                hard = hard_score_resume(jd_skills, res_skills)
                soft = soft_score_resume(jd_text, rtext)
                final = final_score(hard, soft)
                verdict = verdict_from_score(final)
                missing = sorted(set(s.lower() for s in jd_skills) - set(s.lower() for s in res_skills))
                rows.append({
                    "file": rf.filename,
                    "hard_score": hard,
                    "soft_score": soft,
                    "final_score": final,
                    "verdict": verdict,
                    "found_skills": ", ".join(res_skills),
                    "missing_skills": ", ".join(missing)
                })

            results = pd.DataFrame(rows).sort_values("final_score", ascending=False).to_dict(orient="records")

    return render_template("index.html", results=results, jd_text=jd_text)

if __name__ == "__main__":
    app.run(debug=True, port=5000)






Explanation:

Handles GET and POST requests.

Accepts JD text or JD file, and multiple resume files.

Extracts text from JD and resumes.

Calculates hard, soft, and final scores.

Shows results in a sortable HTML table.




7. templates/index.html

HTML template for the web interface.



<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>Resume Relevance Checker</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        table { border-collapse: collapse; width: 100%; margin-top: 20px; }
        th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
        th { background: #f4f4f4; }
        .flash { padding: 10px; background: #ffefc2; margin-bottom: 12px; }
    </style>
</head>
<body>
    <h1>Resume Relevance Checker</h1>

    {% with messages = get_flashed_messages(with_categories=true) %}
      {% if messages %}
        {% for category, msg in messages %}
          <div class="flash {{ category }}">{{ msg }}</div>
        {% endfor %}
      {% endif %}
    {% endwith %}

    <form method="POST" enctype="multipart/form-data">
        <label>Paste Job Description:</label><br>
        <textarea name="jd_text" rows="6" cols="80">{{ jd_text }}</textarea><br><br>

        <label>Or Upload JD File (.txt/.pdf/.docx):</label>
        <input type="file" name="jd_file" accept=".txt,.pdf,.docx"><br><br>

        <label>Upload Resumes (.pdf/.docx) — Multiple:</label>
        <input type="file" name="resumes" multiple accept=".pdf,.docx"><br><br>

        <button type="submit">Evaluate</button>
    </form>

    {% if results %}
    <h2>Results</h2>
    <table>
        <tr>
            <th>File</th>
            <th>Hard Score</th>
            <th>Soft Score</th>
            <th>Final Score</th>
            <th>Verdict</th>
            <th>Found Skills</th>
            <th>Missing Skills</th>
        </tr>
        {% for row in results %}
        <tr>
            <td>{{ row.file }}</td>
            <td>{{ row.hard_score }}</td>
            <td>{{ row.soft_score }}</td>
            <td>{{ row.final_score }}</td>
            <td>{{ row.verdict }}</td>
            <td>{{ row.found_skills }}</td>
            <td>{{ row.missing_skills }}</td>
        </tr>
        {% endfor %}
    </table>
    {% endif %}
</body>
</html>






