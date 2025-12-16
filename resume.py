"""
AI-Powered Resume Screening & Job Match System

Backend ML/NLP pipeline (TF-IDF and optional SBERT)
Streamlit frontend for uploading resumes, pasting job descriptions, ranking results

Usage:
1. pip install -r requirements.txt
2. streamlit run resume_matcher.py

Requirements (suggested): 
    streamlit scikit-learn pandas numpy fitz (pymupdf) spacy wordcloud (optional) python-multipart
Optional for better embeddings:
    sentence-transformers
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import re
from typing import List, Tuple

# --------------------------- Text processing libraries ---------------------------
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except Exception:
    spacy = None
    nlp = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional: sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except Exception:
    SBERT_AVAILABLE = False


# --------------------------- Helper functions ---------------------------

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF bytes using PyMuPDF if available, otherwise fallback to naive decoding."""
    if fitz is None:
        try:
            return file_bytes.decode('utf-8', errors='ignore')
        except Exception:
            return ""

    text = []
    try:
        with fitz.open(stream=file_bytes, filetype='pdf') as doc:
            for page in doc:
                txt = page.get_text()
                if txt:
                    text.append(txt)
    except Exception:
        try:
            return file_bytes.decode('utf-8', errors='ignore')
        except Exception:
            return ""
    return "\n".join(text)


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from .docx using python-docx if available; otherwise return empty."""
    try:
        import docx
    except Exception:
        return ""
    try:
        bio = io.BytesIO(file_bytes)
        doc = docx.Document(bio)
        paras = [p.text for p in doc.paragraphs]
        return "\n".join(paras)
    except Exception:
        return ""


def extract_text_from_file(uploaded_file) -> str:
    """Given a Streamlit uploaded file object, return extracted text."""
    raw = uploaded_file.read()
    fname = uploaded_file.name.lower()
    if fname.endswith('.pdf'):
        return extract_text_from_pdf(raw)
    if fname.endswith('.docx') or fname.endswith('.doc'):
        return extract_text_from_docx(raw)
    try:
        return raw.decode('utf-8', errors='ignore')
    except Exception:
        return ''


def clean_text(text: str) -> str:
    """Lowercase, remove emails, urls, punctuation; lemmatize if spaCy available."""
    if not isinstance(text, str):
        return ""
    text = text.replace('\r', ' ').replace('\n', ' ')
    text = re.sub(r"\S+@\S+", ' ', text)  # remove emails
    text = re.sub(r'http\S+|www.\S+', ' ', text)  # remove urls
    text = re.sub(r"[^A-Za-z+ ]", ' ', text)  # keep A-Z, +, space
    text = re.sub(r"\s+", ' ', text).strip()
    text = text.lower()

    if nlp:
        doc = nlp(text)
        lemmas = [token.lemma_ for token in doc if not token.is_stop]
        return ' '.join(lemmas)
    else:
        stopwords = set([
            "the","and","or","in","of","to","a","an","for","with",
            "on","by","at","from","as","is","are","be"
        ])
        tokens = [t for t in text.split() if t not in stopwords]
        return ' '.join(tokens)


# --------------------------- Matching backend ---------------------------

def build_tfidf_embeddings(corpus: List[str], max_features: int = 5000) -> Tuple[TfidfVectorizer, np.ndarray]:
    """Fit TF-IDF on corpus and return vectorizer and embeddings matrix."""
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X = vectorizer.fit_transform(corpus)
    return vectorizer, X


def compute_cosine_sim_matrix(X_embeds, query_vec) -> np.ndarray:
    """Compute cosine similarity between each row in X_embeds and query_vec."""
    sims = cosine_similarity(X_embeds, query_vec)
    return sims.ravel()


class Matcher:
    def __init__(self, method: str = 'tfidf', tfidf_max_features: int = 5000,
                 sbert_model_name: str = 'all-MiniLM-L6-v2'):
        self.method = method
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.sbert = None
        self.sbert_model_name = sbert_model_name
        self.sbert_embeddings = None
        self.tfidf_max_features = tfidf_max_features

        if method == 'sbert' and SBERT_AVAILABLE:
            self.sbert = SentenceTransformer(self.sbert_model_name)

    def fit(self, docs: List[str]):
        docs = [d if isinstance(d, str) else '' for d in docs]
        if self.method == 'tfidf':
            self.tfidf_vectorizer, self.tfidf_matrix = build_tfidf_embeddings(
                docs, max_features=self.tfidf_max_features
            )
        elif self.method == 'sbert' and self.sbert is not None:
            self.sbert_embeddings = self.sbert.encode(docs, convert_to_numpy=True)
        else:
            raise ValueError('Unsupported method or missing SBERT library')

    def query(self, query_text: str) -> np.ndarray:
        query_text = query_text if isinstance(query_text, str) else ''
        if self.method == 'tfidf':
            qv = self.tfidf_vectorizer.transform([query_text])
            sims = compute_cosine_sim_matrix(self.tfidf_matrix, qv)
            return sims
        elif self.method == 'sbert' and self.sbert is not None:
            qv = self.sbert.encode([query_text], convert_to_numpy=True)
            sims = cosine_similarity(self.sbert_embeddings, qv).ravel()
            return sims
        else:
            raise ValueError('Unsupported method or missing SBERT library')


# --------------------------- Utilities for output ---------------------------

def percent(x: float) -> str:
    return f"{round(float(x)*100, 1)}%"


def extract_top_matched_keywords(tfidf_vectorizer: TfidfVectorizer, resume_text: str,
                                 jd_text: str, top_n: int = 8) -> List[str]:
    """Return top_n keywords from resume that appear in JD."""
    try:
        jd_tokens = set(jd_text.split())
        vec = tfidf_vectorizer.transform([resume_text])
        feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
        scores = vec.toarray().ravel()
        if scores.sum() == 0:
            return []
        top_idx = scores.argsort()[::-1]
        keywords = [feature_names[i] for i in top_idx if feature_names[i] in jd_tokens]
        if len(keywords) < top_n:
            keywords += [feature_names[i] for i in top_idx if feature_names[i] not in keywords]
        return keywords[:top_n]
    except Exception:
        return []


# --------------------------- Streamlit App ---------------------------

st.set_page_config(page_title="Resume Matcher", layout='wide')
st.title("AI-Powered Resume Screening & Job Match System")
st.write("Upload resumes (PDF/DOCX/TXT) and paste the job description — the app ranks resumes by match percentage.")

with st.sidebar:
    st.header("Settings")
    method = st.selectbox(
        "Embedding method",
        options=['tfidf'] + (['sbert'] if SBERT_AVAILABLE else []),
        index=0
    )
    max_features = st.slider("TF-IDF max features", 500, 20000, 5000, 500)
    show_keywords = st.checkbox("Show matched keywords for each resume", value=True)

    file_examples = st.expander("Example job descriptions / tips")
    with file_examples:
        st.markdown(
            "- Use clear JD with skills listed like: Python, SQL, Machine Learning, TensorFlow\n"
            "- Upload multiple resumes at once for ranking"
        )

col1, col2 = st.columns([1, 2])
with col1:
    uploaded_files = st.file_uploader("Upload resumes (pdf/docx/txt)", accept_multiple_files=True)
    st.write(f"{len(uploaded_files) if uploaded_files else 0} files uploaded")

with col2:
    jd_text = st.text_area("Paste job description here", height=200)
    sample_jd = "Data Scientist with experience in Python, Machine Learning, SQL, Deep Learning, and model deployment."
    if st.button("Fill sample JD"):
        jd_text = sample_jd
        st.experimental_rerun()

if not uploaded_files:
    st.info("Upload at least one resume to enable matching.")

if uploaded_files and jd_text.strip() == "":
    st.warning("Please paste the job description text before running matching.")


# --------------------------- Main action ---------------------------

if st.button("Run Matching"):
    if not uploaded_files:
        st.error("Please upload resumes first.")
    elif jd_text.strip() == "":
        st.error("Please paste a job description first.")
    else:
        with st.spinner("Processing files and running the matching engine..."):
            resumes = []
            filenames = []
            for f in uploaded_files:
                txt = extract_text_from_file(f)
                clean = clean_text(txt)
                resumes.append(clean)
                filenames.append(f.name)

            clean_jd = clean_text(jd_text)

            matcher = Matcher(method=method, tfidf_max_features=max_features)
            corpus = resumes + [clean_jd]
            matcher.fit(corpus)

            sims = matcher.query(clean_jd)
            sims = sims[:len(resumes)]

            results = pd.DataFrame({
                'filename': filenames,
                'score': sims
            })
            results['match_percent'] = results['score'].apply(lambda x: round(float(x)*100, 1))
            results = results.sort_values('score', ascending=False).reset_index(drop=True)

            if method == 'tfidf' and show_keywords:
                keywords_list = []
                for rtext in resumes:
                    kws = extract_top_matched_keywords(matcher.tfidf_vectorizer, rtext, clean_jd, top_n=6)
                    keywords_list.append(', '.join(kws))
                results['matched_keywords'] = keywords_list

            st.success("Matching complete!")
            st.subheader("Ranked Resumes")

            display_df = results.copy()
            display_df['match_percent'] = display_df['match_percent'].astype(str) + '%'
            st.dataframe(display_df[['filename','match_percent'] + (
                ['matched_keywords'] if 'matched_keywords' in display_df.columns else []
            )])

            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button("Download results CSV", data=csv,
                               file_name='resume_match_results.csv', mime='text/csv')

            for idx, row in results.iterrows():
                with st.expander(f"{idx+1}. {row['filename']} — {row['match_percent']:.1f}%"):
                    st.write("*Match Score:*", f"{row['match_percent']:.1f}%")
                    if 'matched_keywords' in row:
                        st.write("*Matched keywords:*", row['matched_keywords'])
                    st.write("*Preview of extracted text (first 800 chars):*")
                    preview = resumes[idx][:800] if resumes[idx] else '(No extractable text)'
                    st.text(preview)


# --------------------------- Footer ---------------------------

st.markdown("---")
st.markdown(
    "Tips before demo:\n"
    "- Use clean PDF resumes (text-based PDFs) for best extraction results.\n"
    "- Scanned images need OCR (e.g., Tesseract).\n"
    "- For better semantic matching, install sentence-transformers and select SBERT in the sidebar.\n"
    "- For production-grade parsing, extend with parsers to extract: Education, Experience, Skills, Projects."
)
st.caption("Built with ❤ for final-year project — modify & extend as needed.")
