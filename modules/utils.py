import streamlit as st
import sqlite3
import time
from collections import deque
import re
from io import StringIO
import pypdf
import docx2txt
import urllib.request
import os
import logging
from typing import List
from modules.config import CACHE_DB

# Initialize logging
logger = logging.getLogger("jana")

# Cache DB functions
def init_cache_db():
    conn = sqlite3.connect(CACHE_DB, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS translations (
            id INTEGER PRIMARY KEY,
            src_text TEXT,
            src_lang TEXT,
            model TEXT,
            translation TEXT,
            created_ts INTEGER
        )
    """)
    conn.commit()
    return conn

cache_conn = init_cache_db()

def cache_lookup(src_text: str, src_lang: str, model_name: str):
    cur = cache_conn.cursor()
    cur.execute(
        "SELECT translation FROM translations WHERE src_text = ? AND src_lang = ? AND model = ? LIMIT 1",
        (src_text, src_lang, model_name)
    )
    row = cur.fetchone()
    return row[0] if row else None

def cache_store(src_text: str, src_lang: str, model_name: str, translation: str):
    cur = cache_conn.cursor()
    cur.execute(
        "INSERT INTO translations (src_text, src_lang, model, translation, created_ts) VALUES (?, ?, ?, ?, ?)",
        (src_text, src_lang, model_name, translation, int(time.time()))
    )
    cache_conn.commit()

# Rate limiting
class TokenBucket:
    def __init__(self, capacity: int, refill_seconds: int):
        self.capacity = capacity
        self.refill_seconds = refill_seconds
        self.timestamps = deque()

    def allow(self) -> bool:
        now = time.time()
        while self.timestamps and now - self.timestamps[0] > self.refill_seconds:
            self.timestamps.popleft()
        if len(self.timestamps) < self.capacity:
            self.timestamps.append(now)
            return True
        return False

def get_rate_limiter():
    if "rate_limiter" not in st.session_state:
        st.session_state.rate_limiter = TokenBucket(capacity=30, refill_seconds=60)
    return st.session_state.rate_limiter

# Text processing utilities
def download_fasttext_model():
    model_url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
    model_path = "lid.176.ftz"
    if not os.path.exists(model_path):
        with st.spinner("Downloading language detection model..."):
            try:
                urllib.request.urlretrieve(model_url, model_path)
                st.success("Downloaded language detection model successfully!")
            except Exception as e:
                st.error(f"Failed to download model: {e}")
                logger.exception("Failed to download fasttext model")
                return None
    return model_path

def split_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?。！？])\s+', text)
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 1]

def extract_text_from_file(uploaded_file):
    try:
        if uploaded_file.type == "text/plain":
            return StringIO(uploaded_file.getvalue().decode("utf-8")).read()
        elif uploaded_file.type == "application/pdf":
            reader = pypdf.PdfReader(uploaded_file)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
            text = re.sub(r'\s+', ' ', text)
            return text
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return docx2txt.process(uploaded_file)
        else:
            st.error(f"Unsupported file type: {uploaded_file.type}")
            return None
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        logger.exception("extract_text_from_file failed")
        return None

def generate_furigana(text: str) -> str:
    # ✅ Lazy import to prevent circular import
    from modules.models import get_models

    _, _, _, _, kakasi_instance = get_models()
    try:
        if not kakasi_instance:
            return text
        result = kakasi_instance.convert(text)
        furigana_text = ""
        for item in result:
            if item.get('orig') == item.get('hira'):
                furigana_text += item['orig']
            else:
                furigana_text += f"{item['orig']}[{item['hira']}]"
        return furigana_text
    except Exception as e:
        st.warning(f"Furigana generation failed: {e}")
        logger.exception("generate_furigana failed")
        return text

def post_process_japanese(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = re.sub(r'[\x00-\x1f\x7f]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace(" ?", "？").replace(" !", "！").replace(" :", "：")
    text = re.sub(r'^\[NOT JAPANESE OUTPUT\]\s*', '', text)
    return text

def is_japanese(text: str) -> bool:
    if not text:
        return False
    return bool(re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', text))

def _sudachi_to_string(morphemes) -> str:
    try:
        return " | ".join([f"{m.surface()}({m.part_of_speech()[0]})" for m in morphemes])
    except Exception:
        return str(morphemes)
