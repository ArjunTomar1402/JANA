import streamlit as st
import fasttext
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sudachipy import dictionary
import pykakasi
import torch
from modules.config import MODEL_CONFIGS
from modules.utils import download_fasttext_model  # ✅ now correct

# Global model references
lid_model = None
nllb_model = None
nllb_tokenizer = None
sudachi_tokenizer_obj = None
kakasi_instance = None


@st.cache_resource
def load_models(model_size='standard', device_name: str = "cpu", custom_model_name: str = None):
    """Load all required models"""
    progress_bar = st.progress(0, text="Loading models...")
    models = {}

    # Load FastText
    progress_bar.progress(10, text="Loading language detection model...")
    try:
        model_path = download_fasttext_model()
        if model_path:
            models['lid_model'] = fasttext.load_model(model_path)
        else:
            st.error("Could not load fasttext model")
            models['lid_model'] = None
    except Exception as e:
        st.error(f"Could not load fasttext model: {e}")
        models['lid_model'] = None

    # Load translation model
    progress_bar.progress(40, text="Loading translation model...")
    tried_models = []
    try_names = []
    if custom_model_name:
        try_names.append(custom_model_name)
    try_names.append(MODEL_CONFIGS.get(model_size, MODEL_CONFIGS['standard'])['name'])
    try_names.append(MODEL_CONFIGS['standard']['name'])

    loaded = False
    for nm in try_names:
        if nm in tried_models:
            continue
        tried_models.append(nm)
        try:
            tokenizer = AutoTokenizer.from_pretrained(nm)
            model = AutoModelForSeq2SeqLM.from_pretrained(nm)
            models['nllb_tokenizer'] = tokenizer
            models['nllb_model'] = model.to(device_name)
            models['translator_name'] = nm
            loaded = True
            break
        except Exception as e:
            st.warning(f"Could not load model '{nm}': {e}")
            models['nllb_tokenizer'] = None
            models['nllb_model'] = None
            models['translator_name'] = None

    if not loaded:
        st.error("Failed to load any translation model")

    # Load Sudachi
    progress_bar.progress(70, text="Loading morphological analyzer...")
    try:
        models['sudachi_tokenizer_obj'] = dictionary.Dictionary().create()
    except Exception as e:
        st.error(f"Could not initialize Sudachi: {e}")
        models['sudachi_tokenizer_obj'] = None

    # Load PyKakasi
    progress_bar.progress(90, text="Loading furigana generator...")
    try:
        models['kakasi'] = pykakasi.kakasi()
    except Exception as e:
        st.warning(f"Could not initialize kakasi for furigana: {e}")
        models['kakasi'] = None

    progress_bar.progress(100, text="Models loaded (with warnings if any).")
    return models


def get_models():
    """Return global model references"""
    return lid_model, nllb_model, nllb_tokenizer, sudachi_tokenizer_obj, kakasi_instance


def set_models(lid, nllb, tokenizer, sudachi, kakasi):
    """Set global model references"""
    global lid_model, nllb_model, nllb_tokenizer, sudachi_tokenizer_obj, kakasi_instance
    lid_model = lid
    nllb_model = nllb
    nllb_tokenizer = tokenizer
    sudachi_tokenizer_obj = sudachi
    kakasi_instance = kakasi
