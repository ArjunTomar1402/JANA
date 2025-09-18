import streamlit as st
import pandas as pd
import torch 
from modules.config import MODEL_CONFIGS, LANGUAGE_CODE_MAPPING
from modules.utils import TokenBucket

def render_info_section():
    with st.expander("About JANA", expanded=True):
        st.markdown("""
        <div class="info-box">
            <p>The <strong><a href="https://github.com/ArjunTomar1402/JANA" target="_blank">JANA</a></strong> is a sophisticated translation system designed to handle the complexities of Japanese dialects with cultural and linguistic accuracy.</p>
            <p><span class="feature-text">Phase 1: </span> focuses on multilingual to Standard Japanese translation with integrated morphological analysis using SudachiPy.</p>
            <p><span class="feature-text">Key Features: </span></p>
            <ul>
                <li>Multilingual document translation to Standard Japanese using NLLB-200 (selected model)</li>
                <li>Accurate language detection with fasttext</li>
                <li>Morphological analysis to prepare for dialect conversion (Phase 2)</li>
                <li>Support for PDF, DOCX, and TXT file formats</li>
                <li>Cultural and linguistically-aware processing</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="warning-box">
            <strong>Production tips:</strong> configure the rate limits, caching is enabled (SQLite), use GPU if available for faster translations.
        </div>
        """, unsafe_allow_html=True)

def render_sidebar():
    st.sidebar.header("Configuration")
    st.sidebar.info("Upload files or paste text to translate into Japanese.")

    model_option = st.sidebar.selectbox(
        "Model Size",
        options=list(MODEL_CONFIGS.keys()),
        format_func=lambda x: MODEL_CONFIGS[x]['label'],
        help="Choose between model size (affects speed and quality)"
    )

    custom_model = st.sidebar.text_input(
        "Custom HF model name (optional)",
        value="", 
        help="Provide a HuggingFace model id if you want to try a different translator"
    )

    lang_options = ['AUTO'] + [lang.upper() for lang in LANGUAGE_CODE_MAPPING.keys()]
    manual_lang = st.sidebar.selectbox(
        "Override Language Detection",
        options=lang_options,
        index=0,
        help="Force a specific source language instead of auto-detection"
    )

    st.sidebar.markdown("**Rate limiting (per-session)**")
    rate_capacity = st.sidebar.number_input("Requests per window", min_value=1, max_value=1000, value=30)
    rate_window = st.sidebar.number_input("Window (seconds)", min_value=1, max_value=3600, value=60)
    st.session_state.rate_limiter = TokenBucket(capacity=int(rate_capacity), refill_seconds=int(rate_window))

    use_gpu = False
    try:
        if torch.cuda.is_available():
            use_gpu = st.sidebar.checkbox("Use available GPU for translation", value=True)
        else:
            st.sidebar.checkbox("Use GPU (not available)", value=False, disabled=True)
    except Exception as e:
        st.sidebar.warning(f"torch not usable: {e}")
        use_gpu = False

    st.sidebar.checkbox("Generate Furigana", key="generate_furigana", help="Add furigana readings to Japanese text")
    st.sidebar.checkbox("Debug mode", key="debug_mode", help="Show extra debug information")

    return model_option, custom_model.strip() or None, manual_lang, use_gpu

def render_batch_processor(extract_text_from_file, split_sentences, process_text_batch):
    st.markdown("### Batch File Processing")
    st.markdown('<div class="batch-processor">', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload multiple files for batch processing",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
        key="batch_uploader"
    )

    if uploaded_files and st.button("Process Batch Files", type="secondary"):
        results = []
        for uploaded_file in uploaded_files:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                text = extract_text_from_file(uploaded_file)
                if text:
                    sentences = split_sentences(text)
                    file_results = process_text_batch(sentences, st.session_state.manual_lang, 1)
                    for result in file_results:
                        result["Source File"] = uploaded_file.name
                    results.extend(file_results)

        if results:
            st.success(f"Processed {len(results)} sentences from {len(uploaded_files)} files")
            display_results(results, "GPU" if st.session_state.get('use_gpu', False) else "CPU")

    st.markdown('</div>', unsafe_allow_html=True)

def display_results(results: list, device):
    st.header("Translation Results")
    df = pd.DataFrame(results)
    if "Source File" in df.columns:
        cols = ["Source File"] + [col for col in df.columns if col != "Source File"]
        df = df[cols]

    if "Morphological Analysis" in df.columns:
        df["Morphological Analysis"] = df["Morphological Analysis"].astype(str)

    st.dataframe(df, width="stretch")

    csv = df.to_csv(index=False)
    st.download_button("Download CSV", csv, "jana_phase1_results.csv", "text/csv")

    with st.expander("Technical Details"):
        st.write("**Models & runtime:**")
        st.write(f"- Translator: `{st.session_state.get('translator_name', 'not_loaded')}`")
        st.write(f"- Device: `{device}`")
        st.write(f"- FastText `lid.176.ftz` (local)")
        st.write("- SudachiPy")
        if st.session_state.get('generate_furigana', False):
            st.write("- PyKakasi (for furigana)")
        st.write("**Processing Stats:**")
        st.write(f"- Sentences processed: {len(results)}")
        st.write("**Logs:**")
        st.write(f"- Log file: `jana_app.log` (server-side)")
        if results:
            st.json(results[0])
