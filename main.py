import streamlit as st
import torch
import logging
from modules.ui import render_info_section, render_sidebar, render_batch_processor, display_results
from modules.models import load_models, set_models, get_models
from modules.utils import extract_text_from_file, split_sentences
from modules.processing import process_text_batch
from modules.config import LOG_FILE

# Initialize logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("jana")
logger.info("Starting JANA app")

# Page config
st.set_page_config(
    page_title="JANA Platform - Phase 1",
    page_icon="🗾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load external CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("assets/styles.css")  # Invoke CSS from separate file

# App title
st.markdown('<h1 class="main-header"> J A N A </h1>', unsafe_allow_html=True)
st.markdown('<h2 class="sub-header">Japanese Auto-dialect Normalization & Adaptation - Phase 1</h2>', unsafe_allow_html=True)

def main():
    # Initialize session state defaults
    if 'model_option' not in st.session_state:
        st.session_state.model_option = 'standard'
    if 'manual_lang' not in st.session_state:
        st.session_state.manual_lang = 'auto'
    if 'generate_furigana' not in st.session_state:
        st.session_state.generate_furigana = False
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False
    if 'translator_name' not in st.session_state:
        st.session_state.translator_name = 'facebook/nllb-200-distilled-600M'

    # Render UI
    render_info_section()
    model_option, custom_model_name, manual_lang, use_gpu = render_sidebar()

    # Update session state
    st.session_state.model_option = model_option
    st.session_state.manual_lang = manual_lang
    
    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    logger.info(f"Using device: {device}")

    # Load models
    models = load_models(model_option, device_name=str(device), custom_model_name=custom_model_name)
    set_models(
        models.get('lid_model'),
        models.get('nllb_model'),
        models.get('nllb_tokenizer'),
        models.get('sudachi_tokenizer_obj'),
        models.get('kakasi')
    )
    
    translator_name = models.get('translator_name') or 'facebook/nllb-200-distilled-600M'
    st.session_state.translator_name = translator_name

    lid_model, nllb_model, _, sudachi_tokenizer_obj, _ = get_models()
    if lid_model is None or nllb_model is None or sudachi_tokenizer_obj is None:
        st.error("Failed to load required models. Please check the error messages above and logs.")
        st.stop()

    # Single file/text processing
    st.header("Single Document Input")
    uploaded_file = st.file_uploader("Upload document", type=["txt", "pdf", "docx"], key="single_uploader")
    manual_text = st.text_area("Or enter text manually:", height=150, key="manual_text")

    input_text = None
    if uploaded_file:
        input_text = extract_text_from_file(uploaded_file)
        if input_text:
            st.success("File uploaded successfully!")
            with st.expander("View extracted text"):
                st.text(input_text[:1000] + ("..." if len(input_text) > 1000 else ""))
    elif manual_text:
        input_text = manual_text

    # Process single input
    if input_text and st.button("Translate to Japanese", type="primary"):
        sentences = split_sentences(input_text)
        results = process_text_batch(sentences, manual_lang, 1)
        if results:
            display_results(results, device)

    # Batch processing section
    render_batch_processor(extract_text_from_file, split_sentences, process_text_batch)

    # Footer with GitHub hyperlink
    st.markdown(
        '<div class="footer"><a href="https://github.com/your-github-repo" target="_blank" style="color:#7f7f7f; text-decoration:none;">JANA</a> - Phase 1 Implementation<br>Developed with cultural and linguistic accuracy</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
