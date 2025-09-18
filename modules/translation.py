from transformers import pipeline
import torch
from modules.config import LANGUAGE_CODE_MAPPING
from modules.utils import cache_lookup, cache_store, get_rate_limiter, is_japanese, post_process_japanese
from modules.models import get_models

def translate_text(text: str, src_lang_code: str = "auto", model_name_for_cache: str = "nllb-200") -> str:
    """Translate text to Japanese"""
    lid_model, _, _, _, _ = get_models()
    
    limiter = get_rate_limiter()
    if not limiter.allow():
        return "[Rate limit exceeded: slow down]"

    cached = cache_lookup(text, src_lang_code, model_name_for_cache)
    if cached:
        return cached

    if src_lang_code == "auto":
        predictions = lid_model.predict(text, k=1)
        src_lang_code = predictions[0][0].replace("__label__", "")
        conf = float(predictions[1][0])
        if conf < 0.7:
            src_lang_code = "en"  # fallback to English if uncertain
    else:
        conf = 1.0

    src_lang_hf = LANGUAGE_CODE_MAPPING.get(src_lang_code, f"{src_lang_code}_Latn")

    try:
        translator = pipeline(
            "translation",
            model="facebook/nllb-200-distilled-600M",
            src_lang=src_lang_hf,
            tgt_lang="jpn_Jpan",
            device=0 if torch.cuda.is_available() else -1,
            max_length=2048,
            num_beams=5,
            no_repeat_ngram_size=3
        )

        result = translator(text)[0]['translation_text']
        result = post_process_japanese(result)

        if not is_japanese(result):
            result = "[NOT JAPANESE OUTPUT] " + result

        cache_store(text, src_lang_code, model_name_for_cache, result)
        return result

    except Exception as e:
        return f"[Translation error: {str(e)}]"