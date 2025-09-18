# Configuration constants
LANGUAGE_CODE_MAPPING = {
    'en': 'eng_Latn',
    'ko': 'kor_Hang',
    'fr': 'fra_Latn',
    'es': 'spa_Latn',
    'it': 'ita_Latn',
    'pt': 'por_Latn',
    'ru': 'rus_Cyrl',
    'ja': 'jpn_Jpan',
    'hi': 'hin_Deva',
}

MODEL_CONFIGS = {
    'standard': {
        'name': 'facebook/nllb-200-distilled-600M',
        'label': 'Standard (600M) - Balanced speed and accuracy'
    }
}

LOG_FILE = "jana_app.log"
CACHE_DB = "translation_cache.sqlite"