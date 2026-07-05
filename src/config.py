from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

# Posisi root project
# Karena file ini berada di folder src/, maka parent.parent mengarah ke folder utama project
BASE_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# DIRECTORY PATHS
# ============================================================

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

VADER_DIR = BASE_DIR / "vaderSentiment"
CUSTOM_LEXICON_DIR = BASE_DIR / "custom_lexicon"

DOCS_DIR = BASE_DIR / "docs"


# ============================================================
# INPUT FILE
# ============================================================

# Nama file dataset utama
# Letakkan file Excel dataset di folder data/raw/
INPUT_FILE_NAME = "LABEL Data Review JMO.xlsx"
INPUT_FILE_PATH = RAW_DATA_DIR / INPUT_FILE_NAME


# ============================================================
# DATASET COLUMN CONFIGURATION
# ============================================================

# Kolom teks review pada dataset Bapak
TEXT_COLUMN = "comment"

# Kolom label numerik yang akan dibuat otomatis dari P1 dan P3
# Kolom ini tidak harus sudah ada di file Excel awal
LABEL_COLUMN = "label_umux"

# Kolom asli yang ada di dataset Bapak
RAW_UMUX_LABEL_COLUMN = "label UMUX"

# Kolom indikator UMUX-Lite dari dataset
P1_COLUMN = "mudah digunakan (P1)"
P3_COLUMN = "memenuhi kebutuhan (P3)"

# Kolom lain yang ada pada dataset, tidak wajib dipakai untuk model
RATING_COLUMN = "rating"
USER_COLUMN = "user"
DATE_COLUMN = "date"
LIKES_COLUMN = "likes"

# Kolom "label" pada dataset Bapak sebaiknya tidak dipakai
# karena sebelumnya terlihat berisi error #NAME?
UNUSED_LABEL_COLUMN = "label"


# ============================================================
# OUTPUT COLUMNS
# ============================================================

CLEAN_TEXT_COLUMN = "clean_text"

PREDICTED_LABEL_COLUMN = "predicted_umux_label"
PREDICTED_DIMENSION_COLUMN = "predicted_umux_dimension"

VADER_NEG_COLUMN = "vader_neg"
VADER_NEU_COLUMN = "vader_neu"
VADER_POS_COLUMN = "vader_pos"
VADER_COMPOUND_COLUMN = "vader_compound"

UMUX_SCORE_COLUMN = "umux_lite_score_1_7"
SENTIMENT_CATEGORY_COLUMN = "sentiment_category"


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_FILE_NAME = "umux_lite_labeler.pkl"
MODEL_FILE_PATH = MODEL_DIR / MODEL_FILE_NAME


# ============================================================
# OUTPUT FILES
# ============================================================

OUTPUT_EXCEL_FILE_NAME = "hasil_umux_vader.xlsx"
OUTPUT_CSV_FILE_NAME = "hasil_umux_vader.csv"
SUMMARY_CSV_FILE_NAME = "summary_umux_vader.csv"

OUTPUT_EXCEL_PATH = OUTPUT_DIR / OUTPUT_EXCEL_FILE_NAME
OUTPUT_CSV_PATH = OUTPUT_DIR / OUTPUT_CSV_FILE_NAME
SUMMARY_CSV_PATH = OUTPUT_DIR / SUMMARY_CSV_FILE_NAME


# ============================================================
# CUSTOM LEXICON CONFIGURATION
# ============================================================

CUSTOM_LEXICON_FILE_NAME = "indonesian_vader_lexicon.tsv"
CUSTOM_LEXICON_PATH = CUSTOM_LEXICON_DIR / CUSTOM_LEXICON_FILE_NAME


# ============================================================
# UMUX-LITE LABEL CONFIGURATION
# ============================================================

# Mapping label numerik:
# 0 = Tidak relevan dengan UMUX-Lite
# 1 = Usefulness / memenuhi kebutuhan
# 2 = Ease of Use / mudah digunakan
# 3 = Usefulness + Ease of Use

UMUX_LABEL_MAP = {
    0: "Tidak relevan dengan UMUX-Lite",
    1: "Usefulness",
    2: "Ease of Use",
    3: "Usefulness + Ease of Use",
}


# Mapping dari kolom P1 dan P3:
# P1 = mudah digunakan
# P3 = memenuhi kebutuhan
#
# P1=0, P3=0 -> Label 0
# P1=0, P3=1 -> Label 1
# P1=1, P3=0 -> Label 2
# P1=1, P3=1 -> Label 3

VALID_UMUX_LABELS = [0, 1, 2, 3]


# ============================================================
# SENTIMENT CONFIGURATION
# ============================================================

# Threshold standar VADER:
# compound >= 0.05  = positive
# compound <= -0.05 = negative
# selain itu        = neutral

POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05


# ============================================================
# RANDOM STATE
# ============================================================

RANDOM_STATE = 42


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

TEST_SIZE = 0.2

TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.9


# ============================================================
# HELPER FUNCTION
# ============================================================

def create_required_directories():
    """
    Membuat folder penting jika belum tersedia.
    Fungsi ini akan dipanggil pada file training dan pipeline utama.
    """
    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODEL_DIR,
        OUTPUT_DIR,
        CUSTOM_LEXICON_DIR,
        DOCS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)