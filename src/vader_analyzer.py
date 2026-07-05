import pandas as pd

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.config import (
    CLEAN_TEXT_COLUMN,
    CUSTOM_LEXICON_PATH,
    VADER_NEG_COLUMN,
    VADER_NEU_COLUMN,
    VADER_POS_COLUMN,
    VADER_COMPOUND_COLUMN,
    SENTIMENT_CATEGORY_COLUMN,
    POSITIVE_THRESHOLD,
    NEGATIVE_THRESHOLD,
)


# ============================================================
# CUSTOM LEXICON LOADER
# ============================================================

def load_custom_lexicon(custom_lexicon_path=CUSTOM_LEXICON_PATH):
    """
    Membaca custom lexicon tambahan untuk VADER.

    Format file:
    kata<TAB>score

    Contoh isi file custom_lexicon/indonesian_vader_lexicon.tsv:

    mudah       2.5
    bantu       2.4
    lancar      2.2
    ribet      -2.5
    gagal      -2.8
    error      -2.6

    Returns
    -------
    dict
        Dictionary berisi pasangan kata dan skor sentimen.
    """
    custom_lexicon = {}

    if not custom_lexicon_path.exists():
        return custom_lexicon

    with open(custom_lexicon_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            parts = line.split("\t")

            if len(parts) < 2:
                continue

            word = parts[0].strip()
            score = parts[1].strip()

            try:
                custom_lexicon[word] = float(score)
            except ValueError:
                continue

    return custom_lexicon


# ============================================================
# VADER ANALYZER INITIALIZATION
# ============================================================

def create_vader_analyzer(use_custom_lexicon=True):
    """
    Membuat objek VADER SentimentIntensityAnalyzer.

    Jika use_custom_lexicon=True, maka custom lexicon Bahasa Indonesia
    akan ditambahkan ke dalam lexicon VADER.
    """
    analyzer = SentimentIntensityAnalyzer()

    if use_custom_lexicon:
        custom_lexicon = load_custom_lexicon()

        if custom_lexicon:
            analyzer.lexicon.update(custom_lexicon)

    return analyzer


# ============================================================
# SENTIMENT CATEGORY
# ============================================================

def get_sentiment_category(
    compound_score,
    positive_threshold=POSITIVE_THRESHOLD,
    negative_threshold=NEGATIVE_THRESHOLD,
):
    """
    Mengubah VADER compound score menjadi kategori sentimen.

    Standar VADER:
    compound >= 0.05  = positive
    compound <= -0.05 = negative
    selain itu        = neutral
    """
    if compound_score >= positive_threshold:
        return "positive"

    if compound_score <= negative_threshold:
        return "negative"

    return "neutral"


# ============================================================
# SINGLE TEXT ANALYSIS
# ============================================================

def analyze_sentiment_text(text, analyzer=None):
    """
    Menganalisis sentimen pada satu teks review.

    Output:
    - neg
    - neu
    - pos
    - compound
    - sentiment_category
    """
    if analyzer is None:
        analyzer = create_vader_analyzer()

    if pd.isna(text):
        text = ""

    text = str(text)

    scores = analyzer.polarity_scores(text)

    result = {
        VADER_NEG_COLUMN: scores["neg"],
        VADER_NEU_COLUMN: scores["neu"],
        VADER_POS_COLUMN: scores["pos"],
        VADER_COMPOUND_COLUMN: scores["compound"],
        SENTIMENT_CATEGORY_COLUMN: get_sentiment_category(scores["compound"]),
    }

    return result


# ============================================================
# DATAFRAME ANALYSIS
# ============================================================

def analyze_sentiment_dataframe(
    df,
    text_column=CLEAN_TEXT_COLUMN,
    use_custom_lexicon=True,
):
    """
    Melakukan analisis sentimen VADER pada DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset review.

    text_column : str
        Nama kolom teks yang akan dianalisis.
        Pada project ini default-nya menggunakan kolom clean_text.

    use_custom_lexicon : bool
        Jika True, custom lexicon Bahasa Indonesia akan digunakan.

    Returns
    -------
    pandas.DataFrame
        DataFrame dengan tambahan kolom:
        - vader_neg
        - vader_neu
        - vader_pos
        - vader_compound
        - sentiment_category
    """
    if text_column not in df.columns:
        raise ValueError(
            f"Kolom teks '{text_column}' tidak ditemukan. "
            f"Kolom yang tersedia: {list(df.columns)}"
        )

    df = df.copy()

    analyzer = create_vader_analyzer(
        use_custom_lexicon=use_custom_lexicon
    )

    sentiment_results = df[text_column].apply(
        lambda text: analyze_sentiment_text(text, analyzer)
    )

    sentiment_df = pd.DataFrame(sentiment_results.tolist())

    df[VADER_NEG_COLUMN] = sentiment_df[VADER_NEG_COLUMN]
    df[VADER_NEU_COLUMN] = sentiment_df[VADER_NEU_COLUMN]
    df[VADER_POS_COLUMN] = sentiment_df[VADER_POS_COLUMN]
    df[VADER_COMPOUND_COLUMN] = sentiment_df[VADER_COMPOUND_COLUMN]
    df[SENTIMENT_CATEGORY_COLUMN] = sentiment_df[SENTIMENT_CATEGORY_COLUMN]

    return df


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":
    sample_reviews = [
        "aplikasi jmo mudah guna sangat bantu",
        "login gagal terus aplikasi ribet",
        "aplikasi biasa saja",
        "klaim jht lancar cepat mudah",
    ]

    analyzer = create_vader_analyzer()

    for review in sample_reviews:
        result = analyze_sentiment_text(review, analyzer)

        print("Review:")
        print(review)

        print("Hasil VADER:")
        print(result)

        print("-" * 50)