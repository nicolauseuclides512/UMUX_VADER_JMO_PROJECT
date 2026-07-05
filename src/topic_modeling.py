import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

from src.config import (
    CLEAN_TEXT_COLUMN,
    PREDICTED_LABEL_COLUMN,
    PREDICTED_DIMENSION_COLUMN,
    SENTIMENT_CATEGORY_COLUMN,
    VADER_COMPOUND_COLUMN,
    UMUX_SCORE_COLUMN,
    UMUX_LABEL_MAP,
    RANDOM_STATE,
)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_N_TOPICS = 5
DEFAULT_TOP_N_WORDS = 10
MIN_DOCUMENTS_PER_GROUP = 20


# Stopwords khusus untuk topic modeling.
# Stopwords ini hanya dipakai pada tahap topic modeling,
# tidak mengubah preprocessing utama.
TOPIC_MODELING_STOPWORDS = {
    # kata umum / intensifier
    "sangat",
    "sekali",
    "banget",
    "amat",
    "cukup",
    "lumayan",
    "lebih",
    "paling",
    "terlalu",

    # kata umum aplikasi
    "aplikasi",
    "apk",
    "app",
    "apps",
    "jmo",
    "nya",
    "aja",
    "saja",
    "kan",
    "dong",
    "nih",
    "sih",
    "deh",
    "lah",
    "ya",
    "y",
    "min",
    "admin",

    # kata umum pujian yang terlalu dominan
    "bagus",
    "baik",
    "mantap",
    "oke",
    "ok",
    "keren",
    "top",
    "good",
    "nice",
    "sip",
    "jos",
    "mantul",
    "terbaik",

    # kata umum sapaan/permohonan
    "terima",
    "kasih",
    "terimakasih",
    "thanks",
    "thank",
    "mohon",
    "tolong",
    "semoga",

    # kata umum lain yang biasanya kurang membentuk tema UX
    "jadi",
    "buat",
    "bikin",
    "dapat",
    "dapet",
    "pakai",
    "guna",
    "guna aplikasi",
}


# ============================================================
# BASIC HELPER
# ============================================================

def safe_text(value):
    """
    Mengubah nilai menjadi teks aman.
    """
    if pd.isna(value):
        return ""

    return str(value).strip()


def get_label_name(label):
    """
    Mengubah label numerik menjadi nama dimensi UMUX-Lite.
    """
    try:
        label = int(label)
    except (ValueError, TypeError):
        return "Unknown"

    return UMUX_LABEL_MAP.get(label, "Unknown")


def prepare_topic_dataframe(
    df,
    text_column=CLEAN_TEXT_COLUMN,
    label_column=PREDICTED_LABEL_COLUMN,
):
    """
    Menyiapkan dataframe untuk topic modeling.

    Data yang digunakan:
    - clean_text tidak kosong
    - predicted_umux_label tersedia
    """
    df = df.copy()

    if text_column not in df.columns:
        raise ValueError(
            f"Kolom teks '{text_column}' tidak ditemukan. "
            f"Kolom tersedia: {list(df.columns)}"
        )

    if label_column not in df.columns:
        raise ValueError(
            f"Kolom label '{label_column}' tidak ditemukan. "
            f"Kolom tersedia: {list(df.columns)}"
        )

    df[text_column] = df[text_column].apply(safe_text)

    df = df[df[text_column] != ""].copy()

    df[label_column] = pd.to_numeric(
        df[label_column],
        errors="coerce"
    )

    df = df.dropna(subset=[label_column]).copy()
    df[label_column] = df[label_column].astype(int)

    return df


# ============================================================
# TOPIC MODELING CORE
# ============================================================

def create_tfidf_vectorizer(min_df=2):
    """
    Membuat TF-IDF vectorizer untuk topic modeling.

    Stopwords pada fungsi ini khusus untuk topic modeling agar
    kata-kata umum tidak mendominasi pembentukan topik.
    """
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=0.90,
        max_features=3000,
        sublinear_tf=True,
        stop_words=list(TOPIC_MODELING_STOPWORDS),
    )

    return vectorizer


def get_top_words_for_topic(topic_weights, feature_names, top_n_words):
    """
    Mengambil kata kunci teratas untuk satu topic.
    """
    top_indices = topic_weights.argsort()[::-1][:top_n_words]

    top_words = [
        feature_names[index]
        for index in top_indices
    ]

    return top_words


def interpret_topic_from_keywords(keywords):
    """
    Memberikan interpretasi awal berdasarkan keyword.

    Interpretasi ini hanya bantuan awal.
    Nama topik final tetap sebaiknya dicek manual oleh peneliti.
    """
    keyword_text = " ".join(keywords).lower()

    login_keywords = [
        "login",
        "masuk",
        "otp",
        "kode",
        "verifikasi",
        "akses",
        "daftar",
        "registrasi",
    ]

    performance_keywords = [
        "error",
        "server",
        "loading",
        "lambat",
        "lemot",
        "gagal",
        "macet",
        "hang",
        "crash",
    ]

    claim_keywords = [
        "klaim",
        "jht",
        "cair",
        "pencairan",
        "saldo",
        "rekening",
        "dana",
    ]

    usefulness_keywords = [
        "bantu",
        "membantu",
        "manfaat",
        "bermanfaat",
        "layanan",
        "informasi",
        "kartu",
        "peserta",
    ]

    ease_keywords = [
        "mudah",
        "gampang",
        "praktis",
        "sulit",
        "susah",
        "ribet",
        "bingung",
        "akses",
    ]

    if any(keyword in keyword_text for keyword in claim_keywords):
        return "Layanan klaim, saldo, atau pencairan"

    if any(keyword in keyword_text for keyword in login_keywords):
        return "Masalah login, akses, atau verifikasi"

    if any(keyword in keyword_text for keyword in performance_keywords):
        return "Masalah performa aplikasi atau gangguan teknis"

    if any(keyword in keyword_text for keyword in usefulness_keywords):
        return "Manfaat aplikasi dan dukungan layanan"

    if any(keyword in keyword_text for keyword in ease_keywords):
        return "Kemudahan atau kesulitan penggunaan"

    return "Topik umum berdasarkan kata kunci dominan"


def run_nmf_topic_modeling_for_group(
    group_df,
    label_value,
    text_column=CLEAN_TEXT_COLUMN,
    n_topics=DEFAULT_N_TOPICS,
    top_n_words=DEFAULT_TOP_N_WORDS,
):
    """
    Menjalankan topic modeling untuk satu kelompok label UMUX-Lite.

    Returns
    -------
    topic_keywords_df : pandas.DataFrame
        Kata kunci untuk setiap topic.

    document_topic_df : pandas.DataFrame
        Data review dengan topic dominan.

    topic_summary_df : pandas.DataFrame
        Ringkasan topic berdasarkan jumlah review dan sentimen.
    """
    group_df = group_df.copy()
    group_df[text_column] = group_df[text_column].apply(safe_text)

    group_df = group_df[group_df[text_column] != ""].copy()

    total_documents = len(group_df)

    if total_documents < MIN_DOCUMENTS_PER_GROUP:
        print(
            f"Label {label_value} dilewati karena jumlah dokumen "
            f"hanya {total_documents}."
        )

        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    # Untuk kelompok kecil, min_df dibuat 1 agar tidak semua kata hilang.
    if total_documents < 50:
        min_df = 1
    else:
        min_df = 2

    vectorizer = create_tfidf_vectorizer(min_df=min_df)

    tfidf_matrix = vectorizer.fit_transform(group_df[text_column])

    feature_names = vectorizer.get_feature_names_out()

    total_features = len(feature_names)

    if total_features == 0:
        print(f"Label {label_value} dilewati karena tidak ada fitur TF-IDF.")
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    adjusted_n_topics = min(n_topics, total_features, total_documents)

    if adjusted_n_topics < 1:
        print(f"Label {label_value} dilewati karena topic tidak valid.")
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    nmf_model = NMF(
        n_components=adjusted_n_topics,
        random_state=RANDOM_STATE,
        init="nndsvda",
        max_iter=500,
    )

    document_topic_matrix = nmf_model.fit_transform(tfidf_matrix)
    topic_word_matrix = nmf_model.components_

    dominant_topic = np.argmax(document_topic_matrix, axis=1)
    dominant_topic_score = np.max(document_topic_matrix, axis=1)

    label_name = get_label_name(label_value)

    # ------------------------------------------------------------
    # Topic keywords
    # ------------------------------------------------------------
    topic_keyword_rows = []

    for topic_index, topic_weights in enumerate(topic_word_matrix):
        top_words = get_top_words_for_topic(
            topic_weights=topic_weights,
            feature_names=feature_names,
            top_n_words=top_n_words,
        )

        topic_keyword_rows.append({
            "umux_label": label_value,
            "umux_dimension": label_name,
            "topic_id": topic_index + 1,
            "top_keywords": ", ".join(top_words),
            "topic_interpretation_initial": interpret_topic_from_keywords(top_words),
        })

    topic_keywords_df = pd.DataFrame(topic_keyword_rows)

    # ------------------------------------------------------------
    # Document topic assignment
    # ------------------------------------------------------------
    document_topic_df = group_df.copy()
    document_topic_df["topic_id"] = dominant_topic + 1
    document_topic_df["topic_score"] = dominant_topic_score
    document_topic_df["umux_label"] = label_value
    document_topic_df["umux_dimension"] = label_name

    # ------------------------------------------------------------
    # Topic summary
    # ------------------------------------------------------------
    topic_summary_rows = []

    for topic_id in sorted(document_topic_df["topic_id"].unique()):
        topic_docs = document_topic_df[
            document_topic_df["topic_id"] == topic_id
        ].copy()

        keyword_row = topic_keywords_df[
            topic_keywords_df["topic_id"] == topic_id
        ]

        if not keyword_row.empty:
            top_keywords = keyword_row.iloc[0]["top_keywords"]
            interpretation = keyword_row.iloc[0]["topic_interpretation_initial"]
        else:
            top_keywords = ""
            interpretation = ""

        row = {
            "umux_label": label_value,
            "umux_dimension": label_name,
            "topic_id": topic_id,
            "total_review": len(topic_docs),
            "top_keywords": top_keywords,
            "topic_interpretation_initial": interpretation,
        }

        if SENTIMENT_CATEGORY_COLUMN in topic_docs.columns:
            sentiment_counts = topic_docs[SENTIMENT_CATEGORY_COLUMN].value_counts()

            row["positive_count"] = int(sentiment_counts.get("positive", 0))
            row["neutral_count"] = int(sentiment_counts.get("neutral", 0))
            row["negative_count"] = int(sentiment_counts.get("negative", 0))

            if len(topic_docs) > 0:
                row["positive_percentage"] = round(
                    row["positive_count"] / len(topic_docs) * 100,
                    2
                )
                row["neutral_percentage"] = round(
                    row["neutral_count"] / len(topic_docs) * 100,
                    2
                )
                row["negative_percentage"] = round(
                    row["negative_count"] / len(topic_docs) * 100,
                    2
                )
            else:
                row["positive_percentage"] = 0
                row["neutral_percentage"] = 0
                row["negative_percentage"] = 0

        if VADER_COMPOUND_COLUMN in topic_docs.columns:
            row["avg_vader_compound"] = round(
                topic_docs[VADER_COMPOUND_COLUMN].mean(),
                4
            )

        if UMUX_SCORE_COLUMN in topic_docs.columns:
            row["avg_umux_score_1_7"] = round(
                topic_docs[UMUX_SCORE_COLUMN].mean(),
                4
            )

        topic_summary_rows.append(row)

    topic_summary_df = pd.DataFrame(topic_summary_rows)

    return (
        topic_keywords_df,
        document_topic_df,
        topic_summary_df,
    )


def run_topic_modeling_by_umux_label(
    df,
    labels_to_analyze=None,
    text_column=CLEAN_TEXT_COLUMN,
    label_column=PREDICTED_LABEL_COLUMN,
    n_topics=DEFAULT_N_TOPICS,
    top_n_words=DEFAULT_TOP_N_WORDS,
):
    """
    Menjalankan topic modeling berdasarkan kelompok label UMUX-Lite.

    Default:
    hanya label 1, 2, dan 3 yang dianalisis karena relevan dengan UMUX-Lite.
    """
    if labels_to_analyze is None:
        labels_to_analyze = [1, 2, 3]

    df = prepare_topic_dataframe(
        df=df,
        text_column=text_column,
        label_column=label_column,
    )

    all_topic_keywords = []
    all_document_topics = []
    all_topic_summaries = []

    for label_value in labels_to_analyze:
        group_df = df[df[label_column] == label_value].copy()

        print("=" * 80)
        print(
            f"Menjalankan topic modeling untuk label {label_value} "
            f"({get_label_name(label_value)})"
        )
        print(f"Jumlah review: {len(group_df)}")

        (
            topic_keywords_df,
            document_topic_df,
            topic_summary_df,
        ) = run_nmf_topic_modeling_for_group(
            group_df=group_df,
            label_value=label_value,
            text_column=text_column,
            n_topics=n_topics,
            top_n_words=top_n_words,
        )

        if not topic_keywords_df.empty:
            all_topic_keywords.append(topic_keywords_df)

        if not document_topic_df.empty:
            all_document_topics.append(document_topic_df)

        if not topic_summary_df.empty:
            all_topic_summaries.append(topic_summary_df)

    if all_topic_keywords:
        final_topic_keywords = pd.concat(
            all_topic_keywords,
            ignore_index=True,
        )
    else:
        final_topic_keywords = pd.DataFrame()

    if all_document_topics:
        final_document_topics = pd.concat(
            all_document_topics,
            ignore_index=True,
        )
    else:
        final_document_topics = pd.DataFrame()

    if all_topic_summaries:
        final_topic_summary = pd.concat(
            all_topic_summaries,
            ignore_index=True,
        )
    else:
        final_topic_summary = pd.DataFrame()

    return {
        "topic_keywords": final_topic_keywords,
        "document_topics": final_document_topics,
        "topic_summary": final_topic_summary,
    }