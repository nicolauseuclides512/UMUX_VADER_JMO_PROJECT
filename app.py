from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import (
    CLEAN_TEXT_COLUMN,
    OUTPUT_CSV_PATH,
    OUTPUT_EXCEL_PATH,
    PREDICTED_DIMENSION_COLUMN,
    PREDICTED_LABEL_COLUMN,
    SENTIMENT_CATEGORY_COLUMN,
    TEXT_COLUMN,
    UMUX_SCORE_COLUMN,
    VADER_COMPOUND_COLUMN,
)


st.set_page_config(
    page_title="UMUX-Lite + VADER Dashboard",
    layout="wide",
)


RESULT_COLUMNS = [
    PREDICTED_LABEL_COLUMN,
    PREDICTED_DIMENSION_COLUMN,
    SENTIMENT_CATEGORY_COLUMN,
    VADER_COMPOUND_COLUMN,
    UMUX_SCORE_COLUMN,
]

DISPLAY_COLUMNS = [
    TEXT_COLUMN,
    CLEAN_TEXT_COLUMN,
    PREDICTED_LABEL_COLUMN,
    PREDICTED_DIMENSION_COLUMN,
    SENTIMENT_CATEGORY_COLUMN,
    VADER_COMPOUND_COLUMN,
    UMUX_SCORE_COLUMN,
]

SAMPLE_CSV_PATH = Path("output/sample_hasil_umux_vader.csv")
SAMPLE_EXCEL_PATH = Path("output/sample_hasil_umux_vader.xlsx")
TRAINING_REPORT_PATH = Path("output/training_classification_report.txt")
TOPIC_SUMMARY_PATH = Path("output/topic_summary.csv")
TOPIC_KEYWORDS_PATH = Path("output/topic_keywords.csv")
DOCUMENT_TOPICS_PATH = Path("output/document_topics.csv")


@st.cache_data(show_spinner=False)
def read_csv(path):
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def read_excel(path):
    return pd.read_excel(path, sheet_name="detail_result")


@st.cache_data(show_spinner=False)
def read_text_file(path):
    return Path(path).read_text(encoding="utf-8")


def read_uploaded_file(uploaded_file):
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)

    return pd.read_excel(uploaded_file, sheet_name="detail_result")


def load_repository_result():
    csv_path = Path(OUTPUT_CSV_PATH)
    excel_path = Path(OUTPUT_EXCEL_PATH)

    if csv_path.exists():
        return read_csv(csv_path), csv_path

    if excel_path.exists():
        return read_excel(excel_path), excel_path

    if SAMPLE_CSV_PATH.exists():
        return read_csv(SAMPLE_CSV_PATH), SAMPLE_CSV_PATH

    if SAMPLE_EXCEL_PATH.exists():
        return read_excel(SAMPLE_EXCEL_PATH), SAMPLE_EXCEL_PATH

    return None, None


def prepare_result_data(df):
    df = df.copy()

    for column in [PREDICTED_LABEL_COLUMN, VADER_COMPOUND_COLUMN, UMUX_SCORE_COLUMN]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df.dropna(subset=RESULT_COLUMNS)


def validate_result_columns(df):
    return [column for column in RESULT_COLUMNS if column not in df.columns]


def render_data_source_control():
    st.sidebar.header("Sumber Data")

    source = st.sidebar.radio(
        "Pilih sumber data",
        options=["File output repository", "Upload file hasil analisis"],
    )

    if source == "Upload file hasil analisis":
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV/XLSX hasil pipeline",
            type=["csv", "xlsx"],
        )

        if uploaded_file is None:
            st.info("Upload file hasil pipeline untuk mulai melihat dashboard.")
            st.stop()

        return read_uploaded_file(uploaded_file), uploaded_file.name

    return load_repository_result()


def sidebar_filter(df):
    st.sidebar.header("Filter Data")
    filtered_df = df.copy()

    dimension_options = sorted(
        filtered_df[PREDICTED_DIMENSION_COLUMN].dropna().astype(str).unique()
    )
    if dimension_options:
        selected_dimensions = st.sidebar.multiselect(
            "Dimensi UMUX-Lite",
            options=dimension_options,
            default=dimension_options,
        )
        filtered_df = filtered_df[
            filtered_df[PREDICTED_DIMENSION_COLUMN].astype(str).isin(selected_dimensions)
        ]

    sentiment_options = sorted(
        filtered_df[SENTIMENT_CATEGORY_COLUMN].dropna().astype(str).unique()
    )
    if sentiment_options:
        selected_sentiments = st.sidebar.multiselect(
            "Kategori sentimen",
            options=sentiment_options,
            default=sentiment_options,
        )
        filtered_df = filtered_df[
            filtered_df[SENTIMENT_CATEGORY_COLUMN].astype(str).isin(selected_sentiments)
        ]

    if filtered_df.empty:
        return filtered_df

    min_score = float(filtered_df[UMUX_SCORE_COLUMN].min())
    max_score = float(filtered_df[UMUX_SCORE_COLUMN].max())

    if min_score == max_score:
        st.sidebar.caption(f"Skor UMUX-Lite: {min_score:.2f}")
        return filtered_df

    selected_score_range = st.sidebar.slider(
        "Rentang skor UMUX-Lite",
        min_value=1.0,
        max_value=7.0,
        value=(max(1.0, min_score), min(7.0, max_score)),
        step=0.1,
    )

    return filtered_df[
        (filtered_df[UMUX_SCORE_COLUMN] >= selected_score_range[0])
        & (filtered_df[UMUX_SCORE_COLUMN] <= selected_score_range[1])
    ]


def show_summary_metrics(df):
    total_review = len(df)
    relevant_df = df[df[PREDICTED_LABEL_COLUMN].isin([1, 2, 3])].copy()
    total_relevant = len(relevant_df)
    total_irrelevant = total_review - total_relevant

    avg_compound = df[VADER_COMPOUND_COLUMN].mean()
    avg_umux_score = relevant_df[UMUX_SCORE_COLUMN].mean()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Review", f"{total_review:,}")
    col2.metric("Review Relevan", f"{total_relevant:,}")
    col3.metric("Review Tidak Relevan", f"{total_irrelevant:,}")
    col4.metric("Rata-rata Compound", f"{avg_compound:.4f}")
    col5.metric("Rata-rata UMUX 1-7", f"{avg_umux_score:.2f}")


def show_training_report():
    if not TRAINING_REPORT_PATH.exists():
        return

    st.subheader("Evaluasi Training Model")

    report_text = read_text_file(TRAINING_REPORT_PATH)
    with st.expander("Lihat classification report UMUX-Lite labeler"):
        st.code(report_text, language="text")

    st.download_button(
        label="Download classification report",
        data=report_text.encode("utf-8"),
        file_name=TRAINING_REPORT_PATH.name,
        mime="text/plain",
    )


def load_topic_modeling_data():
    topic_summary = read_csv(TOPIC_SUMMARY_PATH) if TOPIC_SUMMARY_PATH.exists() else None
    topic_keywords = read_csv(TOPIC_KEYWORDS_PATH) if TOPIC_KEYWORDS_PATH.exists() else None
    document_topics = read_csv(DOCUMENT_TOPICS_PATH) if DOCUMENT_TOPICS_PATH.exists() else None

    return topic_summary, topic_keywords, document_topics


def filter_topic_dataframe(df, selected_dimensions):
    if df is None or df.empty or "umux_dimension" not in df.columns:
        return df

    return df[df["umux_dimension"].astype(str).isin(selected_dimensions)].copy()


def show_topic_modeling_results():
    topic_summary, topic_keywords, document_topics = load_topic_modeling_data()

    if topic_summary is None or topic_summary.empty:
        return

    st.subheader("Topic Modeling UMUX-Lite")

    dimension_options = sorted(topic_summary["umux_dimension"].dropna().astype(str).unique())
    if not dimension_options:
        st.info("File topic modeling belum memiliki kolom dimensi yang bisa ditampilkan.")
        return

    selected_dimensions = st.multiselect(
        "Filter dimensi topic modeling",
        options=dimension_options,
        default=dimension_options,
    )

    if not selected_dimensions:
        st.warning("Pilih minimal satu dimensi untuk menampilkan topic modeling.")
        return

    filtered_summary = filter_topic_dataframe(topic_summary, selected_dimensions)
    filtered_keywords = filter_topic_dataframe(topic_keywords, selected_dimensions)
    filtered_documents = filter_topic_dataframe(document_topics, selected_dimensions)

    if filtered_summary.empty:
        st.warning("Tidak ada topic modeling yang sesuai dengan filter.")
        return

    total_topics = len(filtered_summary)
    total_topic_reviews = int(filtered_summary["total_review"].sum())
    avg_topic_score = (
        filtered_summary["avg_umux_score_1_7"].mean()
        if "avg_umux_score_1_7" in filtered_summary.columns
        else 0
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Topik", f"{total_topics:,}")
    col2.metric("Review Bertopik", f"{total_topic_reviews:,}")
    col3.metric("Rata-rata UMUX Topik", f"{avg_topic_score:.2f}")

    display_summary = filtered_summary.copy()
    display_summary["topic_label"] = (
        display_summary["umux_dimension"].astype(str)
        + " - Topik "
        + display_summary["topic_id"].astype(str)
    )

    topic_hover_columns = [
        column
        for column in [
            "top_keywords",
            "topic_interpretation_initial",
            "avg_vader_compound",
            "avg_umux_score_1_7",
        ]
        if column in display_summary.columns
    ]

    fig = px.bar(
        display_summary.sort_values("total_review", ascending=True),
        x="total_review",
        y="topic_label",
        color="umux_dimension",
        orientation="h",
        text="total_review",
        hover_data=topic_hover_columns,
        title="Jumlah Review per Topik",
    )
    fig.update_layout(xaxis_title="Jumlah Review", yaxis_title="Topik")
    st.plotly_chart(fig, use_container_width=True)

    tab_summary, tab_keywords, tab_documents = st.tabs(
        ["Ringkasan Topik", "Keyword Topik", "Dokumen Bertopik"]
    )

    with tab_summary:
        summary_columns = [
            "umux_label",
            "umux_dimension",
            "topic_id",
            "total_review",
            "top_keywords",
            "topic_interpretation_initial",
            "positive_count",
            "neutral_count",
            "negative_count",
            "positive_percentage",
            "neutral_percentage",
            "negative_percentage",
            "avg_vader_compound",
            "avg_umux_score_1_7",
        ]
        available_summary_columns = [
            column for column in summary_columns if column in filtered_summary.columns
        ]
        st.dataframe(
            filtered_summary[available_summary_columns],
            use_container_width=True,
            height=360,
        )

    with tab_keywords:
        if filtered_keywords is None or filtered_keywords.empty:
            st.info("File topic_keywords.csv belum tersedia.")
        else:
            st.dataframe(filtered_keywords, use_container_width=True, height=360)

    with tab_documents:
        if filtered_documents is None or filtered_documents.empty:
            st.info("File document_topics.csv belum tersedia.")
        else:
            document_columns = [
                TEXT_COLUMN,
                CLEAN_TEXT_COLUMN,
                "umux_dimension",
                "topic_id",
                "topic_score",
                SENTIMENT_CATEGORY_COLUMN,
                VADER_COMPOUND_COLUMN,
                UMUX_SCORE_COLUMN,
            ]
            available_document_columns = [
                column for column in document_columns if column in filtered_documents.columns
            ]
            st.dataframe(
                filtered_documents[available_document_columns],
                use_container_width=True,
                height=420,
            )

    download_col1, download_col2, download_col3 = st.columns(3)

    with download_col1:
        st.download_button(
            label="Download topic summary",
            data=filtered_summary.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
            file_name="filtered_topic_summary.csv",
            mime="text/csv",
        )

    if filtered_keywords is not None and not filtered_keywords.empty:
        with download_col2:
            st.download_button(
                label="Download topic keywords",
                data=filtered_keywords.to_csv(index=False, encoding="utf-8-sig").encode(
                    "utf-8-sig"
                ),
                file_name="filtered_topic_keywords.csv",
                mime="text/csv",
            )

    if filtered_documents is not None and not filtered_documents.empty:
        with download_col3:
            st.download_button(
                label="Download document topics",
                data=filtered_documents.to_csv(index=False, encoding="utf-8-sig").encode(
                    "utf-8-sig"
                ),
                file_name="filtered_document_topics.csv",
                mime="text/csv",
            )


def show_label_distribution(df):
    st.subheader("Distribusi Label UMUX-Lite")

    label_df = df[PREDICTED_LABEL_COLUMN].value_counts().sort_index().reset_index()
    label_df.columns = ["label", "total_review"]

    fig = px.bar(
        label_df,
        x="label",
        y="total_review",
        text="total_review",
        title="Jumlah Review Berdasarkan Label UMUX-Lite",
    )
    fig.update_layout(xaxis_title="Label UMUX-Lite", yaxis_title="Jumlah Review")
    st.plotly_chart(fig, use_container_width=True)


def show_dimension_distribution(df):
    st.subheader("Distribusi Dimensi UMUX-Lite")

    dimension_df = df[PREDICTED_DIMENSION_COLUMN].value_counts().reset_index()
    dimension_df.columns = ["dimension", "total_review"]

    fig = px.bar(
        dimension_df,
        x="total_review",
        y="dimension",
        orientation="h",
        text="total_review",
        title="Jumlah Review Berdasarkan Dimensi UMUX-Lite",
    )
    fig.update_layout(xaxis_title="Jumlah Review", yaxis_title="Dimensi UMUX-Lite")
    st.plotly_chart(fig, use_container_width=True)


def show_sentiment_distribution(df):
    st.subheader("Distribusi Sentimen VADER")

    sentiment_df = df[SENTIMENT_CATEGORY_COLUMN].value_counts().reset_index()
    sentiment_df.columns = ["sentiment_category", "total_review"]

    fig = px.pie(
        sentiment_df,
        names="sentiment_category",
        values="total_review",
        title="Proporsi Sentimen VADER",
        hole=0.35,
    )
    st.plotly_chart(fig, use_container_width=True)


def show_umux_score_by_dimension(df):
    st.subheader("Rata-rata Skor UMUX-Lite per Dimensi")

    relevant_df = df[df[PREDICTED_LABEL_COLUMN].isin([1, 2, 3])].copy()
    if relevant_df.empty:
        st.info("Belum ada review relevan UMUX-Lite pada data terfilter.")
        return

    score_df = (
        relevant_df.groupby(PREDICTED_DIMENSION_COLUMN)
        .agg(total_review=(UMUX_SCORE_COLUMN, "count"), average_score=(UMUX_SCORE_COLUMN, "mean"))
        .reset_index()
    )
    score_df["average_score"] = score_df["average_score"].round(2)

    fig = px.bar(
        score_df,
        x=PREDICTED_DIMENSION_COLUMN,
        y="average_score",
        text="average_score",
        title="Rata-rata Skor UMUX-Lite 1-7 Berdasarkan Dimensi",
    )
    fig.update_layout(
        xaxis_title="Dimensi UMUX-Lite",
        yaxis_title="Rata-rata Skor UMUX-Lite",
        yaxis_range=[1, 7],
    )
    st.plotly_chart(fig, use_container_width=True)


def show_compound_vs_umux_score(df):
    st.subheader("Hubungan VADER Compound Score dan Skor UMUX-Lite")

    hover_columns = [
        column
        for column in [TEXT_COLUMN, PREDICTED_DIMENSION_COLUMN, SENTIMENT_CATEGORY_COLUMN]
        if column in df.columns
    ]

    fig = px.scatter(
        df,
        x=VADER_COMPOUND_COLUMN,
        y=UMUX_SCORE_COLUMN,
        color=SENTIMENT_CATEGORY_COLUMN,
        hover_data=hover_columns,
        title="VADER Compound Score vs UMUX-Lite Score",
    )
    fig.update_layout(
        xaxis_title="VADER Compound Score",
        yaxis_title="UMUX-Lite Score 1-7",
    )
    st.plotly_chart(fig, use_container_width=True)


def show_detail_table(df):
    st.subheader("Detail Hasil Analisis")

    available_columns = [column for column in DISPLAY_COLUMNS if column in df.columns]
    if not available_columns:
        available_columns = list(df.columns)

    st.dataframe(df[available_columns], use_container_width=True, height=500)

    csv_data = df[available_columns].to_csv(index=False, encoding="utf-8-sig").encode(
        "utf-8-sig"
    )
    st.download_button(
        label="Download data terfilter sebagai CSV",
        data=csv_data,
        file_name="filtered_umux_vader_result.csv",
        mime="text/csv",
    )


def main():
    st.title("Dashboard UMUX-Lite dan VADER")
    st.write(
        "Dashboard hasil analisis review aplikasi JMO menggunakan klasifikasi "
        "UMUX-Lite dan VADER Sentiment Analysis."
    )

    df, data_source = render_data_source_control()

    if df is None:
        st.warning(
            "File hasil analisis belum ditemukan. Jalankan pipeline terlebih dahulu "
            "atau upload file CSV/XLSX hasil pipeline melalui sidebar."
        )
        st.code("python train_umux_labeler.py\npython run_umux_vader_pipeline.py", language="bash")
        st.stop()

    st.sidebar.caption(f"Data aktif: {data_source}")

    missing_columns = validate_result_columns(df)
    if missing_columns:
        st.error("Beberapa kolom penting tidak ditemukan pada file hasil analisis.")
        st.write("Kolom yang hilang:")
        st.write(missing_columns)
        st.write("Kolom yang tersedia:")
        st.write(list(df.columns))
        st.stop()

    df = prepare_result_data(df)
    if df.empty:
        st.warning("Data tidak memiliki baris valid setelah pembersihan nilai numerik.")
        st.stop()

    filtered_df = sidebar_filter(df)
    if filtered_df.empty:
        st.warning("Tidak ada data yang sesuai dengan filter.")
        st.stop()

    show_summary_metrics(filtered_df)

    st.divider()
    show_training_report()

    st.divider()
    show_topic_modeling_results()

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        show_label_distribution(filtered_df)
    with col2:
        show_sentiment_distribution(filtered_df)

    st.divider()
    show_dimension_distribution(filtered_df)

    st.divider()
    show_umux_score_by_dimension(filtered_df)

    st.divider()
    show_compound_vs_umux_score(filtered_df)

    st.divider()
    show_detail_table(filtered_df)


if __name__ == "__main__":
    main()
