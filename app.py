from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import (
    CLEAN_TEXT_COLUMN,
    DATE_COLUMN,
    OUTPUT_CSV_PATH,
    OUTPUT_EXCEL_PATH,
    PREDICTED_DIMENSION_COLUMN,
    PREDICTED_LABEL_COLUMN,
    RATING_COLUMN,
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
    RATING_COLUMN,
    DATE_COLUMN,
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
SUMMARY_REPORT_PATH = Path("output/summary_report.txt")
TRAINING_REPORT_PATH = Path("output/training_classification_report.txt")
TRAINING_EVALUATION_PATH = Path("output/training_evaluation.xlsx")
MISCLASSIFIED_REVIEWS_PATH = Path("output/misclassified_reviews.csv")
TOPIC_SUMMARY_PATH = Path("output/topic_summary.csv")
TOPIC_KEYWORDS_PATH = Path("output/topic_keywords.csv")
DOCUMENT_TOPICS_PATH = Path("output/document_topics.csv")


@st.cache_data(show_spinner=False)
def read_csv(path):
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def read_excel(path, sheet_name="detail_result"):
    try:
        return pd.read_excel(path, sheet_name=sheet_name)
    except ValueError:
        sheets = pd.read_excel(path, sheet_name=None)
        if not sheets:
            return pd.DataFrame()

        return next(iter(sheets.values()))


@st.cache_data(show_spinner=False)
def read_excel_sheets(path):
    return pd.read_excel(path, sheet_name=None)


@st.cache_data(show_spinner=False)
def read_text_file(path):
    return Path(path).read_text(encoding="utf-8")


def read_uploaded_file(uploaded_file):
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)

    try:
        return pd.read_excel(uploaded_file, sheet_name="detail_result")
    except ValueError:
        uploaded_file.seek(0)
        sheets = pd.read_excel(uploaded_file, sheet_name=None)
        if not sheets:
            return pd.DataFrame()

        return next(iter(sheets.values()))


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


def load_topic_modeling_data():
    topic_summary = read_csv(TOPIC_SUMMARY_PATH) if TOPIC_SUMMARY_PATH.exists() else None
    topic_keywords = read_csv(TOPIC_KEYWORDS_PATH) if TOPIC_KEYWORDS_PATH.exists() else None
    document_topics = read_csv(DOCUMENT_TOPICS_PATH) if DOCUMENT_TOPICS_PATH.exists() else None

    for topic_df in [topic_summary, topic_keywords, document_topics]:
        if topic_df is not None and not topic_df.empty:
            add_topic_label(topic_df)

    return topic_summary, topic_keywords, document_topics


def load_misclassification_data():
    if MISCLASSIFIED_REVIEWS_PATH.exists():
        return read_csv(MISCLASSIFIED_REVIEWS_PATH)

    return None


def add_topic_label(df):
    if "topic_label" in df.columns:
        return df

    if {"umux_dimension", "topic_id"}.issubset(df.columns):
        df["topic_label"] = (
            df["umux_dimension"].astype(str)
            + " - Topik "
            + df["topic_id"].astype(str)
        )

    return df


def prepare_result_data(df):
    df = df.copy()

    for column in [PREDICTED_LABEL_COLUMN, VADER_COMPOUND_COLUMN, UMUX_SCORE_COLUMN, RATING_COLUMN]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if DATE_COLUMN in df.columns:
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")

    return df.dropna(subset=RESULT_COLUMNS)


def validate_result_columns(df):
    return [column for column in RESULT_COLUMNS if column not in df.columns]


def render_data_source_control():
    st.sidebar.header("Sumber Data")

    source = st.sidebar.radio(
        "Pilih sumber data",
        options=["File output repository", "Upload file hasil analisis"],
        key="data_source_radio",
    )

    if source == "Upload file hasil analisis":
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV/XLSX hasil pipeline",
            type=["csv", "xlsx"],
            key="pipeline_result_uploader",
        )

        if uploaded_file is None:
            st.info("Upload file hasil pipeline untuk mulai melihat dashboard.")
            st.stop()

        return read_uploaded_file(uploaded_file), uploaded_file.name

    return load_repository_result()


def topic_document_filter(df, document_topics, selected_topics):
    if document_topics is None or document_topics.empty or "topic_label" not in document_topics.columns:
        return df

    if not selected_topics:
        return df.iloc[0:0].copy()

    topic_docs = document_topics[document_topics["topic_label"].isin(selected_topics)].copy()
    if topic_docs.empty:
        return df.iloc[0:0].copy()

    if TEXT_COLUMN in df.columns and TEXT_COLUMN in topic_docs.columns:
        return df[df[TEXT_COLUMN].astype(str).isin(topic_docs[TEXT_COLUMN].astype(str))].copy()

    if CLEAN_TEXT_COLUMN in df.columns and CLEAN_TEXT_COLUMN in topic_docs.columns:
        return df[
            df[CLEAN_TEXT_COLUMN].astype(str).isin(topic_docs[CLEAN_TEXT_COLUMN].astype(str))
        ].copy()

    return df


def apply_sidebar_filters(df, document_topics):
    st.sidebar.header("Filter")
    filtered_df = df.copy()

    if RATING_COLUMN in filtered_df.columns and filtered_df[RATING_COLUMN].notna().any():
        rating_options = sorted(filtered_df[RATING_COLUMN].dropna().unique())
        selected_ratings = st.sidebar.multiselect(
            "Rating",
            options=rating_options,
            default=rating_options,
            key="sidebar_rating_filter",
        )
        filtered_df = filtered_df[filtered_df[RATING_COLUMN].isin(selected_ratings)]

    label_options = sorted(filtered_df[PREDICTED_LABEL_COLUMN].dropna().unique())
    selected_labels = st.sidebar.multiselect(
        "Label UMUX-Lite",
        options=label_options,
        default=label_options,
        key="sidebar_umux_label_filter",
    )
    filtered_df = filtered_df[filtered_df[PREDICTED_LABEL_COLUMN].isin(selected_labels)]

    sentiment_options = sorted(filtered_df[SENTIMENT_CATEGORY_COLUMN].dropna().astype(str).unique())
    selected_sentiments = st.sidebar.multiselect(
        "Sentiment category",
        options=sentiment_options,
        default=sentiment_options,
        key="sidebar_sentiment_filter",
    )
    filtered_df = filtered_df[
        filtered_df[SENTIMENT_CATEGORY_COLUMN].astype(str).isin(selected_sentiments)
    ]

    selected_topics = []
    if document_topics is not None and not document_topics.empty and "topic_label" in document_topics.columns:
        topic_options = sorted(document_topics["topic_label"].dropna().astype(str).unique())
        selected_topics = st.sidebar.multiselect(
            "Topic",
            options=topic_options,
            default=topic_options,
            key="sidebar_topic_filter",
        )

        if set(selected_topics) != set(topic_options):
            filtered_df = topic_document_filter(filtered_df, document_topics, selected_topics)
    else:
        st.sidebar.caption("Topic belum tersedia. Jalankan run_topic_modeling.py.")

    if DATE_COLUMN in filtered_df.columns and filtered_df[DATE_COLUMN].notna().any():
        min_date = filtered_df[DATE_COLUMN].min().date()
        max_date = filtered_df[DATE_COLUMN].max().date()
        selected_date_range = st.sidebar.date_input(
            "Rentang tanggal",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="sidebar_date_range_filter",
        )

        if isinstance(selected_date_range, tuple) and len(selected_date_range) == 2:
            start_date, end_date = selected_date_range
            filtered_df = filtered_df[
                (filtered_df[DATE_COLUMN].dt.date >= start_date)
                & (filtered_df[DATE_COLUMN].dt.date <= end_date)
            ]

    return filtered_df, {
        "labels": selected_labels,
        "sentiments": selected_sentiments,
        "topics": selected_topics,
    }


def available_columns(df, columns):
    return [column for column in columns if column in df.columns]


def show_summary_metrics(df):
    if df.empty:
        st.info("Tidak ada data untuk diringkas.")
        return

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


def show_main_summary(df):
    st.subheader("Ringkasan Utama")
    show_summary_metrics(df)

    col1, col2 = st.columns(2)
    with col1:
        show_label_distribution(df, "summary_label_distribution")
    with col2:
        show_sentiment_distribution(df, "summary_sentiment_distribution")

    if SUMMARY_REPORT_PATH.exists():
        with st.expander("Ringkasan teks hasil analisis"):
            st.text(read_text_file(SUMMARY_REPORT_PATH))


def show_dataset_overview(df):
    st.subheader("Overview Dataset")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Jumlah Baris", f"{len(df):,}")
    col2.metric("Jumlah Kolom", f"{len(df.columns):,}")
    col3.metric("Kolom Teks Terisi", f"{df[TEXT_COLUMN].notna().sum():,}" if TEXT_COLUMN in df else "-")
    col4.metric("Duplikasi Review", f"{df[TEXT_COLUMN].duplicated().sum():,}" if TEXT_COLUMN in df else "-")

    if DATE_COLUMN in df.columns and df[DATE_COLUMN].notna().any():
        date_df = df.dropna(subset=[DATE_COLUMN]).copy()
        date_df["tanggal"] = date_df[DATE_COLUMN].dt.date
        trend_df = date_df.groupby("tanggal").size().reset_index(name="total_review")
        if not trend_df.empty:
            fig = px.line(
                trend_df,
                x="tanggal",
                y="total_review",
                markers=True,
                title="Tren Jumlah Review",
            )
            fig.update_layout(xaxis_title="Tanggal", yaxis_title="Jumlah Review")
            st.plotly_chart(fig, use_container_width=True, key="dataset_review_trend_chart")

    if RATING_COLUMN in df.columns and df[RATING_COLUMN].notna().any():
        rating_df = df[RATING_COLUMN].value_counts().sort_index().reset_index()
        rating_df.columns = ["rating", "total_review"]
        if not rating_df.empty:
            fig = px.bar(
                rating_df,
                x="rating",
                y="total_review",
                text="total_review",
                title="Distribusi Rating",
            )
            st.plotly_chart(fig, use_container_width=True, key="dataset_rating_distribution_chart")


def show_umux_analysis(df):
    st.subheader("UMUX-Lite Analysis")
    col1, col2 = st.columns(2)
    with col1:
        show_label_distribution(df, "umux_label_distribution")
    with col2:
        show_dimension_distribution(df, "umux_dimension_distribution")

    show_umux_score_by_dimension(df, "umux_score_by_dimension")


def show_vader_analysis(df):
    st.subheader("VADER Sentiment Analysis")
    if df.empty:
        st.info("Tidak ada data sentimen untuk ditampilkan.")
        return

    col1, col2 = st.columns(2)
    with col1:
        show_sentiment_distribution(df, "vader_sentiment_distribution")
    with col2:
        sentiment_score_df = (
            df.groupby(SENTIMENT_CATEGORY_COLUMN)
            .agg(
                total_review=(SENTIMENT_CATEGORY_COLUMN, "count"),
                avg_compound=(VADER_COMPOUND_COLUMN, "mean"),
            )
            .reset_index()
        )
        if not sentiment_score_df.empty:
            sentiment_score_df["avg_compound_text"] = sentiment_score_df[
                "avg_compound"
            ].round(4)
            fig = px.bar(
                sentiment_score_df,
                x=SENTIMENT_CATEGORY_COLUMN,
                y="avg_compound",
                text="avg_compound_text",
                title="Rata-rata Compound per Sentimen",
            )
            st.plotly_chart(fig, use_container_width=True, key="vader_avg_compound_chart")


def show_umux_vader_analysis(df):
    st.subheader("UMUX-Lite x VADER Analysis")
    if df.empty:
        st.info("Tidak ada data UMUX x VADER untuk ditampilkan.")
        return

    show_compound_vs_umux_score(df, "umux_vader_scatter_chart")

    heatmap_df = (
        df.groupby([PREDICTED_DIMENSION_COLUMN, SENTIMENT_CATEGORY_COLUMN])
        .size()
        .reset_index(name="total_review")
    )
    if heatmap_df.empty:
        st.info("Data heatmap UMUX x VADER belum tersedia.")
        return

    fig = px.bar(
        heatmap_df,
        x=PREDICTED_DIMENSION_COLUMN,
        y="total_review",
        color=SENTIMENT_CATEGORY_COLUMN,
        text="total_review",
        barmode="group",
        title="Distribusi UMUX-Lite x Sentimen",
    )
    fig.update_layout(xaxis_title="Dimensi UMUX-Lite", yaxis_title="Jumlah Review")
    st.plotly_chart(fig, use_container_width=True, key="umux_vader_heatmap_chart")


def filter_topic_dataframe(df, selected_topics):
    if df is None or df.empty or "topic_label" not in df.columns:
        return df

    if not selected_topics:
        return df.iloc[0:0].copy()

    return df[df["topic_label"].astype(str).isin(selected_topics)].copy()


def show_topic_modeling_results(topic_summary, topic_keywords, document_topics, selected_topics):
    st.subheader("Topic Modeling per UMUX-Lite Label")

    if topic_summary is None or topic_summary.empty:
        st.info("File topic_summary.csv belum tersedia. Jalankan python run_topic_modeling.py.")
        return

    filtered_summary = filter_topic_dataframe(topic_summary, selected_topics)
    filtered_keywords = filter_topic_dataframe(topic_keywords, selected_topics)
    filtered_documents = filter_topic_dataframe(document_topics, selected_topics)

    if filtered_summary.empty:
        st.warning("Tidak ada topik yang sesuai dengan filter.")
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

    fig = px.bar(
        filtered_summary.sort_values("total_review", ascending=True),
        x="total_review",
        y="topic_label",
        color="umux_dimension",
        orientation="h",
        text="total_review",
        hover_data=available_columns(
            filtered_summary,
            ["top_keywords", "topic_interpretation_initial", "avg_vader_compound", "avg_umux_score_1_7"],
        ),
        title="Jumlah Review per Topik",
    )
    fig.update_layout(xaxis_title="Jumlah Review", yaxis_title="Topik")
    st.plotly_chart(fig, use_container_width=True, key="topic_review_count_chart")

    tab_summary, tab_keywords, tab_documents = st.tabs(
        ["Ringkasan Topik", "Keyword Topik", "Dokumen Bertopik"]
    )

    with tab_summary:
        st.dataframe(
            filtered_summary,
            use_container_width=True,
            height=360,
            key="topic_summary_table",
        )

    with tab_keywords:
        if filtered_keywords is None or filtered_keywords.empty:
            st.info("File topic_keywords.csv belum tersedia.")
        else:
            st.dataframe(
                filtered_keywords,
                use_container_width=True,
                height=360,
                key="topic_keywords_table",
            )

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
            st.dataframe(
                filtered_documents[available_columns(filtered_documents, document_columns)],
                use_container_width=True,
                height=420,
                key="topic_documents_table",
            )


def show_topic_sentiment_interpretation(topic_summary, selected_topics):
    st.subheader("Topic x Sentiment Interpretation")

    if topic_summary is None or topic_summary.empty:
        st.info("File topic_summary.csv belum tersedia.")
        return

    filtered_summary = filter_topic_dataframe(topic_summary, selected_topics)
    required_columns = [
        "topic_label",
        "umux_dimension",
        "positive_count",
        "neutral_count",
        "negative_count",
    ]
    if filtered_summary.empty or not set(required_columns).issubset(filtered_summary.columns):
        st.info("Ringkasan topic x sentiment belum tersedia.")
        return

    topic_id_columns = available_columns(
        filtered_summary,
        ["topic_label", "umux_dimension", "top_keywords", "topic_interpretation_initial"],
    )
    melted_df = filtered_summary.melt(
        id_vars=topic_id_columns,
        value_vars=["positive_count", "neutral_count", "negative_count"],
        var_name="sentiment",
        value_name="total_review",
    )
    melted_df["sentiment"] = melted_df["sentiment"].str.replace("_count", "", regex=False)

    hover_columns = available_columns(
        melted_df,
        ["top_keywords", "topic_interpretation_initial"],
    )
    fig = px.bar(
        melted_df,
        x="topic_label",
        y="total_review",
        color="sentiment",
        hover_data=hover_columns,
        title="Komposisi Sentimen per Topik",
    )
    fig.update_layout(xaxis_title="Topik", yaxis_title="Jumlah Review", xaxis_tickangle=-35)
    st.plotly_chart(fig, use_container_width=True, key="topic_sentiment_composition_chart")

    interpretation_columns = [
        "topic_label",
        "top_keywords",
        "topic_interpretation_initial",
        "positive_percentage",
        "neutral_percentage",
        "negative_percentage",
        "avg_vader_compound",
        "avg_umux_score_1_7",
    ]
    st.dataframe(
        filtered_summary[available_columns(filtered_summary, interpretation_columns)],
        use_container_width=True,
        height=360,
        key="topic_sentiment_interpretation_table",
    )


def show_model_evaluation():
    st.subheader("Model Evaluation")

    if TRAINING_REPORT_PATH.exists():
        report_text = read_text_file(TRAINING_REPORT_PATH)
        st.code(report_text, language="text")
        st.download_button(
            label="Download classification report",
            data=report_text.encode("utf-8"),
            file_name=TRAINING_REPORT_PATH.name,
            mime="text/plain",
            key="download_classification_report",
        )
    else:
        st.info("File training_classification_report.txt belum tersedia.")

    if TRAINING_EVALUATION_PATH.exists():
        with st.expander("Lihat sheet training_evaluation.xlsx"):
            sheets = read_excel_sheets(TRAINING_EVALUATION_PATH)
            for sheet_name, sheet_df in sheets.items():
                st.write(sheet_name)
                st.dataframe(
                    sheet_df,
                    use_container_width=True,
                    height=260,
                    key=f"training_evaluation_{sheet_name}",
                )


def show_misclassification_analysis(filtered_df):
    st.subheader("Misclassification Analysis")

    misclassified_df = load_misclassification_data()
    if misclassified_df is None or misclassified_df.empty:
        st.info("File misclassified_reviews.csv belum tersedia.")
        return

    if TEXT_COLUMN in filtered_df.columns and TEXT_COLUMN in misclassified_df.columns:
        misclassified_df = misclassified_df[
            misclassified_df[TEXT_COLUMN].astype(str).isin(filtered_df[TEXT_COLUMN].astype(str))
        ].copy()
    elif CLEAN_TEXT_COLUMN in filtered_df.columns and CLEAN_TEXT_COLUMN in misclassified_df.columns:
        misclassified_df = misclassified_df[
            misclassified_df[CLEAN_TEXT_COLUMN].astype(str).isin(
                filtered_df[CLEAN_TEXT_COLUMN].astype(str)
            )
        ].copy()

    if misclassified_df.empty:
        st.info("Tidak ada misclassification yang sesuai dengan filter saat ini.")
        return

    col1, col2 = st.columns(2)
    col1.metric("Total Misclassified", f"{len(misclassified_df):,}")
    if "error_type" in misclassified_df.columns:
        col2.metric("Tipe Error", f"{misclassified_df['error_type'].nunique():,}")

    if "error_type" in misclassified_df.columns:
        error_df = misclassified_df["error_type"].value_counts().reset_index()
        error_df.columns = ["error_type", "total_review"]
        fig = px.bar(
            error_df,
            x="total_review",
            y="error_type",
            orientation="h",
            text="total_review",
            title="Distribusi Tipe Kesalahan Klasifikasi",
        )
        st.plotly_chart(fig, use_container_width=True, key="misclassification_error_type_chart")

    table_columns = [
        TEXT_COLUMN,
        CLEAN_TEXT_COLUMN,
        "actual_umux_dimension",
        PREDICTED_DIMENSION_COLUMN,
        "error_type",
        SENTIMENT_CATEGORY_COLUMN,
        VADER_COMPOUND_COLUMN,
        UMUX_SCORE_COLUMN,
        RATING_COLUMN,
        DATE_COLUMN,
    ]
    st.dataframe(
        misclassified_df[available_columns(misclassified_df, table_columns)],
        use_container_width=True,
        height=460,
        key="misclassification_table",
    )


def show_data_explorer(df):
    st.subheader("Data Explorer & Download")

    default_columns = available_columns(df, DISPLAY_COLUMNS)
    selected_columns = st.multiselect(
        "Pilih kolom",
        options=list(df.columns),
        default=default_columns or list(df.columns),
        key="data_explorer_column_selector",
    )

    if not selected_columns:
        st.warning("Pilih minimal satu kolom.")
        return

    st.dataframe(
        df[selected_columns],
        use_container_width=True,
        height=560,
        key="data_explorer_table",
    )
    csv_data = df[selected_columns].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        label="Download data terfilter sebagai CSV",
        data=csv_data,
        file_name="filtered_umux_vader_result.csv",
        mime="text/csv",
        key="download_filtered_data_csv",
    )


def show_label_distribution(df, chart_key):
    label_df = df[PREDICTED_LABEL_COLUMN].value_counts().sort_index().reset_index()
    label_df.columns = ["label", "total_review"]
    if label_df.empty:
        st.info("Tidak ada distribusi label untuk ditampilkan.")
        return

    fig = px.bar(
        label_df,
        x="label",
        y="total_review",
        text="total_review",
        title="Jumlah Review Berdasarkan Label UMUX-Lite",
    )
    fig.update_layout(xaxis_title="Label UMUX-Lite", yaxis_title="Jumlah Review")
    st.plotly_chart(fig, use_container_width=True, key=chart_key)


def show_dimension_distribution(df, chart_key):
    dimension_df = df[PREDICTED_DIMENSION_COLUMN].value_counts().reset_index()
    dimension_df.columns = ["dimension", "total_review"]
    if dimension_df.empty:
        st.info("Tidak ada distribusi dimensi untuk ditampilkan.")
        return

    fig = px.bar(
        dimension_df,
        x="total_review",
        y="dimension",
        orientation="h",
        text="total_review",
        title="Jumlah Review Berdasarkan Dimensi UMUX-Lite",
    )
    fig.update_layout(xaxis_title="Jumlah Review", yaxis_title="Dimensi UMUX-Lite")
    st.plotly_chart(fig, use_container_width=True, key=chart_key)


def show_sentiment_distribution(df, chart_key):
    sentiment_df = df[SENTIMENT_CATEGORY_COLUMN].value_counts().reset_index()
    sentiment_df.columns = ["sentiment_category", "total_review"]
    if sentiment_df.empty:
        st.info("Tidak ada distribusi sentimen untuk ditampilkan.")
        return

    fig = px.pie(
        sentiment_df,
        names="sentiment_category",
        values="total_review",
        title="Proporsi Sentimen VADER",
        hole=0.35,
    )
    st.plotly_chart(fig, use_container_width=True, key=chart_key)


def show_umux_score_by_dimension(df, chart_key):
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
    st.plotly_chart(fig, use_container_width=True, key=chart_key)


def show_compound_vs_umux_score(df, chart_key):
    if df.empty:
        st.info("Tidak ada data scatter untuk ditampilkan.")
        return

    hover_columns = available_columns(
        df,
        [TEXT_COLUMN, PREDICTED_DIMENSION_COLUMN, SENTIMENT_CATEGORY_COLUMN, RATING_COLUMN],
    )
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
    st.plotly_chart(fig, use_container_width=True, key=chart_key)


def main():
    st.title("Dashboard UMUX-Lite dan VADER")
    st.write(
        "Dashboard hasil analisis review aplikasi JMO menggunakan klasifikasi "
        "UMUX-Lite, VADER Sentiment Analysis, dan topic modeling."
    )

    df, data_source = render_data_source_control()

    if df is None:
        st.warning(
            "File hasil analisis belum ditemukan. Jalankan pipeline terlebih dahulu "
            "atau upload file CSV/XLSX hasil pipeline melalui sidebar."
        )
        st.code(
            "python train_umux_labeler.py\npython run_umux_vader_pipeline.py\npython run_topic_modeling.py",
            language="bash",
        )
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

    topic_summary, topic_keywords, document_topics = load_topic_modeling_data()
    filtered_df, filter_state = apply_sidebar_filters(df, document_topics)
    selected_topics = filter_state["topics"]

    if (
        not selected_topics
        and (document_topics is None or document_topics.empty)
        and topic_summary is not None
        and not topic_summary.empty
        and "topic_label" in topic_summary.columns
    ):
        selected_topics = sorted(topic_summary["topic_label"].dropna().astype(str).unique())

    if filtered_df.empty:
        st.warning("Tidak ada data yang sesuai dengan filter.")
        st.stop()

    tabs = st.tabs(
        [
            "1 Ringkasan Utama",
            "2 Overview Dataset",
            "3 UMUX-Lite Analysis",
            "4 VADER Sentiment",
            "5 UMUX x VADER",
            "6 Topic Modeling",
            "7 Topic x Sentiment",
            "8 Model Evaluation",
            "9 Misclassification",
            "10 Data Explorer",
        ]
    )

    with tabs[0]:
        show_main_summary(filtered_df)

    with tabs[1]:
        show_dataset_overview(filtered_df)

    with tabs[2]:
        show_umux_analysis(filtered_df)

    with tabs[3]:
        show_vader_analysis(filtered_df)

    with tabs[4]:
        show_umux_vader_analysis(filtered_df)

    with tabs[5]:
        show_topic_modeling_results(
            topic_summary,
            topic_keywords,
            document_topics,
            selected_topics,
        )

    with tabs[6]:
        show_topic_sentiment_interpretation(topic_summary, selected_topics)

    with tabs[7]:
        show_model_evaluation()

    with tabs[8]:
        show_misclassification_analysis(filtered_df)

    with tabs[9]:
        show_data_explorer(filtered_df)


if __name__ == "__main__":
    main()
