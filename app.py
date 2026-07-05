import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path

from src.config import (
    OUTPUT_CSV_PATH,
    OUTPUT_EXCEL_PATH,
    TEXT_COLUMN,
    CLEAN_TEXT_COLUMN,
    PREDICTED_LABEL_COLUMN,
    PREDICTED_DIMENSION_COLUMN,
    SENTIMENT_CATEGORY_COLUMN,
    VADER_COMPOUND_COLUMN,
    UMUX_SCORE_COLUMN,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="UMUX-Lite + VADER Dashboard",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# DATA LOADER
# ============================================================

@st.cache_data
def load_result_data():
    """
    Membaca hasil analisis dari folder output.
    Prioritas:
    1. output/hasil_umux_vader.csv
    2. output/hasil_umux_vader.xlsx
    """
    csv_path = Path(OUTPUT_CSV_PATH)
    excel_path = Path(OUTPUT_EXCEL_PATH)

    if csv_path.exists():
        return pd.read_csv(csv_path)

    if excel_path.exists():
        return pd.read_excel(excel_path, sheet_name="detail_result")

    return None


def validate_result_columns(df):
    """
    Memastikan kolom penting tersedia pada data hasil analisis.
    """
    required_columns = [
        TEXT_COLUMN,
        CLEAN_TEXT_COLUMN,
        PREDICTED_LABEL_COLUMN,
        PREDICTED_DIMENSION_COLUMN,
        SENTIMENT_CATEGORY_COLUMN,
        VADER_COMPOUND_COLUMN,
        UMUX_SCORE_COLUMN,
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    return missing_columns


# ============================================================
# SIDEBAR FILTER
# ============================================================

def sidebar_filter(df):
    """
    Membuat filter pada sidebar.
    """
    st.sidebar.header("Filter Data")

    filtered_df = df.copy()

    if PREDICTED_DIMENSION_COLUMN in filtered_df.columns:
        dimension_options = sorted(
            filtered_df[PREDICTED_DIMENSION_COLUMN]
            .dropna()
            .astype(str)
            .unique()
        )

        selected_dimensions = st.sidebar.multiselect(
            "Pilih dimensi UMUX-Lite",
            options=dimension_options,
            default=dimension_options,
        )

        filtered_df = filtered_df[
            filtered_df[PREDICTED_DIMENSION_COLUMN].astype(str).isin(selected_dimensions)
        ]

    if SENTIMENT_CATEGORY_COLUMN in filtered_df.columns:
        sentiment_options = sorted(
            filtered_df[SENTIMENT_CATEGORY_COLUMN]
            .dropna()
            .astype(str)
            .unique()
        )

        selected_sentiments = st.sidebar.multiselect(
            "Pilih kategori sentimen",
            options=sentiment_options,
            default=sentiment_options,
        )

        filtered_df = filtered_df[
            filtered_df[SENTIMENT_CATEGORY_COLUMN].astype(str).isin(selected_sentiments)
        ]

    if UMUX_SCORE_COLUMN in filtered_df.columns:
        min_score = float(filtered_df[UMUX_SCORE_COLUMN].min())
        max_score = float(filtered_df[UMUX_SCORE_COLUMN].max())

        selected_score_range = st.sidebar.slider(
            "Rentang skor UMUX-Lite",
            min_value=1.0,
            max_value=7.0,
            value=(min_score, max_score),
            step=0.1,
        )

        filtered_df = filtered_df[
            (filtered_df[UMUX_SCORE_COLUMN] >= selected_score_range[0])
            & (filtered_df[UMUX_SCORE_COLUMN] <= selected_score_range[1])
        ]

    return filtered_df


# ============================================================
# SUMMARY METRICS
# ============================================================

def show_summary_metrics(df):
    """
    Menampilkan ringkasan metrik utama.
    """
    total_review = len(df)

    relevant_df = df[
        df[PREDICTED_LABEL_COLUMN].astype(int).isin([1, 2, 3])
    ].copy()

    total_relevant = len(relevant_df)
    total_irrelevant = total_review - total_relevant

    avg_compound = df[VADER_COMPOUND_COLUMN].mean()
    avg_umux_score = relevant_df[UMUX_SCORE_COLUMN].mean()

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total Review", f"{total_review:,}")
    col2.metric("Review Relevan", f"{total_relevant:,}")
    col3.metric("Review Tidak Relevan", f"{total_irrelevant:,}")
    col4.metric("Rata-rata Compound", f"{avg_compound:.4f}")
    col5.metric("Rata-rata UMUX 1–7", f"{avg_umux_score:.2f}")


# ============================================================
# CHARTS
# ============================================================

def show_label_distribution(df):
    """
    Visualisasi distribusi label UMUX-Lite.
    """
    st.subheader("Distribusi Label UMUX-Lite")

    label_df = (
        df[PREDICTED_LABEL_COLUMN]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    label_df.columns = ["label", "total_review"]

    fig = px.bar(
        label_df,
        x="label",
        y="total_review",
        text="total_review",
        title="Jumlah Review Berdasarkan Label UMUX-Lite",
    )

    fig.update_layout(
        xaxis_title="Label UMUX-Lite",
        yaxis_title="Jumlah Review",
    )

    st.plotly_chart(fig, use_container_width=True)


def show_dimension_distribution(df):
    """
    Visualisasi distribusi dimensi UMUX-Lite.
    """
    st.subheader("Distribusi Dimensi UMUX-Lite")

    dimension_df = (
        df[PREDICTED_DIMENSION_COLUMN]
        .value_counts()
        .reset_index()
    )

    dimension_df.columns = ["dimension", "total_review"]

    fig = px.bar(
        dimension_df,
        x="total_review",
        y="dimension",
        orientation="h",
        text="total_review",
        title="Jumlah Review Berdasarkan Dimensi UMUX-Lite",
    )

    fig.update_layout(
        xaxis_title="Jumlah Review",
        yaxis_title="Dimensi UMUX-Lite",
    )

    st.plotly_chart(fig, use_container_width=True)


def show_sentiment_distribution(df):
    """
    Visualisasi distribusi sentimen VADER.
    """
    st.subheader("Distribusi Sentimen VADER")

    sentiment_df = (
        df[SENTIMENT_CATEGORY_COLUMN]
        .value_counts()
        .reset_index()
    )

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
    """
    Visualisasi rata-rata skor UMUX-Lite per dimensi.
    """
    st.subheader("Rata-rata Skor UMUX-Lite per Dimensi")

    relevant_df = df[
        df[PREDICTED_LABEL_COLUMN].astype(int).isin([1, 2, 3])
    ].copy()

    score_df = (
        relevant_df
        .groupby(PREDICTED_DIMENSION_COLUMN)
        .agg(
            total_review=(UMUX_SCORE_COLUMN, "count"),
            average_score=(UMUX_SCORE_COLUMN, "mean"),
        )
        .reset_index()
    )

    score_df["average_score"] = score_df["average_score"].round(2)

    fig = px.bar(
        score_df,
        x=PREDICTED_DIMENSION_COLUMN,
        y="average_score",
        text="average_score",
        title="Rata-rata Skor UMUX-Lite 1–7 Berdasarkan Dimensi",
    )

    fig.update_layout(
        xaxis_title="Dimensi UMUX-Lite",
        yaxis_title="Rata-rata Skor UMUX-Lite",
        yaxis_range=[1, 7],
    )

    st.plotly_chart(fig, use_container_width=True)


def show_compound_vs_umux_score(df):
    """
    Visualisasi hubungan VADER compound score dan skor UMUX-Lite.
    """
    st.subheader("Hubungan VADER Compound Score dan Skor UMUX-Lite")

    fig = px.scatter(
        df,
        x=VADER_COMPOUND_COLUMN,
        y=UMUX_SCORE_COLUMN,
        color=SENTIMENT_CATEGORY_COLUMN,
        hover_data=[
            TEXT_COLUMN,
            PREDICTED_DIMENSION_COLUMN,
            SENTIMENT_CATEGORY_COLUMN,
        ],
        title="VADER Compound Score vs UMUX-Lite Score",
    )

    fig.update_layout(
        xaxis_title="VADER Compound Score",
        yaxis_title="UMUX-Lite Score 1–7",
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# DETAIL TABLE
# ============================================================

def show_detail_table(df):
    """
    Menampilkan tabel detail hasil analisis.
    """
    st.subheader("Detail Hasil Analisis")

    selected_columns = [
        TEXT_COLUMN,
        CLEAN_TEXT_COLUMN,
        PREDICTED_LABEL_COLUMN,
        PREDICTED_DIMENSION_COLUMN,
        SENTIMENT_CATEGORY_COLUMN,
        VADER_COMPOUND_COLUMN,
        UMUX_SCORE_COLUMN,
    ]

    available_columns = [
        column for column in selected_columns
        if column in df.columns
    ]

    st.dataframe(
        df[available_columns],
        use_container_width=True,
        height=500,
    )

    csv_data = df[available_columns].to_csv(
        index=False,
        encoding="utf-8-sig"
    ).encode("utf-8-sig")

    st.download_button(
        label="Download data terfilter sebagai CSV",
        data=csv_data,
        file_name="filtered_umux_vader_result.csv",
        mime="text/csv",
    )


# ============================================================
# MAIN APP
# ============================================================

def main():
    st.title("Dashboard UMUX-Lite dan VADER")
    st.write(
        "Dashboard ini menampilkan hasil analisis review aplikasi JMO "
        "menggunakan klasifikasi UMUX-Lite dan VADER Sentiment Analysis."
    )

    df = load_result_data()

    if df is None:
        st.warning(
            "File hasil analisis belum ditemukan. "
            "Jalankan terlebih dahulu file berikut:"
        )

        st.code(
            "python train_umux_labeler.py\npython run_umux_vader_pipeline.py",
            language="bash",
        )

        st.stop()

    missing_columns = validate_result_columns(df)

    if missing_columns:
        st.error("Beberapa kolom penting tidak ditemukan pada file hasil analisis.")
        st.write("Kolom yang hilang:")
        st.write(missing_columns)

        st.write("Kolom yang tersedia:")
        st.write(list(df.columns))

        st.stop()

    filtered_df = sidebar_filter(df)

    if len(filtered_df) == 0:
        st.warning("Tidak ada data yang sesuai dengan filter.")
        st.stop()

    show_summary_metrics(filtered_df)

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