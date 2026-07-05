from pathlib import Path
import pandas as pd

from src.config import (
    OUTPUT_DIR,
    OUTPUT_CSV_PATH,
    OUTPUT_EXCEL_PATH,
    CLEAN_TEXT_COLUMN,
    PREDICTED_LABEL_COLUMN,
)

from src.topic_modeling import run_topic_modeling_by_umux_label


# ============================================================
# OUTPUT CONFIGURATION
# ============================================================

TOPIC_MODELING_EXCEL_PATH = OUTPUT_DIR / "topic_modeling_results.xlsx"
TOPIC_KEYWORDS_CSV_PATH = OUTPUT_DIR / "topic_keywords.csv"
TOPIC_SUMMARY_CSV_PATH = OUTPUT_DIR / "topic_summary.csv"
DOCUMENT_TOPICS_CSV_PATH = OUTPUT_DIR / "document_topics.csv"


# ============================================================
# DATA LOADER
# ============================================================

def load_pipeline_result():
    """
    Membaca hasil pipeline UMUX-Lite + VADER.

    Prioritas:
    1. output/hasil_umux_vader.csv
    2. output/hasil_umux_vader.xlsx
    """
    csv_path = Path(OUTPUT_CSV_PATH)
    excel_path = Path(OUTPUT_EXCEL_PATH)

    if csv_path.exists():
        print(f"Membaca file CSV: {csv_path}")
        return pd.read_csv(csv_path)

    if excel_path.exists():
        print(f"Membaca file Excel: {excel_path}")
        return pd.read_excel(excel_path, sheet_name="detail_result")

    raise FileNotFoundError(
        "File hasil pipeline tidak ditemukan.\n"
        "Jalankan terlebih dahulu:\n"
        "python run_umux_vader_pipeline.py"
    )


# ============================================================
# EXPORT
# ============================================================

def safe_sheet_name(sheet_name):
    """
    Membatasi nama sheet Excel maksimal 31 karakter.
    """
    invalid_chars = ["\\", "/", "*", "?", ":", "[", "]"]

    for char in invalid_chars:
        sheet_name = sheet_name.replace(char, "_")

    return sheet_name[:31]


def save_topic_modeling_results(results):
    """
    Menyimpan hasil topic modeling ke Excel dan CSV.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    topic_keywords = results["topic_keywords"]
    topic_summary = results["topic_summary"]
    document_topics = results["document_topics"]

    with pd.ExcelWriter(TOPIC_MODELING_EXCEL_PATH, engine="openpyxl") as writer:
        topic_summary.to_excel(
            writer,
            sheet_name=safe_sheet_name("topic_summary"),
            index=False,
        )

        topic_keywords.to_excel(
            writer,
            sheet_name=safe_sheet_name("topic_keywords"),
            index=False,
        )

        document_topics.to_excel(
            writer,
            sheet_name=safe_sheet_name("document_topics"),
            index=False,
        )

    topic_keywords.to_csv(
        TOPIC_KEYWORDS_CSV_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    topic_summary.to_csv(
        TOPIC_SUMMARY_CSV_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    document_topics.to_csv(
        DOCUMENT_TOPICS_CSV_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    return {
        "excel_output": TOPIC_MODELING_EXCEL_PATH,
        "topic_keywords_csv": TOPIC_KEYWORDS_CSV_PATH,
        "topic_summary_csv": TOPIC_SUMMARY_CSV_PATH,
        "document_topics_csv": DOCUMENT_TOPICS_CSV_PATH,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("Memulai topic modeling berbasis UMUX-Lite...")

    df = load_pipeline_result()

    print(f"Jumlah data awal: {len(df)}")

    required_columns = [
        CLEAN_TEXT_COLUMN,
        PREDICTED_LABEL_COLUMN,
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Kolom berikut tidak ditemukan: {missing_columns}\n"
            f"Pastikan sudah menjalankan python run_umux_vader_pipeline.py"
        )

    results = run_topic_modeling_by_umux_label(
        df=df,
        labels_to_analyze=[1, 2, 3],
        text_column=CLEAN_TEXT_COLUMN,
        label_column=PREDICTED_LABEL_COLUMN,
        n_topics=5,
        top_n_words=10,
    )

    print("\nRingkasan topic modeling:")
    print(results["topic_summary"])

    saved_paths = save_topic_modeling_results(results)

    print("\nFile berhasil dibuat:")
    for name, path in saved_paths.items():
        print(f"- {name}: {path}")

    print("\nSelesai.")


if __name__ == "__main__":
    main()