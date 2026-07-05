from pathlib import Path
import pandas as pd

from src.config import (
    OUTPUT_DIR,
    OUTPUT_EXCEL_PATH,
    OUTPUT_CSV_PATH,
    SUMMARY_CSV_PATH,
    PREDICTED_LABEL_COLUMN,
    PREDICTED_DIMENSION_COLUMN,
    SENTIMENT_CATEGORY_COLUMN,
    VADER_COMPOUND_COLUMN,
    UMUX_SCORE_COLUMN,
)


# ============================================================
# BASIC HELPER
# ============================================================

def create_output_directory(output_dir=OUTPUT_DIR):
    """
    Membuat folder output jika belum tersedia.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def safe_sheet_name(sheet_name):
    """
    Membatasi nama sheet Excel maksimal 31 karakter.
    Excel tidak mengizinkan nama sheet lebih dari 31 karakter.
    """
    sheet_name = str(sheet_name)

    invalid_chars = ["\\", "/", "*", "?", ":", "[", "]"]

    for char in invalid_chars:
        sheet_name = sheet_name.replace(char, "_")

    return sheet_name[:31]


def save_dataframe_to_csv(df, output_path):
    """
    Menyimpan DataFrame ke file CSV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    return output_path


def save_dataframe_to_excel(df, output_path, sheet_name="result"):
    """
    Menyimpan satu DataFrame ke file Excel.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(
            writer,
            sheet_name=safe_sheet_name(sheet_name),
            index=False,
        )

    return output_path


# ============================================================
# EVALUATION RESULT EXPORTER
# ============================================================

def save_evaluation_result_to_excel(
    evaluation_result,
    output_path,
):
    """
    Menyimpan dictionary hasil evaluasi ke file Excel.

    evaluation_result berbentuk dictionary:
    {
        "final_summary": DataFrame,
        "label_distribution": DataFrame,
        "sentiment_distribution": DataFrame,
        ...
    }
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, result_df in evaluation_result.items():
            if isinstance(result_df, pd.DataFrame):
                result_df.to_excel(
                    writer,
                    sheet_name=safe_sheet_name(sheet_name),
                    index=False,
                )

    return output_path


def save_training_evaluation_report(
    evaluation_result,
    output_path=OUTPUT_DIR / "training_evaluation.xlsx",
):
    """
    Menyimpan hasil evaluasi training model UMUX-Lite ke Excel.

    File ini biasanya digunakan setelah proses:
    train_umux_labeler.py

    Isi sheet:
    - classification_summary
    - classification_report
    - confusion_matrix
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, result_df in evaluation_result.items():
            if isinstance(result_df, pd.DataFrame):
                result_df.to_excel(
                    writer,
                    sheet_name=safe_sheet_name(sheet_name),
                    index=True,
                )

    return output_path


# ============================================================
# MAIN PIPELINE RESULT EXPORTER
# ============================================================

def save_pipeline_result(
    df,
    evaluation_result=None,
    output_excel_path=OUTPUT_EXCEL_PATH,
    output_csv_path=OUTPUT_CSV_PATH,
    summary_csv_path=SUMMARY_CSV_PATH,
):
    """
    Menyimpan hasil akhir pipeline penelitian.

    Output utama:
    1. Excel detail hasil dan evaluasi
    2. CSV detail hasil
    3. CSV summary akhir jika tersedia

    Parameters
    ----------
    df : pandas.DataFrame
        Data hasil akhir pipeline.

    evaluation_result : dict, optional
        Dictionary hasil evaluasi dari evaluate_final_result().

    output_excel_path : Path
        Lokasi file Excel output.

    output_csv_path : Path
        Lokasi file CSV detail.

    summary_csv_path : Path
        Lokasi file CSV summary.
    """
    create_output_directory()

    output_excel_path = Path(output_excel_path)
    output_csv_path = Path(output_csv_path)
    summary_csv_path = Path(summary_csv_path)

    # Simpan detail hasil ke CSV
    df.to_csv(
        output_csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    # Simpan detail dan evaluasi ke Excel
    with pd.ExcelWriter(output_excel_path, engine="openpyxl") as writer:
        df.to_excel(
            writer,
            sheet_name="detail_result",
            index=False,
        )

        if evaluation_result:
            for sheet_name, result_df in evaluation_result.items():
                if isinstance(result_df, pd.DataFrame):
                    result_df.to_excel(
                        writer,
                        sheet_name=safe_sheet_name(sheet_name),
                        index=False,
                    )

    # Simpan final summary ke CSV jika tersedia
    if evaluation_result and "final_summary" in evaluation_result:
        final_summary = evaluation_result["final_summary"]

        if isinstance(final_summary, pd.DataFrame):
            final_summary.to_csv(
                summary_csv_path,
                index=False,
                encoding="utf-8-sig",
            )

    return {
        "excel_output": output_excel_path,
        "csv_output": output_csv_path,
        "summary_output": summary_csv_path,
    }


# ============================================================
# TEXT SUMMARY GENERATOR
# ============================================================

def create_text_summary(df):
    """
    Membuat ringkasan teks sederhana dari hasil akhir pipeline.

    Fungsi ini berguna untuk ditampilkan di terminal.
    """
    required_columns = [
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

    if missing_columns:
        raise ValueError(
            f"Kolom berikut tidak ditemukan untuk membuat summary: {missing_columns}"
        )

    total_review = len(df)

    relevant_df = df[
        df[PREDICTED_LABEL_COLUMN].astype(int).isin([1, 2, 3])
    ].copy()

    total_relevant_review = len(relevant_df)
    total_irrelevant_review = total_review - total_relevant_review

    average_compound = df[VADER_COMPOUND_COLUMN].mean()
    average_umux_score = relevant_df[UMUX_SCORE_COLUMN].mean()

    sentiment_counts = df[SENTIMENT_CATEGORY_COLUMN].value_counts()

    positive_count = sentiment_counts.get("positive", 0)
    neutral_count = sentiment_counts.get("neutral", 0)
    negative_count = sentiment_counts.get("negative", 0)

    summary_text = f"""
Ringkasan Hasil Analisis UMUX-Lite dan VADER
===========================================

Total review                         : {total_review}
Review relevan UMUX-Lite             : {total_relevant_review}
Review tidak relevan UMUX-Lite       : {total_irrelevant_review}

Jumlah sentimen positif              : {positive_count}
Jumlah sentimen netral               : {neutral_count}
Jumlah sentimen negatif              : {negative_count}

Rata-rata VADER compound score       : {average_compound:.4f}
Rata-rata UMUX-Lite score 1–7        : {average_umux_score:.4f}
"""

    return summary_text.strip()


def save_text_summary(
    df,
    output_path=OUTPUT_DIR / "summary_report.txt",
):
    """
    Menyimpan ringkasan teks hasil analisis ke file .txt.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary_text = create_text_summary(df)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(summary_text)

    return output_path


# ============================================================
# PRINT HELPER
# ============================================================

def print_report_paths(saved_paths):
    """
    Menampilkan lokasi file output di terminal.
    """
    print("\nFile output berhasil dibuat:")

    for name, path in saved_paths.items():
        print(f"- {name}: {path}")


def print_text_summary(df):
    """
    Menampilkan ringkasan hasil analisis di terminal.
    """
    summary_text = create_text_summary(df)

    print("\n" + summary_text)


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        PREDICTED_LABEL_COLUMN: [1, 2, 3, 0, 1],
        PREDICTED_DIMENSION_COLUMN: [
            "Usefulness",
            "Ease of Use",
            "Usefulness + Ease of Use",
            "Tidak relevan dengan UMUX-Lite",
            "Usefulness",
        ],
        SENTIMENT_CATEGORY_COLUMN: [
            "positive",
            "negative",
            "positive",
            "neutral",
            "positive",
        ],
        VADER_COMPOUND_COLUMN: [0.8, -0.6, 0.4, 0.0, 0.7],
        UMUX_SCORE_COLUMN: [6.4, 2.2, 5.2, 4.0, 6.1],
    })

    sample_evaluation = {
        "final_summary": pd.DataFrame({
            "metric": [
                "total_review",
                "average_vader_compound",
                "average_umux_lite_score_1_7",
            ],
            "value": [
                5,
                0.26,
                4.975,
            ],
        })
    }

    saved_paths = save_pipeline_result(
        df=sample_data,
        evaluation_result=sample_evaluation,
        output_excel_path=OUTPUT_DIR / "sample_hasil_umux_vader.xlsx",
        output_csv_path=OUTPUT_DIR / "sample_hasil_umux_vader.csv",
        summary_csv_path=OUTPUT_DIR / "sample_summary_umux_vader.csv",
    )

    save_text_summary(
        sample_data,
        output_path=OUTPUT_DIR / "sample_summary_report.txt",
    )

    print_text_summary(sample_data)
    print_report_paths(saved_paths)

