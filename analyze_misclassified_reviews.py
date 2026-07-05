import pandas as pd
from pathlib import Path

from src.config import (
    OUTPUT_DIR,
    OUTPUT_CSV_PATH,
    OUTPUT_EXCEL_PATH,
    TEXT_COLUMN,
    LABEL_COLUMN,
    P1_COLUMN,
    P3_COLUMN,
    CLEAN_TEXT_COLUMN,
    PREDICTED_LABEL_COLUMN,
    PREDICTED_DIMENSION_COLUMN,
    SENTIMENT_CATEGORY_COLUMN,
    VADER_COMPOUND_COLUMN,
    UMUX_SCORE_COLUMN,
    UMUX_LABEL_MAP,
)


# ============================================================
# DATA LOADER
# ============================================================

def load_pipeline_result():
    """
    Membaca hasil pipeline dari folder output.

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
# LABEL HELPER
# ============================================================

def create_numeric_umux_label_if_missing(df):
    """
    Jika kolom label_umux belum ada, buat dari kolom P1 dan P3.

    Mapping:
    P1=0, P3=0 -> 0
    P1=0, P3=1 -> 1
    P1=1, P3=0 -> 2
    P1=1, P3=1 -> 3
    """
    if LABEL_COLUMN in df.columns:
        return df

    required_columns = [P1_COLUMN, P3_COLUMN]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Kolom '{LABEL_COLUMN}' tidak ditemukan, "
            f"dan kolom P1/P3 juga tidak lengkap: {missing_columns}"
        )

    df = df.copy()

    df[P1_COLUMN] = pd.to_numeric(
        df[P1_COLUMN],
        errors="coerce"
    ).fillna(0).astype(int)

    df[P3_COLUMN] = pd.to_numeric(
        df[P3_COLUMN],
        errors="coerce"
    ).fillna(0).astype(int)

    df[P1_COLUMN] = df[P1_COLUMN].apply(lambda value: 1 if value == 1 else 0)
    df[P3_COLUMN] = df[P3_COLUMN].apply(lambda value: 1 if value == 1 else 0)

    def map_label(row):
        p1 = row[P1_COLUMN]
        p3 = row[P3_COLUMN]

        if p1 == 0 and p3 == 0:
            return 0
        if p1 == 0 and p3 == 1:
            return 1
        if p1 == 1 and p3 == 0:
            return 2
        return 3

    df[LABEL_COLUMN] = df.apply(map_label, axis=1)

    return df


def get_umux_dimension(label):
    """
    Mengubah label numerik menjadi nama dimensi UMUX-Lite.
    """
    try:
        label = int(label)
    except (ValueError, TypeError):
        return "Unknown"

    return UMUX_LABEL_MAP.get(label, "Unknown")


# ============================================================
# VALIDATION
# ============================================================

def validate_columns(df):
    """
    Memastikan kolom penting tersedia.
    """
    required_columns = [
        TEXT_COLUMN,
        LABEL_COLUMN,
        PREDICTED_LABEL_COLUMN,
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Kolom berikut tidak ditemukan: {missing_columns}\n"
            f"Kolom yang tersedia: {list(df.columns)}"
        )

    return True


# ============================================================
# ANALYSIS
# ============================================================

def add_analysis_columns(df):
    """
    Menambahkan kolom bantu untuk analisis salah klasifikasi.
    """
    df = df.copy()

    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
    df[PREDICTED_LABEL_COLUMN] = df[PREDICTED_LABEL_COLUMN].astype(int)

    df["actual_umux_dimension"] = df[LABEL_COLUMN].apply(get_umux_dimension)

    if PREDICTED_DIMENSION_COLUMN not in df.columns:
        df[PREDICTED_DIMENSION_COLUMN] = df[PREDICTED_LABEL_COLUMN].apply(
            get_umux_dimension
        )

    df["is_correct"] = df[LABEL_COLUMN] == df[PREDICTED_LABEL_COLUMN]

    df["error_type"] = df.apply(
        lambda row: "correct"
        if row["is_correct"]
        else f"actual_{row[LABEL_COLUMN]}_predicted_{row[PREDICTED_LABEL_COLUMN]}",
        axis=1,
    )

    return df


def create_misclassified_dataframe(df):
    """
    Mengambil data yang salah klasifikasi.
    """
    return df[df["is_correct"] == False].copy()


def create_confusion_summary(df):
    """
    Membuat ringkasan jumlah kesalahan berdasarkan actual dan predicted label.
    """
    summary = (
        df.groupby([
            LABEL_COLUMN,
            "actual_umux_dimension",
            PREDICTED_LABEL_COLUMN,
            PREDICTED_DIMENSION_COLUMN,
        ])
        .size()
        .reset_index(name="total_review")
        .sort_values("total_review", ascending=False)
    )

    return summary


def create_confusion_matrix_table(df):
    """
    Membuat confusion matrix dalam bentuk tabel crosstab.
    """
    matrix = pd.crosstab(
        df[LABEL_COLUMN],
        df[PREDICTED_LABEL_COLUMN],
        rownames=["actual_label"],
        colnames=["predicted_label"],
        dropna=False,
    )

    return matrix.reset_index()


def create_error_type_summary(misclassified_df):
    """
    Membuat ringkasan jenis error.
    """
    summary = (
        misclassified_df["error_type"]
        .value_counts()
        .reset_index()
    )

    summary.columns = ["error_type", "total_review"]

    total_error = summary["total_review"].sum()

    summary["percentage"] = (
        summary["total_review"] / total_error * 100
    ).round(2)

    return summary


def create_label_distribution(df, label_column, label_name):
    """
    Membuat distribusi label aktual atau label prediksi.
    """
    distribution = (
        df[label_column]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    distribution.columns = [label_name, "total_review"]

    distribution["dimension"] = distribution[label_name].apply(get_umux_dimension)

    total_data = distribution["total_review"].sum()

    distribution["percentage"] = (
        distribution["total_review"] / total_data * 100
    ).round(2)

    return distribution


# ============================================================
# EXPORT
# ============================================================

def select_readable_columns(df):
    """
    Mengatur kolom yang ingin ditampilkan di file analisis agar mudah dibaca.
    """
    preferred_columns = [
        TEXT_COLUMN,
        CLEAN_TEXT_COLUMN,
        P1_COLUMN,
        P3_COLUMN,
        LABEL_COLUMN,
        "actual_umux_dimension",
        PREDICTED_LABEL_COLUMN,
        PREDICTED_DIMENSION_COLUMN,
        "is_correct",
        "error_type",
        SENTIMENT_CATEGORY_COLUMN,
        VADER_COMPOUND_COLUMN,
        UMUX_SCORE_COLUMN,
    ]

    available_columns = [
        column for column in preferred_columns
        if column in df.columns
    ]

    other_columns = [
        column for column in df.columns
        if column not in available_columns
    ]

    return df[available_columns + other_columns]


def safe_sheet_name(sheet_name):
    """
    Membatasi nama sheet Excel maksimal 31 karakter.
    """
    invalid_chars = ["\\", "/", "*", "?", ":", "[", "]"]

    for char in invalid_chars:
        sheet_name = sheet_name.replace(char, "_")

    return sheet_name[:31]


def save_analysis_result(
    all_df,
    misclassified_df,
    output_excel_path=OUTPUT_DIR / "misclassified_reviews.xlsx",
    output_csv_path=OUTPUT_DIR / "misclassified_reviews.csv",
):
    """
    Menyimpan hasil analisis salah klasifikasi ke Excel dan CSV.
    """
    output_excel_path = Path(output_excel_path)
    output_csv_path = Path(output_csv_path)

    output_excel_path.parent.mkdir(parents=True, exist_ok=True)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    all_df_readable = select_readable_columns(all_df)
    misclassified_readable = select_readable_columns(misclassified_df)

    confusion_summary = create_confusion_summary(all_df)
    confusion_matrix = create_confusion_matrix_table(all_df)
    error_type_summary = create_error_type_summary(misclassified_df)

    actual_distribution = create_label_distribution(
        all_df,
        LABEL_COLUMN,
        "actual_label",
    )

    predicted_distribution = create_label_distribution(
        all_df,
        PREDICTED_LABEL_COLUMN,
        "predicted_label",
    )

    # Sheet khusus untuk label 3
    actual_3_errors = misclassified_readable[
        misclassified_readable[LABEL_COLUMN] == 3
    ].copy()

    predicted_3_errors = misclassified_readable[
        misclassified_readable[PREDICTED_LABEL_COLUMN] == 3
    ].copy()

    with pd.ExcelWriter(output_excel_path, engine="openpyxl") as writer:
        misclassified_readable.to_excel(
            writer,
            sheet_name=safe_sheet_name("misclassified_reviews"),
            index=False,
        )

        error_type_summary.to_excel(
            writer,
            sheet_name=safe_sheet_name("error_type_summary"),
            index=False,
        )

        confusion_summary.to_excel(
            writer,
            sheet_name=safe_sheet_name("confusion_summary"),
            index=False,
        )

        confusion_matrix.to_excel(
            writer,
            sheet_name=safe_sheet_name("confusion_matrix"),
            index=False,
        )

        actual_distribution.to_excel(
            writer,
            sheet_name=safe_sheet_name("actual_distribution"),
            index=False,
        )

        predicted_distribution.to_excel(
            writer,
            sheet_name=safe_sheet_name("predicted_distribution"),
            index=False,
        )

        actual_3_errors.to_excel(
            writer,
            sheet_name=safe_sheet_name("actual_3_errors"),
            index=False,
        )

        predicted_3_errors.to_excel(
            writer,
            sheet_name=safe_sheet_name("predicted_3_errors"),
            index=False,
        )

        all_df_readable.to_excel(
            writer,
            sheet_name=safe_sheet_name("all_reviews_with_flag"),
            index=False,
        )

    misclassified_readable.to_csv(
        output_csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    return {
        "excel_output": output_excel_path,
        "csv_output": output_csv_path,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("Memulai analisis misclassified reviews...")

    df = load_pipeline_result()

    print(f"Jumlah data awal: {len(df)}")

    df = create_numeric_umux_label_if_missing(df)

    validate_columns(df)

    df = add_analysis_columns(df)

    misclassified_df = create_misclassified_dataframe(df)

    total_data = len(df)
    total_error = len(misclassified_df)
    total_correct = total_data - total_error

    accuracy_check = total_correct / total_data if total_data > 0 else 0

    print("\nRingkasan:")
    print(f"Total data        : {total_data}")
    print(f"Prediksi benar    : {total_correct}")
    print(f"Prediksi salah    : {total_error}")
    print(f"Accuracy check    : {accuracy_check:.4f}")

    print("\nRingkasan jenis error terbesar:")
    error_summary = create_error_type_summary(misclassified_df)
    print(error_summary.head(10))

    saved_paths = save_analysis_result(
        all_df=df,
        misclassified_df=misclassified_df,
    )

    print("\nFile berhasil dibuat:")
    for name, path in saved_paths.items():
        print(f"- {name}: {path}")

    print("\nAnalisis selesai.")


if __name__ == "__main__":
    main()