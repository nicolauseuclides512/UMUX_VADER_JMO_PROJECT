import pandas as pd
from pathlib import Path

from src.config import (
    INPUT_FILE_PATH,
    PROCESSED_DATA_DIR,
    OUTPUT_DIR,
    TEXT_COLUMN,
    LABEL_COLUMN,
    P1_COLUMN,
    P3_COLUMN,
    CLEAN_TEXT_COLUMN,
    MODEL_FILE_PATH,
    create_required_directories,
)

from src.preprocessing import preprocess_dataframe
from src.umux_labeler import train_evaluate_and_save
from src.reporting import save_training_evaluation_report


# ============================================================
# DATA LOADER
# ============================================================

def load_dataset(input_path=INPUT_FILE_PATH):
    """
    Membaca dataset dari file Excel atau CSV.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"File dataset tidak ditemukan: {input_path}\n"
            f"Pastikan dataset sudah diletakkan di folder data/raw/."
        )

    if input_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(input_path)
    elif input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError(
            "Format file tidak didukung. Gunakan file .xlsx, .xls, atau .csv."
        )

    return df


# ============================================================
# UMUX-LITE LABEL CREATOR
# ============================================================

def create_numeric_umux_label(df):
    """
    Membuat label_umux numerik dari kolom P1 dan P3.

    Dataset menggunakan:
    - mudah digunakan (P1)
    - memenuhi kebutuhan (P3)

    Mapping:
    P1=0, P3=0 -> 0 = Tidak relevan
    P1=0, P3=1 -> 1 = Usefulness
    P1=1, P3=0 -> 2 = Ease of Use
    P1=1, P3=1 -> 3 = Usefulness + Ease of Use
    """
    required_columns = [P1_COLUMN, P3_COLUMN]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Kolom P1/P3 berikut tidak ditemukan: {missing_columns}\n"
            f"Kolom yang tersedia: {list(df.columns)}\n\n"
            f"Silakan cek kembali nama P1_COLUMN dan P3_COLUMN di src/config.py."
        )

    df = df.copy()

    # Ubah nilai P1 dan P3 menjadi numerik.
    # Nilai kosong/error akan dianggap 0.
    df[P1_COLUMN] = pd.to_numeric(
        df[P1_COLUMN],
        errors="coerce"
    ).fillna(0).astype(int)

    df[P3_COLUMN] = pd.to_numeric(
        df[P3_COLUMN],
        errors="coerce"
    ).fillna(0).astype(int)

    # Pastikan hanya nilai 0 dan 1 yang digunakan.
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


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_training_dataset(df):
    """
    Memastikan dataset memiliki kolom teks review dan kolom label UMUX-Lite.
    """
    required_columns = [TEXT_COLUMN, LABEL_COLUMN]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Kolom berikut tidak ditemukan di dataset: {missing_columns}\n"
            f"Kolom yang tersedia: {list(df.columns)}\n\n"
            f"Silakan sesuaikan TEXT_COLUMN dan LABEL_COLUMN di src/config.py."
        )

    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN]).copy()

    # Bersihkan review kosong.
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).str.strip()
    df = df[df[TEXT_COLUMN] != ""].copy()

    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

    valid_labels = {0, 1, 2, 3}
    dataset_labels = set(df[LABEL_COLUMN].unique())

    invalid_labels = dataset_labels.difference(valid_labels)

    if invalid_labels:
        raise ValueError(
            f"Ditemukan label tidak valid: {invalid_labels}\n"
            f"Label yang diperbolehkan hanya 0, 1, 2, dan 3."
        )

    label_counts = df[LABEL_COLUMN].value_counts().sort_index()

    print("\nDistribusi label pada dataset:")
    print(label_counts)

    if label_counts.min() < 2:
        raise ValueError(
            "Setiap label minimal perlu memiliki 2 data agar train-test split "
            "dengan stratify dapat berjalan dengan baik.\n"
            f"Distribusi label saat ini:\n{label_counts}"
        )

    return df


# ============================================================
# REPORT CONVERTER
# ============================================================

def prepare_training_report(evaluation_result):
    """
    Mengubah hasil evaluasi model menjadi beberapa DataFrame
    agar dapat disimpan ke Excel.
    """
    classification_summary = pd.DataFrame({
        "metric": ["accuracy"],
        "value": [round(evaluation_result["accuracy"], 4)],
    })

    classification_report_df = pd.DataFrame(
        evaluation_result["classification_report_dict"]
    ).transpose()

    confusion_matrix_df = evaluation_result["confusion_matrix"]

    training_report = {
        "classification_summary": classification_summary,
        "classification_report": classification_report_df,
        "confusion_matrix": confusion_matrix_df,
    }

    return training_report


def save_classification_report_text(evaluation_result):
    """
    Menyimpan classification report dalam bentuk teks.
    """
    output_path = OUTPUT_DIR / "training_classification_report.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("Classification Report UMUX-Lite Labeler\n")
        file.write("======================================\n\n")
        file.write(f"Accuracy: {evaluation_result['accuracy']:.4f}\n\n")
        file.write(evaluation_result["classification_report_text"])

    return output_path


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():
    """
    Menjalankan proses training model klasifikasi UMUX-Lite.
    """
    print("Memulai training model UMUX-Lite labeler...")

    create_required_directories()

    print(f"\nMembaca dataset dari: {INPUT_FILE_PATH}")
    df = load_dataset(INPUT_FILE_PATH)

    print(f"Jumlah data awal: {len(df)}")

    print("\nMembuat label_umux dari kolom P1 dan P3...")
    df = create_numeric_umux_label(df)

    df = validate_training_dataset(df)

    print(f"Jumlah data setelah validasi: {len(df)}")

    print("\nMelakukan preprocessing...")
    df = preprocess_dataframe(
        df=df,
        text_column=TEXT_COLUMN,
        output_column=CLEAN_TEXT_COLUMN,
    )

    processed_path = PROCESSED_DATA_DIR / "training_data_preprocessed.csv"
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        processed_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Data hasil preprocessing disimpan ke: {processed_path}")

    print("\nMelatih dan mengevaluasi model...")
    model, evaluation_result = train_evaluate_and_save(
        df=df,
        text_column=CLEAN_TEXT_COLUMN,
        label_column=LABEL_COLUMN,
        model_path=MODEL_FILE_PATH,
    )

    print(f"\nModel berhasil disimpan ke: {MODEL_FILE_PATH}")

    print("\nHasil evaluasi model:")
    print(f"Accuracy: {evaluation_result['accuracy']:.4f}")
    print("\nClassification Report:")
    print(evaluation_result["classification_report_text"])

    training_report = prepare_training_report(evaluation_result)

    training_report_path = save_training_evaluation_report(
        evaluation_result=training_report,
        output_path=OUTPUT_DIR / "training_evaluation.xlsx",
    )

    text_report_path = save_classification_report_text(evaluation_result)

    print("\nLaporan evaluasi training berhasil disimpan:")
    print(f"- Excel report : {training_report_path}")
    print(f"- Text report  : {text_report_path}")

    print("\nTraining selesai.")


if __name__ == "__main__":
    main()