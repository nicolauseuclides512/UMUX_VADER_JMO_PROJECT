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
from src.umux_labeler import load_umux_labeler, predict_umux_labels
from src.vader_analyzer import analyze_sentiment_dataframe
from src.mapping_score import apply_umux_score_mapping
from src.evaluation import (
    evaluate_final_result,
    evaluate_classification_dataframe,
)
from src.reporting import (
    save_pipeline_result,
    save_text_summary,
    print_text_summary,
    print_report_paths,
)


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
    Membuat label_umux numerik dari kolom P1 dan P3 jika kedua kolom tersedia.

    Dataset menggunakan:
    - mudah digunakan (P1)
    - memenuhi kebutuhan (P3)

    Mapping:
    P1=0, P3=0 -> 0 = Tidak relevan
    P1=0, P3=1 -> 1 = Usefulness
    P1=1, P3=0 -> 2 = Ease of Use
    P1=1, P3=1 -> 3 = Usefulness + Ease of Use

    Jika P1_COLUMN dan P3_COLUMN tidak tersedia, fungsi ini tidak mengubah df.
    Hal ini berguna jika pipeline dipakai untuk data baru yang belum dilabeli manual.
    """
    required_columns = [P1_COLUMN, P3_COLUMN]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        print(
            "\nKolom P1/P3 tidak lengkap. "
            "Pembuatan label_umux manual dilewati."
        )
        print(f"Kolom yang tidak ditemukan: {missing_columns}")
        return df

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

def validate_pipeline_dataset(df):
    """
    Memastikan dataset memiliki kolom teks review.
    Kolom label tidak wajib, karena pipeline bisa dijalankan untuk data baru
    yang belum memiliki label manual.
    """
    if TEXT_COLUMN not in df.columns:
        raise ValueError(
            f"Kolom teks '{TEXT_COLUMN}' tidak ditemukan di dataset.\n"
            f"Kolom yang tersedia: {list(df.columns)}\n\n"
            f"Silakan sesuaikan TEXT_COLUMN di src/config.py."
        )

    df = df.dropna(subset=[TEXT_COLUMN]).copy()

    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).str.strip()
    df = df[df[TEXT_COLUMN] != ""].copy()

    if len(df) == 0:
        raise ValueError(
            "Dataset kosong setelah menghapus data yang tidak memiliki teks review."
        )

    return df


# ============================================================
# OPTIONAL CLASSIFICATION EVALUATION
# ============================================================

def add_optional_classification_evaluation(df, evaluation_result):
    """
    Jika dataset memiliki label manual, maka hasil prediksi model
    dapat dievaluasi dengan accuracy, precision, recall, f1-score,
    dan confusion matrix.

    Jika dataset tidak memiliki label manual, bagian ini akan dilewati.
    """
    if LABEL_COLUMN not in df.columns:
        print(
            f"\nKolom label manual '{LABEL_COLUMN}' tidak ditemukan. "
            "Evaluasi klasifikasi dilewati."
        )
        return evaluation_result

    try:
        classification_result = evaluate_classification_dataframe(df)

        for sheet_name, result_df in classification_result.items():
            evaluation_result[f"classification_{sheet_name}"] = result_df

        print("\nEvaluasi klasifikasi ditambahkan karena dataset memiliki label manual.")

    except Exception as error:
        print("\nEvaluasi klasifikasi tidak dapat dilakukan.")
        print(f"Alasan: {error}")

    return evaluation_result


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    """
    Menjalankan pipeline utama penelitian UMUX-Lite dan VADER.
    """
    print("Memulai pipeline UMUX-Lite dan VADER...")

    create_required_directories()

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------
    print(f"\nMembaca dataset dari: {INPUT_FILE_PATH}")

    df = load_dataset(INPUT_FILE_PATH)

    print(f"Jumlah data awal: {len(df)}")

    # --------------------------------------------------------
    # 2. Create numeric label from P1 and P3, if available
    # --------------------------------------------------------
    print("\nMembuat label_umux dari kolom P1 dan P3 jika tersedia...")
    df = create_numeric_umux_label(df)

    # --------------------------------------------------------
    # 3. Validate dataset
    # --------------------------------------------------------
    df = validate_pipeline_dataset(df)

    print(f"Jumlah data setelah validasi: {len(df)}")

    # --------------------------------------------------------
    # 4. Preprocessing
    # --------------------------------------------------------
    print("\nMelakukan preprocessing...")

    df = preprocess_dataframe(
        df=df,
        text_column=TEXT_COLUMN,
        output_column=CLEAN_TEXT_COLUMN,
    )

    processed_path = PROCESSED_DATA_DIR / "pipeline_data_preprocessed.csv"
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        processed_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Data hasil preprocessing disimpan ke: {processed_path}")

    # --------------------------------------------------------
    # 5. Load model UMUX-Lite labeler
    # --------------------------------------------------------
    print(f"\nMemuat model UMUX-Lite dari: {MODEL_FILE_PATH}")

    model = load_umux_labeler(MODEL_FILE_PATH)

    print("Model berhasil dimuat.")

    # --------------------------------------------------------
    # 6. Predict UMUX-Lite label
    # --------------------------------------------------------
    print("\nMelakukan prediksi label UMUX-Lite...")

    df = predict_umux_labels(
        df=df,
        model=model,
        text_column=CLEAN_TEXT_COLUMN,
    )

    print("Prediksi label UMUX-Lite selesai.")

    # --------------------------------------------------------
    # 7. VADER sentiment analysis
    # --------------------------------------------------------
    print("\nMelakukan analisis sentimen menggunakan VADER...")

    df = analyze_sentiment_dataframe(
        df=df,
        text_column=CLEAN_TEXT_COLUMN,
        use_custom_lexicon=True,
    )

    print("Analisis sentimen VADER selesai.")

    # --------------------------------------------------------
    # 8. Mapping VADER compound score to UMUX-Lite score 1–7
    # --------------------------------------------------------
    print("\nMelakukan mapping VADER compound score ke UMUX-Lite score 1–7...")

    df = apply_umux_score_mapping(df)

    print("Mapping skor selesai.")

    # --------------------------------------------------------
    # 9. Evaluation
    # --------------------------------------------------------
    print("\nMembuat evaluasi hasil akhir...")

    evaluation_result = evaluate_final_result(df)

    evaluation_result = add_optional_classification_evaluation(
        df=df,
        evaluation_result=evaluation_result,
    )

    print("Evaluasi hasil akhir selesai.")

    # --------------------------------------------------------
    # 10. Save output
    # --------------------------------------------------------
    print("\nMenyimpan hasil pipeline...")

    saved_paths = save_pipeline_result(
        df=df,
        evaluation_result=evaluation_result,
    )

    summary_text_path = save_text_summary(
        df=df,
        output_path=OUTPUT_DIR / "summary_report.txt",
    )

    saved_paths["text_summary"] = summary_text_path

    # --------------------------------------------------------
    # 11. Print summary
    # --------------------------------------------------------
    print_text_summary(df)

    print_report_paths(saved_paths)

    print("\nPipeline selesai.")


if __name__ == "__main__":
    main()