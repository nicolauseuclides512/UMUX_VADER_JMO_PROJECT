from pathlib import Path
import pandas as pd

from src.config import (
    OUTPUT_DIR,
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
)


# ============================================================
# INPUT AND OUTPUT CONFIGURATION
# ============================================================

INPUT_EXCEL_PATH = OUTPUT_DIR / "label_audit_candidates.xlsx"
INPUT_CSV_PATH = OUTPUT_DIR / "label_audit_candidates.csv"

OUTPUT_EXCEL_PATH = OUTPUT_DIR / "label_audit_priority.xlsx"
OUTPUT_CSV_PATH = OUTPUT_DIR / "label_audit_priority.csv"


# ============================================================
# PRIORITY CONFIGURATION
# ============================================================

LABEL_1_GENERAL_PRAISE_REASONS = {
    "label_1_general_praise_only",
    "label_1_short_praise",
}

LABEL_0_EASE_REASONS = {
    "label_0_contains_ease_keyword",
    "label_0_contains_both_dimensions",
}

LABEL_0_USEFULNESS_REASONS = {
    "label_0_contains_usefulness_keyword",
}

LABEL_3_REVIEW_REASONS = {
    "label_3_only_usefulness_detected",
    "label_3_only_ease_detected",
    "label_3_no_clear_dimension_keyword",
}

SINGLE_TO_BOTH_REASONS = {
    "single_dimension_label_contains_both",
}

LOW_PRIORITY_REASONS = {
    "too_short_no_clear_keyword",
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


def parse_bool(value):
    """
    Mengubah berbagai bentuk nilai menjadi boolean.
    Berguna karena hasil dari Excel/CSV kadang terbaca sebagai string.
    """
    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    if isinstance(value, (int, float)):
        return value == 1

    text = str(value).strip().lower()

    true_values = {"true", "1", "yes", "ya", "y", "benar"}
    false_values = {"false", "0", "no", "tidak", "n", "salah", ""}

    if text in true_values:
        return True

    if text in false_values:
        return False

    return False


def safe_numeric(value, default=None):
    """
    Mengubah nilai menjadi angka jika memungkinkan.
    """
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except (ValueError, TypeError):
        return default


def safe_sheet_name(sheet_name):
    """
    Membatasi nama sheet Excel maksimal 31 karakter.
    """
    invalid_chars = ["\\", "/", "*", "?", ":", "[", "]"]

    for char in invalid_chars:
        sheet_name = sheet_name.replace(char, "_")

    return sheet_name[:31]


# ============================================================
# DATA LOADER
# ============================================================

def load_audit_candidates():
    """
    Membaca hasil dari generate_label_audit_candidates.py.

    Prioritas:
    1. output/label_audit_candidates.xlsx, sheet all_candidates
    2. output/label_audit_candidates.csv
    """
    if INPUT_EXCEL_PATH.exists():
        print(f"Membaca file Excel: {INPUT_EXCEL_PATH}")

        excel_data = pd.read_excel(
            INPUT_EXCEL_PATH,
            sheet_name=None,
        )

        if "all_candidates" in excel_data:
            return excel_data["all_candidates"]

        # Fallback jika nama sheet berubah
        first_sheet_name = list(excel_data.keys())[0]
        print(
            f"Sheet 'all_candidates' tidak ditemukan. "
            f"Menggunakan sheet pertama: {first_sheet_name}"
        )

        return excel_data[first_sheet_name]

    if INPUT_CSV_PATH.exists():
        print(f"Membaca file CSV: {INPUT_CSV_PATH}")
        return pd.read_csv(INPUT_CSV_PATH)

    raise FileNotFoundError(
        "File kandidat audit tidak ditemukan.\n"
        "Jalankan terlebih dahulu:\n"
        "python generate_label_audit_candidates.py"
    )


# ============================================================
# VALIDATION AND PREPARATION
# ============================================================

def validate_and_prepare_columns(df):
    """
    Memastikan kolom penting tersedia dan menambahkan kolom bantu jika belum ada.
    """
    df = df.copy()

    required_columns = [
        TEXT_COLUMN,
        LABEL_COLUMN,
        "audit_reason",
        "suggested_label",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Kolom berikut tidak ditemukan: {missing_columns}\n"
            f"Kolom tersedia: {list(df.columns)}"
        )

    df[LABEL_COLUMN] = pd.to_numeric(
        df[LABEL_COLUMN],
        errors="coerce"
    ).fillna(0).astype(int)

    df["suggested_label"] = pd.to_numeric(
        df["suggested_label"],
        errors="coerce"
    ).fillna(df[LABEL_COLUMN]).astype(int)

    if "is_suggestion_different" not in df.columns:
        df["is_suggestion_different"] = (
            df[LABEL_COLUMN] != df["suggested_label"]
        )

    if "is_prediction_correct" not in df.columns:
        if PREDICTED_LABEL_COLUMN in df.columns:
            df[PREDICTED_LABEL_COLUMN] = pd.to_numeric(
                df[PREDICTED_LABEL_COLUMN],
                errors="coerce"
            ).fillna(-1).astype(int)

            df["is_prediction_correct"] = (
                df[LABEL_COLUMN] == df[PREDICTED_LABEL_COLUMN]
            )
        else:
            df["is_prediction_correct"] = False

    df["is_suggestion_different"] = df["is_suggestion_different"].apply(parse_bool)
    df["is_prediction_correct"] = df["is_prediction_correct"].apply(parse_bool)

    if "audit_note" not in df.columns:
        df["audit_note"] = ""

    if "suggested_dimension" not in df.columns:
        df["suggested_dimension"] = ""

    if "actual_umux_dimension" not in df.columns:
        df["actual_umux_dimension"] = ""

    return df


# ============================================================
# PRIORITY RULES
# ============================================================

def assign_priority(row):
    """
    Memberikan prioritas audit.

    Prioritas 1 adalah yang paling penting untuk dicek terlebih dahulu.
    """
    audit_reason = safe_text(row.get("audit_reason", ""))

    is_suggestion_different = parse_bool(
        row.get("is_suggestion_different", False)
    )

    is_prediction_correct = parse_bool(
        row.get("is_prediction_correct", True)
    )

    # PRIORITY 1:
    # Label dicurigai perlu berubah dan prediksi model juga salah.
    if is_suggestion_different and not is_prediction_correct:
        return (
            1,
            "P1 - Different suggestion and wrong prediction",
            "Prioritas tertinggi: saran label berbeda dari label aktual dan prediksi model juga salah."
        )

    # PRIORITY 2:
    # Label 1 tetapi komentar hanya pujian umum.
    if audit_reason in LABEL_1_GENERAL_PRAISE_REASONS:
        return (
            2,
            "P2 - Label 1 general praise",
            "Label Usefulness perlu dicek karena komentar tampak hanya pujian umum."
        )

    # PRIORITY 3:
    # Label 0 tetapi ada indikasi ease of use.
    if audit_reason in LABEL_0_EASE_REASONS:
        return (
            3,
            "P3 - Label 0 contains ease keyword",
            "Label tidak relevan perlu dicek karena komentar mengandung indikasi ease of use atau masalah penggunaan."
        )

    # PRIORITY 4:
    # Label 0 tetapi ada indikasi usefulness.
    if audit_reason in LABEL_0_USEFULNESS_REASONS:
        return (
            4,
            "P4 - Label 0 contains usefulness keyword",
            "Label tidak relevan perlu dicek karena komentar mengandung indikasi usefulness atau manfaat aplikasi."
        )

    # PRIORITY 5:
    # Label 3 perlu dicek karena mungkin hanya satu dimensi.
    if audit_reason in LABEL_3_REVIEW_REASONS:
        return (
            5,
            "P5 - Label 3 dimension check",
            "Label gabungan perlu dicek apakah benar mengandung usefulness dan ease of use sekaligus."
        )

    # PRIORITY 6:
    # Label satu dimensi tetapi keyword mendeteksi dua dimensi.
    if audit_reason in SINGLE_TO_BOTH_REASONS:
        return (
            6,
            "P6 - Single dimension may contain both",
            "Label satu dimensi perlu dicek apakah sebenarnya mengandung dua dimensi."
        )

    # PRIORITY 7:
    # Komentar sangat pendek dan tidak jelas.
    if audit_reason in LOW_PRIORITY_REASONS:
        return (
            7,
            "P7 - Too short or unclear",
            "Prioritas rendah: komentar terlalu pendek atau tidak memiliki keyword jelas."
        )

    return (
        8,
        "P8 - Other audit candidates",
        "Kandidat audit lain."
    )


def add_priority_columns(df):
    """
    Menambahkan kolom prioritas audit.
    """
    df = df.copy()

    priority_results = df.apply(
        assign_priority,
        axis=1,
        result_type="expand"
    )

    priority_results.columns = [
        "priority_rank",
        "priority_name",
        "priority_note",
    ]

    df = pd.concat([df, priority_results], axis=1)

    df["label_change_direction"] = (
        df[LABEL_COLUMN].astype(str)
        + " -> "
        + df["suggested_label"].astype(str)
    )

    return df


def add_manual_audit_columns(df):
    """
    Menambahkan kolom kosong untuk diisi manual oleh peneliti di Excel.
    """
    df = df.copy()

    manual_columns = {
        "manual_check_status": "",
        "manual_final_label": "",
        "manual_final_dimension": "",
        "manual_decision": "",
        "manual_note": "",
    }

    for column, default_value in manual_columns.items():
        if column not in df.columns:
            df[column] = default_value

    return df


def add_audit_id(df):
    """
    Menambahkan audit_id agar data mudah dilacak.
    """
    df = df.copy()

    if "audit_id" not in df.columns:
        df.insert(0, "audit_id", range(1, len(df) + 1))

    return df


# ============================================================
# SUMMARY
# ============================================================

def create_priority_summary(df):
    """
    Membuat ringkasan jumlah data per prioritas.
    """
    summary = (
        df.groupby([
            "priority_rank",
            "priority_name",
        ])
        .size()
        .reset_index(name="total_review")
        .sort_values("priority_rank")
    )

    total_data = summary["total_review"].sum()

    summary["percentage"] = (
        summary["total_review"] / total_data * 100
    ).round(2)

    return summary


def create_priority_reason_summary(df):
    """
    Membuat ringkasan prioritas berdasarkan audit_reason.
    """
    summary = (
        df.groupby([
            "priority_rank",
            "priority_name",
            "audit_reason",
            LABEL_COLUMN,
            "suggested_label",
            "label_change_direction",
        ])
        .size()
        .reset_index(name="total_review")
        .sort_values(
            ["priority_rank", "total_review"],
            ascending=[True, False]
        )
    )

    return summary


def create_label_change_summary(df):
    """
    Membuat ringkasan arah perubahan label yang disarankan.
    """
    summary = (
        df.groupby([
            "label_change_direction",
            LABEL_COLUMN,
            "suggested_label",
        ])
        .size()
        .reset_index(name="total_review")
        .sort_values("total_review", ascending=False)
    )

    total_data = summary["total_review"].sum()

    summary["percentage"] = (
        summary["total_review"] / total_data * 100
    ).round(2)

    return summary


def create_prediction_priority_summary(df):
    """
    Membuat ringkasan berdasarkan apakah prediksi model benar/salah.
    """
    summary = (
        df.groupby([
            "priority_rank",
            "priority_name",
            "is_prediction_correct",
        ])
        .size()
        .reset_index(name="total_review")
        .sort_values(
            ["priority_rank", "is_prediction_correct"],
            ascending=[True, True]
        )
    )

    return summary


# ============================================================
# COLUMN SELECTION
# ============================================================

def select_readable_columns(df):
    """
    Mengatur urutan kolom agar nyaman untuk audit manual.
    """
    preferred_columns = [
        "audit_id",
        "priority_rank",
        "priority_name",
        "priority_note",

        "manual_check_status",
        "manual_final_label",
        "manual_final_dimension",
        "manual_decision",
        "manual_note",

        TEXT_COLUMN,
        CLEAN_TEXT_COLUMN,

        P1_COLUMN,
        P3_COLUMN,
        LABEL_COLUMN,
        "actual_umux_dimension",

        "suggested_label",
        "suggested_dimension",
        "label_change_direction",

        PREDICTED_LABEL_COLUMN,
        PREDICTED_DIMENSION_COLUMN,
        "is_prediction_correct",

        "audit_reason",
        "audit_note",

        "word_count",
        "matched_general_praise",
        "matched_usefulness",
        "matched_ease",
        "matched_negative_usability",

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


# ============================================================
# FILTER SHEETS
# ============================================================

def filter_priority_1(df):
    return df[df["priority_rank"] == 1].copy()


def filter_priority_2(df):
    return df[df["priority_rank"] == 2].copy()


def filter_priority_3(df):
    return df[df["priority_rank"] == 3].copy()


def filter_priority_4(df):
    return df[df["priority_rank"] == 4].copy()


def filter_priority_5(df):
    return df[df["priority_rank"] == 5].copy()


def filter_high_priority(df):
    """
    Prioritas tinggi yang disarankan dicek lebih dahulu.
    """
    return df[df["priority_rank"].isin([1, 2, 3, 4, 5])].copy()


def filter_label_1_praise_all(df):
    """
    Semua kandidat label 1 yang berupa pujian umum,
    termasuk yang sudah masuk prioritas 1.
    """
    return df[
        df["audit_reason"].isin(LABEL_1_GENERAL_PRAISE_REASONS)
    ].copy()


def filter_label_0_ease_all(df):
    """
    Semua kandidat label 0 yang mengandung ease keyword,
    termasuk yang sudah masuk prioritas 1.
    """
    return df[
        df["audit_reason"].isin(LABEL_0_EASE_REASONS)
    ].copy()


def filter_label_0_usefulness_all(df):
    """
    Semua kandidat label 0 yang mengandung usefulness keyword,
    termasuk yang sudah masuk prioritas 1.
    """
    return df[
        df["audit_reason"].isin(LABEL_0_USEFULNESS_REASONS)
    ].copy()


def filter_label_3_all(df):
    """
    Semua kandidat label 3 yang perlu dicek.
    """
    return df[
        df["audit_reason"].isin(LABEL_3_REVIEW_REASONS)
    ].copy()


# ============================================================
# EXPORT
# ============================================================

def save_priority_audit(df):
    """
    Menyimpan hasil filter prioritas ke Excel dan CSV.
    """
    OUTPUT_EXCEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    df_readable = select_readable_columns(df)

    priority_summary = create_priority_summary(df)
    priority_reason_summary = create_priority_reason_summary(df)
    label_change_summary = create_label_change_summary(df)
    prediction_priority_summary = create_prediction_priority_summary(df)

    priority_1 = select_readable_columns(filter_priority_1(df))
    priority_2 = select_readable_columns(filter_priority_2(df))
    priority_3 = select_readable_columns(filter_priority_3(df))
    priority_4 = select_readable_columns(filter_priority_4(df))
    priority_5 = select_readable_columns(filter_priority_5(df))

    high_priority = select_readable_columns(filter_high_priority(df))

    label_1_praise_all = select_readable_columns(filter_label_1_praise_all(df))
    label_0_ease_all = select_readable_columns(filter_label_0_ease_all(df))
    label_0_usefulness_all = select_readable_columns(filter_label_0_usefulness_all(df))
    label_3_all = select_readable_columns(filter_label_3_all(df))

    with pd.ExcelWriter(OUTPUT_EXCEL_PATH, engine="openpyxl") as writer:
        priority_summary.to_excel(
            writer,
            sheet_name=safe_sheet_name("priority_summary"),
            index=False,
        )

        priority_reason_summary.to_excel(
            writer,
            sheet_name=safe_sheet_name("priority_reason_summary"),
            index=False,
        )

        label_change_summary.to_excel(
            writer,
            sheet_name=safe_sheet_name("label_change_summary"),
            index=False,
        )

        prediction_priority_summary.to_excel(
            writer,
            sheet_name=safe_sheet_name("prediction_summary"),
            index=False,
        )

        high_priority.to_excel(
            writer,
            sheet_name=safe_sheet_name("high_priority"),
            index=False,
        )

        priority_1.to_excel(
            writer,
            sheet_name=safe_sheet_name("priority_1"),
            index=False,
        )

        priority_2.to_excel(
            writer,
            sheet_name=safe_sheet_name("priority_2"),
            index=False,
        )

        priority_3.to_excel(
            writer,
            sheet_name=safe_sheet_name("priority_3"),
            index=False,
        )

        priority_4.to_excel(
            writer,
            sheet_name=safe_sheet_name("priority_4"),
            index=False,
        )

        priority_5.to_excel(
            writer,
            sheet_name=safe_sheet_name("priority_5"),
            index=False,
        )

        label_1_praise_all.to_excel(
            writer,
            sheet_name=safe_sheet_name("all_label_1_praise"),
            index=False,
        )

        label_0_ease_all.to_excel(
            writer,
            sheet_name=safe_sheet_name("all_label_0_ease"),
            index=False,
        )

        label_0_usefulness_all.to_excel(
            writer,
            sheet_name=safe_sheet_name("all_label_0_usefulness"),
            index=False,
        )

        label_3_all.to_excel(
            writer,
            sheet_name=safe_sheet_name("all_label_3_review"),
            index=False,
        )

        df_readable.to_excel(
            writer,
            sheet_name=safe_sheet_name("all_priority_candidates"),
            index=False,
        )

    high_priority.to_csv(
        OUTPUT_CSV_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    return {
        "excel_output": OUTPUT_EXCEL_PATH,
        "csv_output": OUTPUT_CSV_PATH,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("Memulai filter kandidat audit prioritas...")

    df = load_audit_candidates()

    print(f"Jumlah kandidat audit awal: {len(df)}")

    df = validate_and_prepare_columns(df)
    df = add_priority_columns(df)
    df = add_manual_audit_columns(df)
    df = add_audit_id(df)

    priority_summary = create_priority_summary(df)

    print("\nRingkasan prioritas audit:")
    print(priority_summary)

    high_priority = filter_high_priority(df)

    print("\nRingkasan:")
    print(f"Total kandidat audit          : {len(df)}")
    print(f"Total high priority P1-P5     : {len(high_priority)}")

    if len(df) > 0:
        high_priority_percentage = len(high_priority) / len(df) * 100
    else:
        high_priority_percentage = 0

    print(f"Persentase high priority      : {high_priority_percentage:.2f}%")

    saved_paths = save_priority_audit(df)

    print("\nFile berhasil dibuat:")
    for name, path in saved_paths.items():
        print(f"- {name}: {path}")

    print("\nSaran penggunaan:")
    print("1. Buka output/label_audit_priority.xlsx")
    print("2. Mulai dari sheet 'priority_1'")
    print("3. Lanjutkan ke 'priority_2', 'priority_3', 'priority_4', dan 'priority_5'")
    print("4. Isi kolom manual_final_label hanya jika Bapak yakin label perlu diperbaiki")
    print("5. Jangan mengganti label otomatis tanpa pengecekan manual")

    print("\nSelesai.")


if __name__ == "__main__":
    main()