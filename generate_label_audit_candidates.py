from pathlib import Path
import pandas as pd

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
# KEYWORD CONFIGURATION
# ============================================================

GENERAL_PRAISE_KEYWORDS = {
    "bagus",
    "baik",
    "mantap",
    "oke",
    "ok",
    "keren",
    "top",
    "good",
    "nice",
    "puas",
    "terbaik",
    "luar biasa",
    "sempurna",
    "sip",
    "jos",
    "mantul",
}


USEFULNESS_KEYWORDS = {
    "bantu",
    "membantu",
    "terbantu",
    "berguna",
    "guna",
    "bermanfaat",
    "manfaat",
    "kebutuhan",
    "butuh",
    "sesuai",
    "klaim",
    "jht",
    "saldo",
    "cair",
    "pencairan",
    "layanan",
    "layan",
    "informasi",
    "fitur",
    "kartu",
    "peserta",
    "data",
    "bpjs",
    "ketenagakerjaan",
}


EASE_OF_USE_KEYWORDS = {
    "mudah",
    "gampang",
    "praktis",
    "simpel",
    "simple",
    "sederhana",
    "jelas",
    "sulit",
    "susah",
    "ribet",
    "rumit",
    "bingung",
    "pakai",
    "gunakan",
    "akses",
    "login",
    "masuk",
    "daftar",
    "registrasi",
    "verifikasi",
    "validasi",
    "buka",
    "error",
    "eror",
    "gagal",
    "lemot",
    "lambat",
    "loading",
    "server",
    "otp",
    "kode",
    "crash",
    "hang",
    "macet",
    "pending",
    "tidak bisa",
    "ga bisa",
    "gak bisa",
    "nggak bisa",
    "gabisa",
}


NEGATIVE_USABILITY_PATTERNS = {
    "tidak bisa login",
    "tidak bisa masuk",
    "tidak bisa buka",
    "tidak bisa dibuka",
    "tidak bisa daftar",
    "tidak bisa verifikasi",
    "gagal login",
    "gagal masuk",
    "gagal verifikasi",
    "otp tidak masuk",
    "kode tidak masuk",
    "server error",
    "loading terus",
    "aplikasi error",
    "aplikasi lemot",
    "aplikasi lambat",
    "aplikasi tidak bisa",
    "force close",
}


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
# TEXT HELPER
# ============================================================

def safe_text(value):
    """
    Mengubah nilai teks menjadi string aman.
    """
    if pd.isna(value):
        return ""

    return str(value).lower().strip()


def contains_any_keyword(text, keywords):
    """
    Mengecek apakah teks mengandung salah satu keyword.
    """
    text = safe_text(text)

    for keyword in keywords:
        if keyword in text:
            return True

    return False


def matched_keywords(text, keywords):
    """
    Mengambil daftar keyword yang cocok dengan teks.
    """
    text = safe_text(text)

    matches = []

    for keyword in sorted(keywords):
        if keyword in text:
            matches.append(keyword)

    return ", ".join(matches)


def count_tokens(text):
    """
    Menghitung jumlah token sederhana.
    """
    text = safe_text(text)

    if not text:
        return 0

    return len(text.split())


# ============================================================
# FEATURE BUILDER
# ============================================================

def add_audit_features(df):
    """
    Menambahkan fitur bantu untuk audit label.
    """
    df = df.copy()

    if TEXT_COLUMN not in df.columns:
        raise ValueError(
            f"Kolom teks '{TEXT_COLUMN}' tidak ditemukan. "
            f"Kolom tersedia: {list(df.columns)}"
        )

    if LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"Kolom label '{LABEL_COLUMN}' tidak ditemukan. "
            f"Kolom tersedia: {list(df.columns)}"
        )

    df[LABEL_COLUMN] = pd.to_numeric(
        df[LABEL_COLUMN],
        errors="coerce"
    ).fillna(0).astype(int)

    df["review_text_lower"] = df[TEXT_COLUMN].apply(safe_text)

    if CLEAN_TEXT_COLUMN in df.columns:
        df["audit_text"] = (
            df[TEXT_COLUMN].apply(safe_text)
            + " "
            + df[CLEAN_TEXT_COLUMN].apply(safe_text)
        )
    else:
        df["audit_text"] = df[TEXT_COLUMN].apply(safe_text)

    df["word_count"] = df["review_text_lower"].apply(count_tokens)

    df["has_general_praise"] = df["audit_text"].apply(
        lambda text: contains_any_keyword(text, GENERAL_PRAISE_KEYWORDS)
    )

    df["has_usefulness_keyword"] = df["audit_text"].apply(
        lambda text: contains_any_keyword(text, USEFULNESS_KEYWORDS)
    )

    df["has_ease_keyword"] = df["audit_text"].apply(
        lambda text: contains_any_keyword(text, EASE_OF_USE_KEYWORDS)
    )

    df["has_negative_usability_pattern"] = df["audit_text"].apply(
        lambda text: contains_any_keyword(text, NEGATIVE_USABILITY_PATTERNS)
    )

    df["matched_general_praise"] = df["audit_text"].apply(
        lambda text: matched_keywords(text, GENERAL_PRAISE_KEYWORDS)
    )

    df["matched_usefulness"] = df["audit_text"].apply(
        lambda text: matched_keywords(text, USEFULNESS_KEYWORDS)
    )

    df["matched_ease"] = df["audit_text"].apply(
        lambda text: matched_keywords(text, EASE_OF_USE_KEYWORDS)
    )

    df["matched_negative_usability"] = df["audit_text"].apply(
        lambda text: matched_keywords(text, NEGATIVE_USABILITY_PATTERNS)
    )

    df["actual_umux_dimension"] = df[LABEL_COLUMN].apply(get_umux_dimension)

    if PREDICTED_LABEL_COLUMN in df.columns:
        df[PREDICTED_LABEL_COLUMN] = pd.to_numeric(
            df[PREDICTED_LABEL_COLUMN],
            errors="coerce"
        ).fillna(-1).astype(int)

        df["is_prediction_correct"] = (
            df[LABEL_COLUMN] == df[PREDICTED_LABEL_COLUMN]
        )

        if PREDICTED_DIMENSION_COLUMN not in df.columns:
            df[PREDICTED_DIMENSION_COLUMN] = df[PREDICTED_LABEL_COLUMN].apply(
                get_umux_dimension
            )

    return df


# ============================================================
# AUDIT RULES
# ============================================================

def assign_audit_reason_and_suggestion(row):
    """
    Memberikan alasan audit dan saran label.

    Fungsi ini tidak mengganti label otomatis.
    Output hanya kandidat untuk dicek manual.
    """
    actual_label = int(row[LABEL_COLUMN])

    word_count = row["word_count"]
    has_general_praise = row["has_general_praise"]
    has_usefulness = row["has_usefulness_keyword"]
    has_ease = row["has_ease_keyword"]
    has_negative_usability = row["has_negative_usability_pattern"]

    # Rule 1:
    # Label 1 tetapi komentar hanya pujian umum.
    if actual_label == 1:
        if has_general_praise and not has_usefulness and not has_ease:
            return (
                "label_1_general_praise_only",
                0,
                "Label aktual Usefulness, tetapi komentar tampak hanya pujian umum. Cek apakah lebih tepat menjadi tidak relevan UMUX-Lite."
            )

        if word_count <= 2 and has_general_praise:
            return (
                "label_1_short_praise",
                0,
                "Label aktual Usefulness, tetapi komentar sangat pendek dan hanya bernada positif umum."
            )

    # Rule 2:
    # Label 0 tetapi mengandung masalah penggunaan.
    if actual_label == 0:
        if has_negative_usability or has_ease:
            if has_usefulness:
                return (
                    "label_0_contains_both_dimensions",
                    3,
                    "Label aktual tidak relevan, tetapi komentar mengandung indikasi usefulness dan ease of use."
                )

            return (
                "label_0_contains_ease_keyword",
                2,
                "Label aktual tidak relevan, tetapi komentar mengandung indikasi ease of use atau masalah penggunaan aplikasi."
            )

    # Rule 3:
    # Label 0 tetapi mengandung manfaat/kebutuhan.
    if actual_label == 0:
        if has_usefulness:
            return (
                "label_0_contains_usefulness_keyword",
                1,
                "Label aktual tidak relevan, tetapi komentar mengandung indikasi usefulness atau manfaat aplikasi."
            )

    # Rule 4:
    # Label 3 tetapi hanya terlihat satu dimensi.
    if actual_label == 3:
        if has_usefulness and not has_ease:
            return (
                "label_3_only_usefulness_detected",
                1,
                "Label aktual gabungan, tetapi keyword yang terdeteksi lebih dominan usefulness saja."
            )

        if has_ease and not has_usefulness:
            return (
                "label_3_only_ease_detected",
                2,
                "Label aktual gabungan, tetapi keyword yang terdeteksi lebih dominan ease of use saja."
            )

        if not has_usefulness and not has_ease:
            return (
                "label_3_no_clear_dimension_keyword",
                0,
                "Label aktual gabungan, tetapi tidak terdeteksi keyword usefulness atau ease of use yang jelas."
            )

    # Rule 5:
    # Label 1 atau 2 tetapi mengandung dua dimensi.
    if actual_label in [1, 2]:
        if has_usefulness and has_ease:
            return (
                "single_dimension_label_contains_both",
                3,
                "Label aktual hanya satu dimensi, tetapi komentar mengandung indikasi usefulness dan ease of use."
            )

    # Rule 6:
    # Komentar terlalu pendek dan tidak mengandung keyword jelas.
    if word_count <= 2:
        if not has_usefulness and not has_ease and not has_general_praise:
            return (
                "too_short_no_clear_keyword",
                0,
                "Komentar sangat pendek dan tidak mengandung keyword UMUX-Lite yang jelas."
            )

    return (
        "",
        actual_label,
        ""
    )


def generate_audit_candidates(df):
    """
    Membuat kandidat data yang perlu diaudit manual.
    """
    df = df.copy()

    audit_results = df.apply(
        assign_audit_reason_and_suggestion,
        axis=1,
        result_type="expand"
    )

    audit_results.columns = [
        "audit_reason",
        "suggested_label",
        "audit_note",
    ]

    df = pd.concat([df, audit_results], axis=1)

    df["suggested_dimension"] = df["suggested_label"].apply(get_umux_dimension)

    candidates = df[df["audit_reason"] != ""].copy()

    candidates["is_suggestion_different"] = (
        candidates[LABEL_COLUMN] != candidates["suggested_label"]
    )

    return candidates


# ============================================================
# SUMMARY
# ============================================================

def create_audit_summary(candidates):
    """
    Membuat ringkasan kandidat audit berdasarkan alasan audit.
    """
    if candidates.empty:
        return pd.DataFrame(
            columns=[
                "audit_reason",
                "total_review",
                "percentage",
                "suggested_label",
                "suggested_dimension",
            ]
        )

    summary = (
        candidates
        .groupby([
            "audit_reason",
            "suggested_label",
            "suggested_dimension",
        ])
        .size()
        .reset_index(name="total_review")
        .sort_values("total_review", ascending=False)
    )

    total_candidates = summary["total_review"].sum()

    summary["percentage"] = (
        summary["total_review"] / total_candidates * 100
    ).round(2)

    return summary


def create_actual_to_suggested_summary(candidates):
    """
    Membuat ringkasan actual label ke suggested label.
    """
    if candidates.empty:
        return pd.DataFrame()

    summary = (
        candidates
        .groupby([
            LABEL_COLUMN,
            "actual_umux_dimension",
            "suggested_label",
            "suggested_dimension",
        ])
        .size()
        .reset_index(name="total_review")
        .sort_values("total_review", ascending=False)
    )

    return summary


def create_prediction_error_audit_summary(candidates):
    """
    Membuat ringkasan kandidat audit yang juga salah prediksi.
    """
    if PREDICTED_LABEL_COLUMN not in candidates.columns:
        return pd.DataFrame()

    if "is_prediction_correct" not in candidates.columns:
        return pd.DataFrame()

    error_candidates = candidates[
        candidates["is_prediction_correct"] == False
    ].copy()

    if error_candidates.empty:
        return pd.DataFrame()

    summary = (
        error_candidates
        .groupby([
            "audit_reason",
            LABEL_COLUMN,
            "actual_umux_dimension",
            PREDICTED_LABEL_COLUMN,
            PREDICTED_DIMENSION_COLUMN,
            "suggested_label",
            "suggested_dimension",
        ])
        .size()
        .reset_index(name="total_review")
        .sort_values("total_review", ascending=False)
    )

    return summary


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


def select_readable_columns(df):
    """
    Memilih dan mengurutkan kolom agar mudah dibaca saat audit.
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
        "is_prediction_correct",
        "suggested_label",
        "suggested_dimension",
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


def save_audit_candidates(
    candidates,
    output_excel_path=OUTPUT_DIR / "label_audit_candidates.xlsx",
    output_csv_path=OUTPUT_DIR / "label_audit_candidates.csv",
):
    """
    Menyimpan kandidat audit label ke Excel dan CSV.
    """
    output_excel_path = Path(output_excel_path)
    output_csv_path = Path(output_csv_path)

    output_excel_path.parent.mkdir(parents=True, exist_ok=True)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    candidates_readable = select_readable_columns(candidates)

    audit_summary = create_audit_summary(candidates)
    actual_to_suggested_summary = create_actual_to_suggested_summary(candidates)
    prediction_error_summary = create_prediction_error_audit_summary(candidates)

    label_1_general_praise = candidates_readable[
        candidates_readable["audit_reason"].isin([
            "label_1_general_praise_only",
            "label_1_short_praise",
        ])
    ].copy()

    label_0_ease = candidates_readable[
        candidates_readable["audit_reason"].isin([
            "label_0_contains_ease_keyword",
        ])
    ].copy()

    label_0_usefulness = candidates_readable[
        candidates_readable["audit_reason"].isin([
            "label_0_contains_usefulness_keyword",
        ])
    ].copy()

    label_0_both = candidates_readable[
        candidates_readable["audit_reason"].isin([
            "label_0_contains_both_dimensions",
        ])
    ].copy()

    label_3_review = candidates_readable[
        candidates_readable[LABEL_COLUMN] == 3
    ].copy()

    single_to_both = candidates_readable[
        candidates_readable["audit_reason"].isin([
            "single_dimension_label_contains_both",
        ])
    ].copy()

    with pd.ExcelWriter(output_excel_path, engine="openpyxl") as writer:
        audit_summary.to_excel(
            writer,
            sheet_name=safe_sheet_name("audit_summary"),
            index=False,
        )

        actual_to_suggested_summary.to_excel(
            writer,
            sheet_name=safe_sheet_name("actual_to_suggested"),
            index=False,
        )

        prediction_error_summary.to_excel(
            writer,
            sheet_name=safe_sheet_name("prediction_error_audit"),
            index=False,
        )

        candidates_readable.to_excel(
            writer,
            sheet_name=safe_sheet_name("all_candidates"),
            index=False,
        )

        label_1_general_praise.to_excel(
            writer,
            sheet_name=safe_sheet_name("label_1_general_praise"),
            index=False,
        )

        label_0_ease.to_excel(
            writer,
            sheet_name=safe_sheet_name("label_0_ease"),
            index=False,
        )

        label_0_usefulness.to_excel(
            writer,
            sheet_name=safe_sheet_name("label_0_usefulness"),
            index=False,
        )

        label_0_both.to_excel(
            writer,
            sheet_name=safe_sheet_name("label_0_both"),
            index=False,
        )

        label_3_review.to_excel(
            writer,
            sheet_name=safe_sheet_name("label_3_review"),
            index=False,
        )

        single_to_both.to_excel(
            writer,
            sheet_name=safe_sheet_name("single_to_both"),
            index=False,
        )

    candidates_readable.to_csv(
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
    print("Memulai pembuatan kandidat audit label...")

    df = load_pipeline_result()

    print(f"Jumlah data awal: {len(df)}")

    df = create_numeric_umux_label_if_missing(df)
    df = add_audit_features(df)

    candidates = generate_audit_candidates(df)

    print("\nRingkasan kandidat audit:")
    print(f"Total data                  : {len(df)}")
    print(f"Total kandidat audit         : {len(candidates)}")

    if len(df) > 0:
        percentage = len(candidates) / len(df) * 100
    else:
        percentage = 0

    print(f"Persentase kandidat audit    : {percentage:.2f}%")

    audit_summary = create_audit_summary(candidates)

    print("\nTop 10 alasan audit:")
    print(audit_summary.head(10))

    saved_paths = save_audit_candidates(candidates)

    print("\nFile berhasil dibuat:")
    for name, path in saved_paths.items():
        print(f"- {name}: {path}")

    print("\nCatatan:")
    print(
        "File ini hanya berisi kandidat audit. "
        "Jangan langsung mengganti label otomatis tanpa pengecekan manual."
    )

    print("\nSelesai.")


if __name__ == "__main__":
    main()