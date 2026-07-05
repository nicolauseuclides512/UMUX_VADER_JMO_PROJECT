import pandas as pd

from src.config import (
    VADER_COMPOUND_COLUMN,
    UMUX_SCORE_COLUMN,
    PREDICTED_LABEL_COLUMN,
    PREDICTED_DIMENSION_COLUMN,
)


# ============================================================
# BASIC SCORE MAPPING
# ============================================================

def clamp_score(value, min_value, max_value):
    """
    Membatasi nilai agar tetap berada pada rentang tertentu.

    Contoh:
    jika value < min_value, maka hasilnya min_value
    jika value > max_value, maka hasilnya max_value
    """
    return max(min_value, min(value, max_value))


def vader_compound_to_umux_score(compound_score):
    """
    Mengubah VADER compound score menjadi skor UMUX-Lite skala 1–7.

    VADER compound score memiliki rentang:
    -1 sampai +1

    UMUX-Lite score yang digunakan dalam penelitian ini:
    1 sampai 7

    Rumus:
    UMUX-Lite Score = 1 + ((compound + 1) * 3)

    Contoh:
    compound = -1.00 → score = 1.00
    compound =  0.00 → score = 4.00
    compound =  1.00 → score = 7.00
    """
    if pd.isna(compound_score):
        return None

    try:
        compound_score = float(compound_score)
    except (ValueError, TypeError):
        return None

    # Pastikan compound tetap berada pada rentang -1 sampai +1
    compound_score = clamp_score(compound_score, -1.0, 1.0)

    umux_score = 1 + ((compound_score + 1) * 3)

    # Pastikan skor akhir tetap berada pada rentang 1 sampai 7
    umux_score = clamp_score(umux_score, 1.0, 7.0)

    return round(umux_score, 2)


# ============================================================
# SCORE CATEGORY
# ============================================================

def get_umux_score_category(umux_score):
    """
    Memberikan kategori interpretasi sederhana untuk skor UMUX-Lite 1–7.

    Kategori ini bersifat deskriptif agar hasil lebih mudah dibaca.
    """
    if pd.isna(umux_score):
        return "unknown"

    try:
        umux_score = float(umux_score)
    except (ValueError, TypeError):
        return "unknown"

    if umux_score < 3:
        return "low"

    if umux_score < 5:
        return "moderate"

    return "high"


def is_umux_relevant(label):
    """
    Menentukan apakah review relevan untuk perhitungan UMUX-Lite.

    Label 0 berarti review tidak relevan dengan dimensi UMUX-Lite.
    Label 1, 2, dan 3 berarti review relevan.
    """
    try:
        label = int(label)
    except (ValueError, TypeError):
        return False

    return label in [1, 2, 3]


# ============================================================
# DATAFRAME MAPPING
# ============================================================

def map_vader_to_umux_dataframe(
    df,
    compound_column=VADER_COMPOUND_COLUMN,
    output_score_column=UMUX_SCORE_COLUMN,
    output_category_column="umux_score_category",
):
    """
    Menambahkan kolom skor UMUX-Lite 1–7 berdasarkan VADER compound score.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset review.

    compound_column : str
        Nama kolom yang berisi VADER compound score.

    output_score_column : str
        Nama kolom baru untuk menyimpan skor UMUX-Lite 1–7.

    output_category_column : str
        Nama kolom baru untuk menyimpan kategori skor.

    Returns
    -------
    pandas.DataFrame
        DataFrame dengan tambahan kolom:
        - umux_lite_score_1_7
        - umux_score_category
    """
    if compound_column not in df.columns:
        raise ValueError(
            f"Kolom compound score '{compound_column}' tidak ditemukan. "
            f"Kolom yang tersedia: {list(df.columns)}"
        )

    df = df.copy()

    df[output_score_column] = df[compound_column].apply(
        vader_compound_to_umux_score
    )

    df[output_category_column] = df[output_score_column].apply(
        get_umux_score_category
    )

    return df


# ============================================================
# RELEVANT UMUX DATA
# ============================================================

def add_umux_relevance_column(
    df,
    label_column=PREDICTED_LABEL_COLUMN,
    output_column="is_umux_relevant",
):
    """
    Menambahkan kolom penanda apakah review relevan dengan UMUX-Lite.

    Label 0 = tidak relevan
    Label 1 = usefulness
    Label 2 = ease of use
    Label 3 = usefulness + ease of use
    """
    if label_column not in df.columns:
        raise ValueError(
            f"Kolom label '{label_column}' tidak ditemukan. "
            f"Kolom yang tersedia: {list(df.columns)}"
        )

    df = df.copy()

    df[output_column] = df[label_column].apply(is_umux_relevant)

    return df


def get_relevant_umux_dataframe(
    df,
    relevance_column="is_umux_relevant",
):
    """
    Mengambil hanya review yang relevan dengan UMUX-Lite.

    Fungsi ini berguna saat menghitung rata-rata skor UMUX-Lite.
    Label 0 biasanya tidak dimasukkan dalam perhitungan skor akhir.
    """
    if relevance_column not in df.columns:
        raise ValueError(
            f"Kolom relevance '{relevance_column}' tidak ditemukan. "
            f"Jalankan add_umux_relevance_column() terlebih dahulu."
        )

    return df[df[relevance_column] == True].copy()


# ============================================================
# SUMMARY HELPER
# ============================================================

def calculate_umux_score_summary(
    df,
    dimension_column=PREDICTED_DIMENSION_COLUMN,
    score_column=UMUX_SCORE_COLUMN,
    relevance_column="is_umux_relevant",
):
    """
    Menghitung ringkasan skor UMUX-Lite per dimensi.

    Ringkasan ini hanya menghitung review yang relevan dengan UMUX-Lite,
    yaitu label 1, 2, dan 3.
    """
    required_columns = [dimension_column, score_column]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Kolom '{column}' tidak ditemukan. "
                f"Kolom yang tersedia: {list(df.columns)}"
            )

    df = df.copy()

    if relevance_column in df.columns:
        df = df[df[relevance_column] == True].copy()

    summary = df.groupby(dimension_column).agg(
        total_review=(score_column, "count"),
        average_umux_score=(score_column, "mean"),
        minimum_umux_score=(score_column, "min"),
        maximum_umux_score=(score_column, "max"),
    ).reset_index()

    summary["average_umux_score"] = summary["average_umux_score"].round(2)
    summary["minimum_umux_score"] = summary["minimum_umux_score"].round(2)
    summary["maximum_umux_score"] = summary["maximum_umux_score"].round(2)

    return summary


# ============================================================
# FULL MAPPING PIPELINE
# ============================================================

def apply_umux_score_mapping(df):
    """
    Fungsi ringkas untuk menjalankan proses mapping skor UMUX-Lite.

    Tahapan:
    1. mapping VADER compound score ke skor UMUX-Lite 1–7
    2. menambahkan kategori skor
    3. menambahkan penanda relevansi UMUX-Lite
    """
    df = map_vader_to_umux_dataframe(df)

    if PREDICTED_LABEL_COLUMN in df.columns:
        df = add_umux_relevance_column(df)

    return df


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        VADER_COMPOUND_COLUMN: [-1.0, -0.5, 0.0, 0.5, 1.0],
        PREDICTED_LABEL_COLUMN: [1, 2, 3, 0, 1],
        PREDICTED_DIMENSION_COLUMN: [
            "Usefulness",
            "Ease of Use",
            "Usefulness + Ease of Use",
            "Tidak relevan dengan UMUX-Lite",
            "Usefulness",
        ],
    })

    result = apply_umux_score_mapping(sample_data)
    summary = calculate_umux_score_summary(result)

    print("Hasil mapping:")
    print(result)

    print("\nRingkasan skor:")
    print(summary)
