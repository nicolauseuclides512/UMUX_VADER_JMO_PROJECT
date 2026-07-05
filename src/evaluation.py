import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from src.config import (
    LABEL_COLUMN,
    PREDICTED_LABEL_COLUMN,
    PREDICTED_DIMENSION_COLUMN,
    SENTIMENT_CATEGORY_COLUMN,
    VADER_COMPOUND_COLUMN,
    UMUX_SCORE_COLUMN,
    UMUX_LABEL_MAP,
)


# ============================================================
# BASIC VALIDATION
# ============================================================

def validate_columns(df, required_columns):
    """
    Memastikan semua kolom yang dibutuhkan tersedia di DataFrame.
    """
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Kolom berikut tidak ditemukan: {missing_columns}. "
            f"Kolom yang tersedia: {list(df.columns)}"
        )

    return True


# ============================================================
# CLASSIFICATION EVALUATION
# ============================================================

def evaluate_classification_result(y_true, y_pred):
    """
    Mengevaluasi hasil klasifikasi UMUX-Lite.

    Output:
    - accuracy
    - precision
    - recall
    - f1-score
    - classification report
    - confusion matrix
    """
    labels = list(UMUX_LABEL_MAP.keys())
    target_names = list(UMUX_LABEL_MAP.values())

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report_dict).transpose()

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    confusion_matrix_df = pd.DataFrame(
        cm,
        index=[f"actual_{label}" for label in labels],
        columns=[f"predicted_{label}" for label in labels],
    )

    summary_df = pd.DataFrame({
        "metric": [
            "accuracy",
            "weighted_precision",
            "weighted_recall",
            "weighted_f1_score",
        ],
        "value": [
            accuracy,
            precision,
            recall,
            f1_score,
        ],
    })

    summary_df["value"] = summary_df["value"].round(4)

    return {
        "classification_summary": summary_df,
        "classification_report": report_df,
        "confusion_matrix": confusion_matrix_df,
    }


def evaluate_classification_dataframe(
    df,
    actual_label_column=LABEL_COLUMN,
    predicted_label_column=PREDICTED_LABEL_COLUMN,
):
    """
    Mengevaluasi klasifikasi UMUX-Lite berdasarkan kolom label aktual
    dan kolom label prediksi pada DataFrame.

    Fungsi ini digunakan jika dataset memiliki label manual.
    """
    validate_columns(
        df,
        [actual_label_column, predicted_label_column]
    )

    df = df.dropna(
        subset=[actual_label_column, predicted_label_column]
    ).copy()

    y_true = df[actual_label_column].astype(int)
    y_pred = df[predicted_label_column].astype(int)

    return evaluate_classification_result(y_true, y_pred)


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

def calculate_label_distribution(
    df,
    label_column=PREDICTED_LABEL_COLUMN,
):
    """
    Menghitung distribusi label UMUX-Lite.

    Output:
    - label
    - dimension
    - total_review
    - percentage
    """
    validate_columns(df, [label_column])

    distribution = (
        df[label_column]
        .value_counts(dropna=False)
        .sort_index()
        .reset_index()
    )

    distribution.columns = ["label", "total_review"]

    total_data = distribution["total_review"].sum()

    distribution["percentage"] = (
        distribution["total_review"] / total_data * 100
    ).round(2)

    distribution["dimension"] = distribution["label"].apply(
        lambda label: UMUX_LABEL_MAP.get(int(label), "Unknown")
        if pd.notna(label) else "Unknown"
    )

    distribution = distribution[
        ["label", "dimension", "total_review", "percentage"]
    ]

    return distribution


def calculate_dimension_distribution(
    df,
    dimension_column=PREDICTED_DIMENSION_COLUMN,
):
    """
    Menghitung distribusi berdasarkan nama dimensi UMUX-Lite.
    """
    validate_columns(df, [dimension_column])

    distribution = (
        df[dimension_column]
        .value_counts(dropna=False)
        .reset_index()
    )

    distribution.columns = ["dimension", "total_review"]

    total_data = distribution["total_review"].sum()

    distribution["percentage"] = (
        distribution["total_review"] / total_data * 100
    ).round(2)

    return distribution


# ============================================================
# SENTIMENT EVALUATION
# ============================================================

def calculate_sentiment_distribution(
    df,
    sentiment_column=SENTIMENT_CATEGORY_COLUMN,
):
    """
    Menghitung distribusi kategori sentimen VADER.
    """
    validate_columns(df, [sentiment_column])

    distribution = (
        df[sentiment_column]
        .value_counts(dropna=False)
        .reset_index()
    )

    distribution.columns = ["sentiment_category", "total_review"]

    total_data = distribution["total_review"].sum()

    distribution["percentage"] = (
        distribution["total_review"] / total_data * 100
    ).round(2)

    return distribution


def calculate_vader_compound_statistics(
    df,
    compound_column=VADER_COMPOUND_COLUMN,
):
    """
    Menghitung statistik VADER compound score.
    """
    validate_columns(df, [compound_column])

    statistics = df[compound_column].describe().reset_index()
    statistics.columns = ["statistic", "value"]

    statistics["value"] = statistics["value"].round(4)

    return statistics


# ============================================================
# UMUX SCORE EVALUATION
# ============================================================

def calculate_umux_score_statistics(
    df,
    score_column=UMUX_SCORE_COLUMN,
):
    """
    Menghitung statistik skor UMUX-Lite 1–7.
    """
    validate_columns(df, [score_column])

    statistics = df[score_column].describe().reset_index()
    statistics.columns = ["statistic", "value"]

    statistics["value"] = statistics["value"].round(4)

    return statistics


def calculate_umux_score_by_dimension(
    df,
    dimension_column=PREDICTED_DIMENSION_COLUMN,
    score_column=UMUX_SCORE_COLUMN,
):
    """
    Menghitung rata-rata skor UMUX-Lite berdasarkan dimensi UMUX-Lite.
    """
    validate_columns(df, [dimension_column, score_column])

    summary = df.groupby(dimension_column).agg(
        total_review=(score_column, "count"),
        average_score=(score_column, "mean"),
        minimum_score=(score_column, "min"),
        maximum_score=(score_column, "max"),
        standard_deviation=(score_column, "std"),
    ).reset_index()

    numeric_columns = [
        "average_score",
        "minimum_score",
        "maximum_score",
        "standard_deviation",
    ]

    for column in numeric_columns:
        summary[column] = summary[column].round(2)

    return summary


def calculate_umux_score_by_sentiment(
    df,
    sentiment_column=SENTIMENT_CATEGORY_COLUMN,
    score_column=UMUX_SCORE_COLUMN,
):
    """
    Menghitung rata-rata skor UMUX-Lite berdasarkan kategori sentimen.
    """
    validate_columns(df, [sentiment_column, score_column])

    summary = df.groupby(sentiment_column).agg(
        total_review=(score_column, "count"),
        average_score=(score_column, "mean"),
        minimum_score=(score_column, "min"),
        maximum_score=(score_column, "max"),
        standard_deviation=(score_column, "std"),
    ).reset_index()

    numeric_columns = [
        "average_score",
        "minimum_score",
        "maximum_score",
        "standard_deviation",
    ]

    for column in numeric_columns:
        summary[column] = summary[column].round(2)

    return summary


# ============================================================
# FINAL EVALUATION SUMMARY
# ============================================================

def create_final_evaluation_summary(df):
    """
    Membuat ringkasan evaluasi akhir penelitian.

    Fungsi ini digunakan setelah pipeline utama selesai:
    preprocessing → klasifikasi UMUX-Lite → VADER → mapping skor.
    """
    required_columns = [
        PREDICTED_LABEL_COLUMN,
        PREDICTED_DIMENSION_COLUMN,
        SENTIMENT_CATEGORY_COLUMN,
        VADER_COMPOUND_COLUMN,
        UMUX_SCORE_COLUMN,
    ]

    validate_columns(df, required_columns)

    total_review = len(df)

    relevant_df = df[
        df[PREDICTED_LABEL_COLUMN].astype(int).isin([1, 2, 3])
    ].copy()

    total_relevant_review = len(relevant_df)

    total_irrelevant_review = total_review - total_relevant_review

    average_compound = df[VADER_COMPOUND_COLUMN].mean()
    average_umux_score = relevant_df[UMUX_SCORE_COLUMN].mean()

    summary = pd.DataFrame({
        "metric": [
            "total_review",
            "total_relevant_umux_review",
            "total_irrelevant_review",
            "average_vader_compound",
            "average_umux_lite_score_1_7",
        ],
        "value": [
            total_review,
            total_relevant_review,
            total_irrelevant_review,
            round(average_compound, 4),
            round(average_umux_score, 4),
        ],
    })

    return summary


# ============================================================
# FULL EVALUATION PIPELINE
# ============================================================

def evaluate_final_result(df):
    """
    Menjalankan seluruh evaluasi hasil akhir.

    Output berupa dictionary berisi beberapa DataFrame evaluasi.
    """
    evaluation_result = {}

    evaluation_result["final_summary"] = create_final_evaluation_summary(df)

    evaluation_result["label_distribution"] = calculate_label_distribution(df)

    evaluation_result["dimension_distribution"] = calculate_dimension_distribution(df)

    evaluation_result["sentiment_distribution"] = calculate_sentiment_distribution(df)

    evaluation_result["vader_compound_statistics"] = calculate_vader_compound_statistics(df)

    evaluation_result["umux_score_statistics"] = calculate_umux_score_statistics(df)

    evaluation_result["umux_score_by_dimension"] = calculate_umux_score_by_dimension(df)

    evaluation_result["umux_score_by_sentiment"] = calculate_umux_score_by_sentiment(df)

    return evaluation_result


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        LABEL_COLUMN: [1, 2, 3, 0, 1],
        PREDICTED_LABEL_COLUMN: [1, 2, 3, 0, 2],
        PREDICTED_DIMENSION_COLUMN: [
            "Usefulness",
            "Ease of Use",
            "Usefulness + Ease of Use",
            "Tidak relevan dengan UMUX-Lite",
            "Ease of Use",
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

    print("Evaluasi klasifikasi:")
    classification_eval = evaluate_classification_dataframe(sample_data)
    print(classification_eval["classification_summary"])

    print("\nEvaluasi hasil akhir:")
    final_eval = evaluate_final_result(sample_data)

    for name, result in final_eval.items():
        print(f"\n{name}")
        print(result)