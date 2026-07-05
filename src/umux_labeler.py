import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.config import (
    CLEAN_TEXT_COLUMN,
    LABEL_COLUMN,
    PREDICTED_LABEL_COLUMN,
    PREDICTED_DIMENSION_COLUMN,
    MODEL_FILE_PATH,
    UMUX_LABEL_MAP,
    RANDOM_STATE,
    TEST_SIZE,
    TFIDF_NGRAM_RANGE,
    TFIDF_MIN_DF,
    TFIDF_MAX_DF,
)

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


# ============================================================
# LABEL HELPER
# ============================================================

def get_umux_dimension(label):
    """
    Mengubah label numerik UMUX-Lite menjadi nama kategori.

    Label:
    0 = Tidak relevan dengan UMUX-Lite
    1 = Usefulness
    2 = Ease of Use
    3 = Usefulness + Ease of Use
    """
    try:
        label = int(label)
    except (ValueError, TypeError):
        return "Unknown"

    return UMUX_LABEL_MAP.get(label, "Unknown")


def validate_umux_labels(df, label_column=LABEL_COLUMN):
    """
    Memastikan kolom label hanya berisi nilai 0, 1, 2, atau 3.
    """
    if label_column not in df.columns:
        raise ValueError(
            f"Kolom label '{label_column}' tidak ditemukan. "
            f"Kolom yang tersedia: {list(df.columns)}"
        )

    valid_labels = set(UMUX_LABEL_MAP.keys())
    dataset_labels = set(df[label_column].dropna().astype(int).unique())

    invalid_labels = dataset_labels.difference(valid_labels)

    if invalid_labels:
        raise ValueError(
            f"Ditemukan label tidak valid: {invalid_labels}. "
            f"Label yang diperbolehkan hanya: {valid_labels}"
        )

    return True


# ============================================================
# MODEL BUILDER
# ============================================================

def build_umux_labeler():
    """
    Membuat model klasifikasi UMUX-Lite dengan Random Oversampling.

    Model:
    TF-IDF word n-gram + Random Oversampling + Linear SVM

    Oversampling digunakan untuk membantu label minoritas,
    terutama label 3: Usefulness + Ease of Use.

    Catatan:
    Oversampling hanya terjadi pada data training saat model.fit().
    Data testing tidak ikut diover-sampling.
    """
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
            )
        ),
        (
            "oversampling",
            RandomOverSampler(
                sampling_strategy={
                    3: 1500,
                },
                random_state=RANDOM_STATE,
            )
        ),
        (
            "classifier",
            LinearSVC(
                C=1.0,
                random_state=RANDOM_STATE,
            )
        )
    ])

    return model


# ============================================================
# DATA SPLITTING
# ============================================================

def split_dataset(
    df,
    text_column=CLEAN_TEXT_COLUMN,
    label_column=LABEL_COLUMN,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
):
    """
    Membagi dataset menjadi data latih dan data uji.
    """
    if text_column not in df.columns:
        raise ValueError(
            f"Kolom teks '{text_column}' tidak ditemukan. "
            f"Kolom yang tersedia: {list(df.columns)}"
        )

    if label_column not in df.columns:
        raise ValueError(
            f"Kolom label '{label_column}' tidak ditemukan. "
            f"Kolom yang tersedia: {list(df.columns)}"
        )

    df = df.dropna(subset=[text_column, label_column]).copy()

    # Pastikan label bertipe integer
    df[label_column] = df[label_column].astype(int)

    validate_umux_labels(df, label_column)

    X = df[text_column]
    y = df[label_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


# ============================================================
# TRAINING
# ============================================================

def train_umux_labeler(X_train, y_train):
    """
    Melatih model klasifikasi UMUX-Lite.
    """
    model = build_umux_labeler()
    model.fit(X_train, y_train)

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate_umux_labeler(model, X_test, y_test):
    """
    Mengevaluasi model klasifikasi UMUX-Lite.

    Output berupa dictionary yang berisi:
    - accuracy
    - classification_report_text
    - classification_report_dict
    - confusion_matrix
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    report_text = classification_report(
        y_test,
        y_pred,
        labels=list(UMUX_LABEL_MAP.keys()),
        target_names=list(UMUX_LABEL_MAP.values()),
        zero_division=0,
    )

    report_dict = classification_report(
        y_test,
        y_pred,
        labels=list(UMUX_LABEL_MAP.keys()),
        target_names=list(UMUX_LABEL_MAP.values()),
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=list(UMUX_LABEL_MAP.keys()),
    )

    cm_df = pd.DataFrame(
        cm,
        index=[f"actual_{label}" for label in UMUX_LABEL_MAP.keys()],
        columns=[f"predicted_{label}" for label in UMUX_LABEL_MAP.keys()],
    )

    evaluation_result = {
        "accuracy": accuracy,
        "classification_report_text": report_text,
        "classification_report_dict": report_dict,
        "confusion_matrix": cm_df,
    }

    return evaluation_result


# ============================================================
# SAVE AND LOAD MODEL
# ============================================================

def save_umux_labeler(model, model_path=MODEL_FILE_PATH):
    """
    Menyimpan model klasifikasi UMUX-Lite ke file .pkl.
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    return model_path


def load_umux_labeler(model_path=MODEL_FILE_PATH):
    """
    Memuat model klasifikasi UMUX-Lite dari file .pkl.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan di: {model_path}. "
            f"Jalankan train_umux_labeler.py terlebih dahulu."
        )

    model = joblib.load(model_path)

    return model


# ============================================================
# PREDICTION
# ============================================================

def predict_umux_labels(
    df,
    model,
    text_column=CLEAN_TEXT_COLUMN,
    output_label_column=PREDICTED_LABEL_COLUMN,
    output_dimension_column=PREDICTED_DIMENSION_COLUMN,
):
    """
    Melakukan prediksi label UMUX-Lite pada DataFrame.

    Input:
    - DataFrame yang sudah memiliki kolom clean_text
    - model klasifikasi UMUX-Lite

    Output:
    - DataFrame dengan tambahan kolom predicted_umux_label
    - DataFrame dengan tambahan kolom predicted_umux_dimension
    """
    if text_column not in df.columns:
        raise ValueError(
            f"Kolom teks '{text_column}' tidak ditemukan. "
            f"Kolom yang tersedia: {list(df.columns)}"
        )

    df = df.copy()

    df[output_label_column] = model.predict(df[text_column])
    df[output_dimension_column] = df[output_label_column].apply(get_umux_dimension)

    return df


# ============================================================
# TRAINING PIPELINE HELPER
# ============================================================

def train_evaluate_and_save(
    df,
    text_column=CLEAN_TEXT_COLUMN,
    label_column=LABEL_COLUMN,
    model_path=MODEL_FILE_PATH,
):
    """
    Fungsi ringkas untuk:
    1. membagi dataset
    2. melatih model
    3. mengevaluasi model
    4. menyimpan model

    Fungsi ini akan dipanggil di file train_umux_labeler.py.
    """
    X_train, X_test, y_train, y_test = split_dataset(
        df=df,
        text_column=text_column,
        label_column=label_column,
    )

    model = train_umux_labeler(X_train, y_train)

    evaluation_result = evaluate_umux_labeler(
        model=model,
        X_test=X_test,
        y_test=y_test,
    )

    save_umux_labeler(model, model_path)

    return model, evaluation_result