import re
import pandas as pd

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


# ============================================================
# STEMMER AND STOPWORDS INITIALIZATION
# ============================================================

stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

stopword_factory = StopWordRemoverFactory()
default_stopwords = set(stopword_factory.get_stop_words())


# ============================================================
# IMPORTANT WORDS CONFIGURATION
# ============================================================

# Kata negasi sebaiknya tidak dihapus karena berpengaruh pada sentimen.
# Contoh:
# "tidak mudah" berbeda makna dengan "mudah"
NEGATION_WORDS = {
    "tidak",
    "bukan",
    "belum",
    "jangan",
    "kurang",
    "tanpa",
    "tak",
    "ga",
    "gak",
    "nggak",
    "ngga",
    "enggak",
    "kagak",
    "ndak",
    "gk",
    "tdk",
}


# Kata penting yang berkaitan dengan UMUX-Lite dan konteks aplikasi JMO.
# Kata-kata ini tidak boleh hilang saat stopword removal.
IMPORTANT_WORDS = {
    # UMUX-Lite: usefulness / memenuhi kebutuhan
    "guna",
    "berguna",
    "manfaat",
    "bermanfaat",
    "bantu",
    "membantu",
    "terbantu",
    "butuh",
    "kebutuhan",
    "sesuai",
    "layan",
    "layanan",
    "klaim",
    "saldo",
    "cair",
    "pencairan",
    "jht",
    "bpjs",
    "ketenagakerjaan",
    "informasi",
    "fitur",
    "kartu",
    "peserta",
    "data",
    "profil",
    "rekening",
    "bank",
    "transfer",
    "dana",

    # UMUX-Lite: ease of use / mudah digunakan
    "mudah",
    "gampang",
    "praktis",
    "simpel",
    "simple",
    "sederhana",
    "jelas",
    "paham",
    "sulit",
    "susah",
    "ribet",
    "rumit",
    "bingung",
    "pakai",
    "akses",
    "login",
    "masuk",
    "daftar",
    "registrasi",
    "verifikasi",
    "validasi",
    "buka",
    "gunakan",
    "guna",

    # Masalah teknis / usability issue
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
    "force",
    "close",
    "forceclose",
    "hang",
    "macet",
    "hilang",
    "kosong",
    "pending",
    "blokir",
    "terblokir",
    "jaringan",
    "koneksi",
    "internet",
    "update",
    "upgrade",
    "bug",
    "blank",
    "stuck",

    # Kata bantu penting
    "bisa",
    "tidak",
    "belum",
    "kurang",
    "gak",
    "ga",
    "nggak",
    "ngga",
    "tdk",
}


# Stopword final:
# - stopword default Sastrawi tetap dipakai,
# - kata negasi tidak dihapus,
# - kata penting UMUX/JMO tidak dihapus.
STOPWORDS = (
    default_stopwords
    .difference(NEGATION_WORDS)
    .difference(IMPORTANT_WORDS)
)


# ============================================================
# SLANG / INFORMAL WORD NORMALIZATION
# ============================================================

NORMALIZATION_DICT = {
    # Negasi informal
    "gk": "tidak",
    "ga": "tidak",
    "gak": "tidak",
    "ngga": "tidak",
    "nggak": "tidak",
    "enggak": "tidak",
    "tdk": "tidak",
    "tak": "tidak",
    "ndak": "tidak",
    "kagak": "tidak",

    # Bentuk gabungan umum
    "gabisa": "tidak bisa",
    "gakbisa": "tidak bisa",
    "nggakbisa": "tidak bisa",
    "tidakbisa": "tidak bisa",
    "tdkbisa": "tidak bisa",

    # Typo / variasi umum
    "eror": "error",
    "lemot": "lambat",
    "lelet": "lambat",
    "apk": "aplikasi",
    "app": "aplikasi",
    "aplikasinya": "aplikasi",
    "apps": "aplikasi",
    "loginya": "login",
    "loginnya": "login",
    "otpnya": "otp",
    "servernya": "server",
    "loadingnya": "loading",
    "saldoanya": "saldo",
    "saldonya": "saldo",
    "jaringannya": "jaringan",

    # Kata positif informal
    "mantul": "mantap",
    "mantab": "mantap",
    "recommended": "rekomendasi",
    "recomended": "rekomendasi",
    "rekomended": "rekomendasi",

    # Kata negatif informal
    "ribettt": "ribet",
    "ribett": "ribet",
    "parahh": "parah",
    "susaah": "susah",
    "sussah": "susah",
}


# ============================================================
# BASIC PREPROCESSING FUNCTIONS
# ============================================================

def case_folding(text):
    """
    Mengubah teks menjadi huruf kecil.
    """
    if pd.isna(text):
        return ""

    return str(text).lower()


def cleaning_text(text):
    """
    Membersihkan teks dari URL, mention, hashtag, angka,
    tanda baca, dan karakter selain huruf.

    Tahap ini dibuat sederhana agar sesuai dengan alur preprocessing:
    cleaning.
    """
    text = str(text)

    # Menghapus URL
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)

    # Menghapus mention username
    text = re.sub(r"@\w+", " ", text)

    # Menghapus simbol hashtag, tetapi kata setelah hashtag tetap dipertahankan
    text = re.sub(r"#", " ", text)

    # Menghapus angka
    text = re.sub(r"\d+", " ", text)

    # Menghapus karakter selain huruf dan spasi
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Menghapus spasi berlebih
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenizing_text(text):
    """
    Memecah teks menjadi token/kata.
    """
    if not text:
        return []

    return text.split()


def normalize_tokens(tokens):
    """
    Menormalisasi kata tidak baku menjadi bentuk yang lebih konsisten.

    Contoh:
    - "gk" menjadi "tidak"
    - "gabisa" menjadi "tidak bisa"
    - "eror" menjadi "error"

    Jika hasil normalisasi terdiri dari lebih dari satu kata,
    maka akan dipecah kembali menjadi token.
    """
    normalized_tokens = []

    for token in tokens:
        normalized = NORMALIZATION_DICT.get(token, token)

        if " " in normalized:
            normalized_tokens.extend(normalized.split())
        else:
            normalized_tokens.append(normalized)

    return normalized_tokens


def remove_stopwords(tokens):
    """
    Menghapus stopword dari daftar token.
    Kata negasi dan kata penting UMUX/JMO tetap dipertahankan.
    """
    return [token for token in tokens if token not in STOPWORDS]


def stemming_tokens(tokens):
    """
    Mengubah token menjadi bentuk kata dasar menggunakan Sastrawi.
    """
    stemmed_tokens = []

    for token in tokens:
        stemmed_token = stemmer.stem(token)
        stemmed_tokens.append(stemmed_token)

    return stemmed_tokens


def join_tokens(tokens):
    """
    Menggabungkan token menjadi teks kembali.
    """
    return " ".join(tokens)


# ============================================================
# MAIN PREPROCESSING FUNCTION
# ============================================================

def preprocess_text(text):
    """
    Menjalankan seluruh tahapan preprocessing:
    1. case folding
    2. cleaning
    3. tokenizing
    4. normalisasi kata tidak baku
    5. stemming
    6. stopword removal

    Output berupa teks bersih.
    """
    text = case_folding(text)
    text = cleaning_text(text)

    tokens = tokenizing_text(text)

    # Normalisasi dilakukan sebelum stemming agar kata tidak baku lebih rapi.
    tokens = normalize_tokens(tokens)

    # Urutan utama tetap mengikuti diagram:
    # tokenizing → stemming → stopword removal
    tokens = stemming_tokens(tokens)
    tokens = remove_stopwords(tokens)

    clean_text = join_tokens(tokens)

    return clean_text


def preprocess_text_with_tokens(text):
    """
    Versi tambahan jika ingin melihat hasil token.
    Fungsi ini berguna untuk debugging atau eksplorasi data.
    """
    text = case_folding(text)
    text = cleaning_text(text)

    tokens = tokenizing_text(text)
    normalized_tokens = normalize_tokens(tokens)
    stemmed_tokens = stemming_tokens(normalized_tokens)
    filtered_tokens = remove_stopwords(stemmed_tokens)

    return {
        "case_folding": text,
        "tokens": tokens,
        "normalized_tokens": normalized_tokens,
        "stemmed_tokens": stemmed_tokens,
        "filtered_tokens": filtered_tokens,
        "clean_text": join_tokens(filtered_tokens),
    }


# ============================================================
# DATAFRAME PREPROCESSING FUNCTION
# ============================================================

def preprocess_dataframe(df, text_column, output_column="clean_text"):
    """
    Melakukan preprocessing pada kolom teks di DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset dalam bentuk DataFrame.

    text_column : str
        Nama kolom yang berisi teks review.

    output_column : str
        Nama kolom baru untuk menyimpan hasil preprocessing.

    Returns
    -------
    pandas.DataFrame
        DataFrame dengan tambahan kolom hasil preprocessing.
    """
    if text_column not in df.columns:
        raise ValueError(
            f"Kolom '{text_column}' tidak ditemukan di dataset. "
            f"Kolom yang tersedia: {list(df.columns)}"
        )

    df = df.copy()
    df[output_column] = df[text_column].apply(preprocess_text)

    return df


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":
    sample_reviews = [
        "Aplikasi JMO ini TIDAK mudah digunakan!!! Login gagal terus, ribet banget.",
        "Sangat berguna dan membantu untuk klaim JHT.",
        "Gabisa login, OTP tidak masuk, server error.",
        "Aplikasi bagus tapi saldo tidak muncul.",
    ]

    for sample_review in sample_reviews:
        result = preprocess_text_with_tokens(sample_review)

        print("=" * 80)
        print("Review asli:")
        print(sample_review)

        print("\nHasil preprocessing:")
        print(result["clean_text"])

        print("\nDetail token:")
        print(result)