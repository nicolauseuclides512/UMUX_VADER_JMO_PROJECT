# UMUX-Lite and VADER Sentiment Analysis for JMO Application Reviews

Project ini digunakan untuk melakukan analisis usability review aplikasi JMO dari Google Play Store dengan menggabungkan pendekatan **UMUX-Lite** dan **VADER Sentiment Analysis**.

Alur penelitian yang digunakan adalah:

```text
Data Collection
(Scraping Google Play Store)
        ↓
Preprocessing
(case folding, cleaning, tokenizing, stemming, stopword removal)
        ↓
Classification Algorithm
UMUX-Lite Labeling
(Label 0, 1, 2, 3)
        ↓
VADER Compound Score Mapping
to UMUX-Lite Score (1–7)
        ↓
Result & Interpretation
```

## UMUX-Lite Label

Label UMUX-Lite yang digunakan dalam penelitian ini adalah:

| Label | Keterangan                                         |
| ----: | -------------------------------------------------- |
|     0 | Review tidak relevan dengan UMUX-Lite              |
|     1 | Review berkaitan dengan usefulness                 |
|     2 | Review berkaitan dengan ease of use                |
|     3 | Review berkaitan dengan usefulness dan ease of use |

## Struktur Project

```text
UMUX_VADER_JMO_PROJECT_STARTER/
├── README.md
├── requirements.txt
├── train_umux_labeler.py
├── run_umux_vader_pipeline.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── vaderSentiment/
│   ├── __init__.py
│   ├── vaderSentiment.py
│   ├── vader_lexicon.txt
│   └── emoji_utf8_lexicon.txt
│
├── custom_lexicon/
│   └── indonesian_vader_lexicon.tsv
│
├── models/
├── output/
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── umux_labeler.py
│   ├── vader_analyzer.py
│   ├── mapping_score.py
│   ├── evaluation.py
│   └── reporting.py
│
├── notebooks/
│   └── 01_exploration.ipynb
│
└── docs/
    ├── research_flow.md
    └── label_guideline.md
```

## Cara Instalasi

Buat virtual environment terlebih dahulu:

```bash
python -m venv venv
```

Aktifkan virtual environment.

Untuk Windows:

```bash
venv\Scripts\activate
```

Untuk Linux/Mac:

```bash
source venv/bin/activate
```

Install library yang dibutuhkan:

```bash
pip install -r requirements.txt
```

## Deploy ke Streamlit Cloud

Project ini sudah disiapkan agar bisa dijalankan di Streamlit Cloud.

Langkah deploy:

1. Push seluruh folder project ke repository GitHub.
2. Buka Streamlit Cloud dan pilih repository tersebut.
3. Isi main file path dengan:

```text
app.py
```

4. Pastikan file berikut ikut ada di repository:

```text
requirements.txt
runtime.txt
.streamlit/config.toml
app.py
src/
output/hasil_umux_vader.csv
```

Jika `output/hasil_umux_vader.csv` tidak ikut di-upload ke repository, dashboard tetap bisa digunakan dengan memilih menu upload file hasil analisis pada sidebar aplikasi.

## Data Input

Letakkan dataset review aplikasi JMO pada folder:

```text
data/raw/
```

Contoh nama file:

```text
data/raw/LABEL Data Review JMO.xlsx
```

Dataset minimal memiliki kolom teks review. Jika digunakan untuk pelatihan model klasifikasi UMUX-Lite, dataset juga perlu memiliki kolom label UMUX-Lite.

Contoh format dataset:

| review                          | label_umux |
| ------------------------------- | ---------: |
| Aplikasinya mudah digunakan     |          2 |
| Sangat membantu untuk klaim JHT |          1 |
| Mudah dan sangat bermanfaat     |          3 |
| Bagus                           |          0 |

## Menjalankan Training Model

File berikut digunakan untuk melatih model klasifikasi UMUX-Lite:

```bash
python train_umux_labeler.py
```

Model hasil training akan disimpan ke folder:

```text
models/
```

Contoh output model:

```text
models/umux_lite_labeler.pkl
```

## Menjalankan Pipeline Analisis

File berikut digunakan untuk menjalankan keseluruhan proses analisis:

```bash
python run_umux_vader_pipeline.py
```

Output hasil analisis akan disimpan ke folder:

```text
output/
```

Output yang dihasilkan:

```text
output/hasil_umux_vader.xlsx
output/hasil_umux_vader.csv
output/summary_umux_vader.csv
```

## Penjelasan Singkat Proses

Tahap preprocessing digunakan untuk membersihkan data review sebelum dilakukan klasifikasi dan analisis sentimen. Tahapan preprocessing meliputi case folding, cleaning, tokenizing, stemming, dan stopword removal.

Setelah preprocessing, review diklasifikasikan ke dalam label UMUX-Lite, yaitu label 0, 1, 2, atau 3. Selanjutnya, setiap review dianalisis menggunakan VADER untuk memperoleh nilai compound score. Nilai compound score tersebut kemudian dipetakan ke dalam skala UMUX-Lite 1–7.

Rumus mapping yang digunakan adalah:

```text
UMUX-Lite Score = 1 + ((compound + 1) × 3)
```

Dengan rumus tersebut, nilai VADER compound score dari rentang -1 sampai +1 dikonversi menjadi skor UMUX-Lite dari rentang 1 sampai 7.

## Catatan

Project ini menggunakan VADER sebagai metode analisis sentimen berbasis leksikon. File utama VADER tidak dimodifikasi secara langsung. Penyesuaian untuk Bahasa Indonesia dan domain aplikasi JMO dapat dilakukan melalui file tambahan:

```text
custom_lexicon/indonesian_vader_lexicon.tsv
```
