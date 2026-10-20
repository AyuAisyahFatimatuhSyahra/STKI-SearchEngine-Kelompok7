# 🎮 IoT-GamiSearch

**Mesin pencari dokumen penelitian Internet of Things (IoT) & Gamification** menggunakan metode **TF-IDF** dan **Vector Space Model (VSM)** dengan **Cosine Similarity**.

Aplikasi web ini melakukan **indeksasi, pembobotan kata, dan perangkingan dokumen secara otomatis** terhadap **1.000 dokumen penelitian** (judul + abstrak) berdasarkan kueri pengguna. Semua perhitungan TF-IDF & VSM **diimplementasikan sendiri** (tanpa `TfidfVectorizer`), sehingga setiap angka dapat ditelusuri langsung di antarmuka.

---

## ✨ Fitur

| No | Fitur | Halaman | Isi |
|---|---|---|---|
| 1 | **Halaman Utama (Dashboard Pencarian)** | `/` | Kotak pencarian + filter kategori, contoh kueri, riwayat pencarian, statistik indeks, alur kerja sistem, sebaran kategori, term paling sering muncul |
| 2 | **Sistem Pra-proses Teks (Preprocessing Pipeline)** | `/preprocessing` | Uji coba interaktif 5 tahap: *case folding → cleaning → tokenizing → stopword removal → stemming*, dengan hasil tiap tahap. Bisa mengambil dokumen acak dari dataset |
| 3 | **Mesin Hitung VSM & Perangkingan (Backend Engine)** | `/engine` | Rumus, statistik indeks, contoh matriks TF-IDF, distribusi df, tabel kosakata + *inverted index* (bisa difilter & diurutkan), API JSON |
| 4 | **Halaman Hasil Pencarian** | `/search?q=...` | Hasil terurut berdasarkan skor cosine, *highlight* kata yang cocok, skor %, waktu pencarian, filter kategori & tahun, paginasi, analisis vektor kueri, dan **rincian perhitungan cosine untuk setiap dokumen** |
| + | Detail Dokumen | `/document/<i>` | Abstrak lengkap, link sumber (DOI), term TF-IDF tertinggi, perhitungan skor terhadap kueri, 5 dokumen paling mirip |
| + | Dataset | `/dataset` | Statistik dataset, grafik per kategori & tahun, daftar dokumen dengan filter |

---

## 🧠 Metode

### 1. Preprocessing (`engine/preprocessing.py`)
1. **Case folding** – semua huruf menjadi kecil
2. **Cleaning** – menghapus URL, angka, tanda baca, simbol
3. **Tokenizing** – memecah teks menjadi kata (regex, tanpa unduhan data NLTK)
4. **Stopword removal** – kamus stopword Bahasa Indonesia (±350 kata) + Bahasa Inggris (±260 kata)
5. **Stemming** – dataset dwibahasa, sehingga bahasa tiap dokumen dideteksi otomatis:
   - Bahasa Indonesia → **Sastrawi** (algoritma Nazief–Adriani)
   - Bahasa Inggris → **Porter Stemmer** (NLTK)

### 2. Indexing & TF-IDF (`engine/tfidf.py`)

```
tf(t,d) = frekuensi term t pada dokumen d
df(t)   = jumlah dokumen yang memuat term t
idf(t)  = log10( N / df(t) )
w(t,d)  = tf(t,d) × idf(t)
```

Hasilnya disimpan dalam **inverted index** (`term → [(dokumen, tf), ...]`) dan vektor TF-IDF per dokumen. Indeks di-*cache* ke `cache/index.pkl` dan dibangun ulang otomatis bila dataset berubah.

### 3. Vector Space Model & Ranking (`engine/vsm.py`)

```
                  Σ w(t,q) · w(t,d)
cos(q,d) = ───────────────────────────────
            √Σ w(t,q)²  ×  √Σ w(t,d)²
```

Kueri diproses dengan pipeline yang sama dan diubah menjadi vektor TF-IDF. Melalui inverted index, hanya dokumen yang memuat minimal satu term kueri yang dihitung. Dokumen diurutkan dari skor cosine tertinggi; dokumen dengan skor 0 tidak ditampilkan.

Kebenaran perhitungan diuji di `tests/test_engine.py` (dibandingkan dengan hitungan manual dan perkalian matriks NumPy).

---

## 📚 Dataset

- **1.000 dokumen penelitian nyata** (judul, abstrak, kata kunci, penulis, tahun, jurnal, DOI) bertopik IoT, Gamification, IoT-Gamification, dan Smart System, dalam Bahasa Inggris & Indonesia.
- Sumber: **[OpenAlex](https://openalex.org)**, basis data publikasi ilmiah terbuka.
- Dikumpulkan dengan `scripts/fetch_dataset.py` → `dataset/iot_gamification_dataset.csv`

Kolom CSV: `id, title, abstract, keyword, category, year, authors, venue, language, doi, url, source_id`

> Ingin memakai dataset lain? Cukup siapkan CSV dengan minimal kolom `title` dan `abstract`, lalu simpan sebagai `dataset/iot_gamification_dataset.csv` (atau atur variabel lingkungan `DATASET_PATH`).

---

## 🚀 Cara Menjalankan

```bash
# 1. Clone
git clone https://github.com/<username>/IoT-GamiSearch.git
cd IoT-GamiSearch

# 2. Virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 3. Install library
pip install -r requirements.txt

# 4. Unduh 1.000 dokumen (sekali saja, butuh internet ± 1-2 menit)
python scripts/fetch_dataset.py

# 5. Jalankan aplikasi
python app.py
```

Buka **http://127.0.0.1:5000**. Saat pertama dijalankan, indeks dibangun beberapa detik lalu disimpan ke cache.

Uji perhitungan:

```bash
python -m unittest discover tests -v
```

API JSON: `GET /api/search?q=smart+home&k=5`

---

## 📁 Struktur Project

```
IoT-GamiSearch/
├── app.py                    # Flask: routing semua halaman + API
├── engine/
│   ├── preprocessing.py      # Fitur 2 – pipeline pra-proses teks
│   ├── stopwords.py          # Kamus stopword ID + EN
│   ├── tfidf.py              # Indexing, inverted index, bobot TF-IDF
│   └── vsm.py                # Fitur 3 – cosine similarity & ranking
├── scripts/
│   └── fetch_dataset.py      # Pengumpul 1.000 dokumen dari OpenAlex
├── dataset/
│   ├── iot_gamification_dataset.csv   # dataset utama (1.000 dokumen)
│   └── sample_dataset.csv             # 5 dokumen contoh (cadangan)
├── templates/                # base, index (F1), preprocessing (F2), engine (F3), result (F4), document, dataset
├── static/style.css
├── tests/test_engine.py
├── docs/SKENARIO_VIDEO_DEMO.md
└── requirements.txt
```

---

## 👥 Kelompok 7

| No | Nama | NIM |
|---|---|---|
| 1 | Claria Privana Ratna Wati | 23050974086 |
| 2 | Briant Tedja Mulya | 23050974087 |
| 3 | Ayu Aisyah Fatimatu Syahra | 23050974091 |
| 4 | Muhammad Rifky Wardhana | 23050974104 |
---

