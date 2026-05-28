# Proyek Analisis Sentimen Ulasan Aplikasi Genshin Impact: Pendekatan Ensemble Learning

Repositori ini memuat implementasi praktis dari *Machine Learning & Deep Learning Workflow* untuk melakukan **Analisis Sentimen Multi-kelas** (Positif, Netral, Negatif) pada data ulasan pengguna (*user reviews*) aplikasi Genshin Impact berbahasa Indonesia. Proyek ini membandingkan algoritma klasifikasi tradisional dengan jaringan saraf tiruan (Deep Learning) serta menggabungkannya lewat mekanisme *Majority Voting*.

---

## Detail Dataset & Skenario Bisnis
Dataset berasal dari file lokal `genshin_full_review.csv` yang memuat teks ulasan asli dari pengguna, nilai rating/skor (1-5), dan tanggal ulasan.
* **Transformasi Label (Pelabelan):** Nilai rating dikelompokkan ke dalam 3 kelas sentimen:
  * Rating 1-2 $\rightarrow$ **Negatif** (Label `0`)
  * Rating 3 $\rightarrow$ **Netral** (Label `1`)
  * Rating 4-5 $\rightarrow$ **Positif** (Label `2`)

---

## Alur Preprocessing & NLP Pipeline

Proses penyiapan data teks (*Text Preprocessing*) dilakukan secara ketat karena karakteristik ulasan di Google Play Store banyak mengandung bahasa tidak baku (*slang words*):

1.  **Data Cleansing Awal:** Menghapus data kosong (*missing values*) dan baris duplikat menggunakan gabungan fungsi `dropna()` dan `drop_duplicates()`.
2.  **Case Folding:** Mengubah seluruh karakter teks menjadi huruf kecil (*lowercase*).
3.  **Regex Cleaning:** Menghapus *mention* (`@username`), *hashtag* (`#`), angka, tanda baca, serta tautan URL (`http...`).
4.  **Handling Negation (Penanganan Negasi):** Mengecualikan kata negasi kunci (`tidak`, `bukan`, `kurang`, `jangan`) dari daftar *Stopwords*, lalu menyatukannya dengan kata setelahnya (misal: `"tidak bagus"` $\rightarrow$ `"tidak_bagus"`) agar konteks negatif tetap tertangkap oleh model.
5.  **Slang Word Mapping:** Mengubah kata-kata singkatan/gaul menjadi kata baku menggunakan kamus pemetaan khusus (`slang_map`), seperti: `gk`/`ga` $\rightarrow$ `tidak`, `ngelag`/`lag` $\rightarrow$ `lambat`, `apk` $\rightarrow$ `aplikasi`.
6.  **Stopwords Removal:** Menghapus kata-kata umum yang tidak membawa esensi sentimen menggunakan daftar kata dari library `nltk` bahasa Indonesia.

---

## Eksplorasi Model Klasifikasi (Model Selection)

Proyek ini melatih dan menguji 3 buah model dengan arsitektur yang sepenuhnya berbeda:

### 1. Random Forest Classifier (Machine Learning)
* **Ekstraksi Fitur:** Menggunakan **TF-IDF Vectorizer** untuk mengubah kumpulan teks ulasan menjadi representasi matriks bobot frekuensi kata.
* **Arsitektur:** Menggunakan `RandomForestClassifier` berbasis *ensemble trees* dengan 150 estimator pohon untuk meminimalkan varians galat.

### 2. Support Vector Machine - SVM (Machine Learning)
* **Ekstraksi Fitur:** Menggunakan matriks TF-IDF yang sama dengan Random Forest.
* **Arsitektur:** Menggunakan `SVC` dengan konfigurasi `kernel='linear'` dan parameter penalti `C=1` untuk mencari batas pemisah (*hyperplane*) optimal antar-kelas sentimen.

### 3. Bidirectional LSTM - BiLSTM (Deep Learning)
* **Ekstraksi Fitur:** Menggunakan `Tokenizer` dari Keras dengan batas maksimal 15.000 kosakata unik (`num_words=15000`) dan penanganan kata baru (`<OOV>`), kemudian teks disamakan panjangnya memakai `pad_sequences`.
* **Arsitektur:** Jaringan *Recurrent Neural Network* yang terdiri dari lapisan `Embedding`, lapisan **Bidirectional LSTM** (agar model dapat membaca urutan teks dari depan ke belakang maupun sebaliknya), lapisan `Dropout` untuk mencegah overfitting, dan ditutup dengan lapisan *Dense* beraktivasi `softmax` untuk klasifikasi multi-kelas.
* **Class Weighting:** Menerapkan fungsi `compute_class_weight` saat proses *training* untuk mengatasi masalah ketidakseimbangan jumlah sampel ulasan antar-kelas sentimen.

---

## Sistem Prediksi Akhir: Majority Voting Ensemble

Untuk meningkatkan stabilitas dan akurasi prediksi pada data ulasan baru di lingkungan produksi, proyek ini mengimplementasikan fungsi kustom `voting()`:

* Fungsi ini menerima hasil prediksi individual dari model **Random Forest**, **SVM**, dan **BiLSTM**.
* Menggunakan modul `collections.Counter` untuk mengambil keputusan berdasarkan **suara terbanyak** (*Majority Voting*). Jika terjadi perbedaan pendapat antar-model, sistem akan memicu indikasi peringatan (`Model berbeda pendapat`), namun tetap mengeluarkan hasil prediksi akhir yang paling stabil secara konsensus.

---

## Library Python Utama yang Digunakan
* `pandas` & `numpy` - Manipulasi matriks dan tabel ulasan.
* `nltk` - Tokenisasi dan penyaringan *stopwords* bahasa Indonesia.
* `matplotlib` & `seaborn` - Visualisasi grafik distribusi rating dan panjang kata.
* `wordcloud` - Membuat visualisasi awan kata kata kunci sentimen.
* `scikit-learn` - Implementasi TF-IDF Vectorizer, pembagian data (*stratified split*), model RF, SVM, dan laporan evaluasi metrik klasifikasi.
* `tensorflow.keras` - Pembuatan arsitektur model Deep Learning BiLSTM dan pra-pemrosesan sekuens teks.
