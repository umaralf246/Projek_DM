# 🏙️ Dashboard Pemantauan & Prediksi Kualitas Udara (ISPU) DKI Jakarta

Aplikasi interaktif berbasis web untuk memantau tren historis kualitas udara di DKI Jakarta dan memprediksi kategori Indeks Standar Pencemar Udara (ISPU) menggunakan algoritma Machine Learning.

## 🚀 Fitur Utama

Aplikasi ini dibagi menjadi dua bagian utama untuk memudahkan petugas dalam melakukan analisis dan pengambilan keputusan:

### 1. Dashboard Overview (Analisis Historis)
* **Kondisi Terkini:** Menampilkan ringkasan status kualitas udara terbaru dari 5 stasiun pemantauan di DKI Jakarta, lengkap dengan parameter dominan (PM2.5).
* **Distribusi Kategori ISPU:** Visualisasi *bar chart* yang membandingkan total hari dengan kategori udara BAIK, SEDANG, dan TIDAK SEHAT antar stasiun selama periode pengamatan.
* **Feature Importance:** Interpretasi model (*Explainable AI*) menggunakan grafik horizontal yang menunjukkan seberapa besar pengaruh setiap polutan (PM2.5, SO2, PM10, CO, O3, NO2) terhadap hasil klasifikasi model Decision Tree.
* **Tren Fluktuasi Harian:** Grafik garis interaktif untuk melacak pergerakan nilai polutan tertentu dari hari ke hari, dilengkapi dengan filter dinamis (Periode, Stasiun, Parameter Polutan) dan penanganan otomatis untuk data yang kosong (*interpolasi*).

### 2. Prediksi ISPU (Machine Learning)
* **Form Input Manual:** Antarmuka interaktif bagi pengguna untuk memasukkan nilai polutan terkini.
* **Grafik Batas Aman:** Visualisasi *real-time* yang membandingkan nilai input pengguna dengan ambang batas kategori udara "BAIK" (ISPU = 50).
* **Hasil Analisis Cerdas:** Model *Decision Tree Classification* memproses data yang telah di-*scaling* untuk memprediksi apakah kualitas udara masuk dalam kategori **BAIK**, **SEDANG**, atau **TIDAK SEHAT**, lengkap dengan rekomendasi tindakannya.

## 🛠️ Teknologi yang Digunakan

* **Bahasa Pemrograman:** Python
* **Web Framework:** Streamlit
* **Data Manipulasi:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Decision Tree Classifier, StandardScaler)
* **Visualisasi Data:** Matplotlib
* **Model Persistance:** Joblib

## 📁 Struktur Repositori

* `app.py` — File utama yang berisi kode *frontend* Streamlit dan logika aplikasi.
* `data_ispu.csv` — Dataset historis kualitas udara DKI Jakarta.
* `decision_tree_model.pkl` — Model Machine Learning yang sudah dilatih (*pre-trained*).
* `scaler_ispu.pkl` — File scaler untuk normalisasi data input sebelum diprediksi.
* `requirements.txt` — Daftar *library* Python yang dibutuhkan untuk menjalankan aplikasi.

## 💻 Cara Menjalankan di Komputer Lokal

1. **Clone repositori ini** ke lokal mesin Anda:
   ```bash
   git clone [https://github.com/umaralf246/](https://github.com/umaralf246/)<nama-repo-lu>.git
   cd <nama-repo-lu>
