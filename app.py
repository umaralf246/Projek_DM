import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Konfigurasi Halaman (Wajib ditaruh di baris paling atas)
st.set_page_config(
    page_title="Prediksi ISPU Jakarta",
    page_icon="🏙️",
    layout="centered"
)

# Load model dan scaler
model = joblib.load('decision_tree_model.pkl')
scaler = joblib.load('scaler_ispu.pkl')

# Label mapping
label_map = {0: 'BAIK', 1: 'SEDANG', 2: 'TIDAK SEHAT'}

# 2. Judul dan Deskripsi
st.title("🏙️ Prediksi Kualitas Udara (ISPU) DKI Jakarta")
st.markdown("Aplikasi berbasis Machine Learning untuk memprediksi kategori kualitas udara berdasarkan konsentrasi polutan.")
st.divider() # Garis pembatas

# 3. Sidebar untuk Input Data
st.sidebar.header("⚙️ Panel Parameter Input")
st.sidebar.write("Geser slider untuk mengubah nilai polutan:")

# Menggunakan slider di sidebar agar lebih visual
pm_sepuluh = st.sidebar.slider("PM10", 0.0, 200.0, 50.0)
pm_duakomalima = st.sidebar.slider("PM2.5", 0.0, 200.0, 70.0)
sulfur_dioksida = st.sidebar.slider("SO2 (Sulfur Dioksida)", 0.0, 150.0, 30.0)
karbon_monoksida = st.sidebar.slider("CO (Karbon Monoksida)", 0.0, 100.0, 15.0)
ozon = st.sidebar.slider("O3 (Ozon)", 0.0, 150.0, 20.0)
nitrogen_dioksida = st.sidebar.slider("NO2 (Nitrogen Dioksida)", 0.0, 100.0, 25.0)

# 4. Tampilan Dashboard Metrik di Halaman Utama
st.subheader("📊 Nilai Polutan Saat Ini")
col1, col2, col3 = st.columns(3)
col1.metric(label="PM10", value=f"{pm_sepuluh}")
col2.metric(label="PM2.5", value=f"{pm_duakomalima}")
col3.metric(label="SO2", value=f"{sulfur_dioksida}")

col4, col5, col6 = st.columns(3)
col4.metric(label="CO", value=f"{karbon_monoksida}")
col5.metric(label="Ozon (O3)", value=f"{ozon}")
col6.metric(label="NO2", value=f"{nitrogen_dioksida}")

st.divider()

# 5. Tombol Eksekusi & Hasil Prediksi
if st.button("🔍 Prediksi Kualitas Udara", use_container_width=True):
    # Proses standar
    data_input = np.array([[pm_sepuluh, pm_duakomalima, sulfur_dioksida, karbon_monoksida, ozon, nitrogen_dioksida]])
    data_scaled = scaler.transform(data_input)
    prediksi = model.predict(data_scaled)
    hasil_kategori = label_map[prediksi[0]]
    
    # 6. Tampilan Hasil dengan Indikator Warna
    st.subheader("💡 Hasil Analisis")
    
    if hasil_kategori == 'BAIK':
        st.success("🟢 **Kategori: BAIK** \n\nTingkat kualitas udara sangat baik, tidak memberikan efek negatif terhadap manusia ataupun hewan.")
    elif hasil_kategori == 'SEDANG':
        st.warning("🟡 **Kategori: SEDANG** \n\nTingkat kualitas udara masih dapat diterima, namun kelompok sensitif disarankan untuk mengurangi aktivitas berat di luar ruangan.")
    else:
        st.error("🔴 **Kategori: TIDAK SEHAT** \n\nTingkat kualitas udara bersifat merugikan pada manusia atau kelompok hewan yang sensitif. Segera gunakan masker jika harus keluar ruangan!")
