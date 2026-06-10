import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load model dan scaler 
model = joblib.load('decision_tree_model.pkl')
scaler = joblib.load('scaler_ispu.pkl')

# Label mapping 
label_map = {0: 'BAIK', 1: 'SEDANG', 2: 'TIDAK SEHAT'}

# 2. Desain Interface Web
st.title("Web Prediksi Kualitas Udara (ISPU) DKI Jakarta")
st.write("Masukkan nilai polutan di bawah ini untuk mengetahui kategori kualitas udara.")

# Buat input form sesuai urutan kolom_polutan di notebook
pm_sepuluh = st.number_input("PM10", min_value=0.0, value=50.0)
pm_duakomalima = st.number_input("PM2.5", min_value=0.0, value=70.0)
sulfur_dioksida = st.number_input("Sulfur Dioksida (SO2)", min_value=0.0, value=30.0)
karbon_monoksida = st.number_input("Karbon Monoksida (CO)", min_value=0.0, value=15.0)
ozon = st.number_input("Ozon (O3)", min_value=0.0, value=20.0)
nitrogen_dioksida = st.number_input("Nitrogen Dioksida (NO2)", min_value=0.0, value=25.0)

# 3. Proses Prediksi
if st.button("Prediksi Kualitas Udara"):
    # Gabungkan input jadi array
    data_input = np.array([[pm_sepuluh, pm_duakomalima, sulfur_dioksida, karbon_monoksida, ozon, nitrogen_dioksida]])
    
    # WAJIB: Standarisasi dulu datanya pake scaler yang di-load
    data_scaled = scaler.transform(data_input)
    
    # Prediksi pake Decision Tree
    prediksi = model.predict(data_scaled)
    hasil_kategori = label_map[prediksi[0]]
    
    # Tampilkan hasil
    st.subheader(f"Hasil Prediksi: Kategori {hasil_kategori}")