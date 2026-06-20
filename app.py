import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Prediksi ISPU Jakarta", page_icon="🏙️", layout="wide")

# --- LOAD DATA HISTORIS ---
# Gunakan st.cache_data agar Streamlit tidak membaca ulang CSV setiap kali ada perubahan input
@st.cache_data
def load_data():
    # UBAH 'data_ispu.csv' SESUAI NAMA FILE LU DI GITHUB
    df = pd.read_csv('data_ispu.csv') 
    
    # UBAH 'tanggal' SESUAI NAMA KOLOM TANGGAL DI DATASET LU
    # df['tanggal'] = pd.to_datetime(df['tanggal']) 
    return df

try:
    df_historis = load_data()
    data_tersedia = True
except FileNotFoundError:
    data_tersedia = False

# --- LOAD MODEL & SCALER ---
model = joblib.load('decision_tree_model.pkl')
scaler = joblib.load('scaler_ispu.pkl')
label_map = {0: 'BAIK', 1: 'SEDANG', 2: 'TIDAK SEHAT'}

st.title("🏙️ Dashboard Kualitas Udara (ISPU) DKI Jakarta")
st.markdown("Aplikasi untuk memantau tren historis dan memprediksi kualitas udara.")
st.divider()

# --- MEMBUAT TAB TAMPILAN ---
tab1, tab2 = st.tabs(["📈 Tren Historis per Stasiun", "🔍 Prediksi Kualitas Udara"])

# ==========================================
# TAB 1: VISUALISASI HISTORIS
# ==========================================
with tab1:
    st.header("Tren Kualitas Udara Historis")
    
    if data_tersedia:
        # Pilihan filter stasiun (UBAH 'stasiun' SESUAI NAMA KOLOM STASIUN LU)
        daftar_stasiun = df_historis['stasiun'].unique()
        stasiun_terpilih = st.selectbox("Pilih Stasiun Pemantau:", daftar_stasiun)
        
        # Filter data berdasarkan stasiun terpilih
        df_filter = df_historis[df_historis['stasiun'] == stasiun_terpilih]
        
        # Pilihan metrik polutan yang ingin dilihat trennya
        pilihan_polutan = st.selectbox("Pilih Parameter Polutan:", ['pm10', 'pm25', 'so2', 'co', 'o3', 'no2'])
        
        # Menampilkan line chart
        # Pastikan kolom 'tanggal' diset sebagai index agar sumbu X berupa waktu
        st.line_chart(df_filter.set_index('tanggal')[pilihan_polutan])
        
        # Menampilkan cuplikan data mentah (opsional)
        with st.expander("Lihat Data Tabel"):
            st.dataframe(df_filter[['tanggal', 'stasiun', pilihan_polutan]].tail(10))
    else:
        st.warning("File dataset (CSV) tidak ditemukan. Pastikan file CSV sudah di-upload ke GitHub.")

# ==========================================
# TAB 2: PREDIKSI (Sama seperti sebelumnya)
# ==========================================
with tab2:
    st.header("Prediksi Kategori ISPU")
    
    col_input1, col_input2 = st.columns(2)
    
    with col_input1:
        pm_sepuluh = st.number_input("PM10", 0.0, value=50.0)
        pm_duakomalima = st.number_input("PM2.5", 0.0, value=70.0)
        sulfur_dioksida = st.number_input("SO2 (Sulfur Dioksida)", 0.0, value=30.0)
        
    with col_input2:
        karbon_monoksida = st.number_input("CO (Karbon Monoksida)", 0.0, value=15.0)
        ozon = st.number_input("O3 (Ozon)", 0.0, value=20.0)
        nitrogen_dioksida = st.number_input("NO2 (Nitrogen Dioksida)", 0.0, value=25.0)

    st.divider()

    if st.button("Prediksi Kualitas Udara", use_container_width=True):
        data_input = np.array([[pm_sepuluh, pm_duakomalima, sulfur_dioksida, karbon_monoksida, ozon, nitrogen_dioksida]])
        data_scaled = scaler.transform(data_input)
        prediksi = model.predict(data_scaled)
        hasil_kategori = label_map[prediksi[0]]
        
        st.subheader("💡 Hasil Analisis")
        if hasil_kategori == 'BAIK':
            st.success("🟢 **Kategori: BAIK**")
        elif hasil_kategori == 'SEDANG':
            st.warning("🟡 **Kategori: SEDANG**")
        else:
            st.error("🔴 **Kategori: TIDAK SEHAT**")
