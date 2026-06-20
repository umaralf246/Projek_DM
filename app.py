import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Konfigurasi Halaman
st.set_page_config(page_title="Dashboard ISPU Jakarta", page_icon="🏙️", layout="wide")

# ==========================================
# FUNGSI LOAD DATA & MODEL
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv('data_ispu.csv')
    return df

try:
    df_historis = load_data()
    data_tersedia = True
except FileNotFoundError:
    data_tersedia = False

try:
    model = joblib.load('decision_tree_model.pkl')
    scaler = joblib.load('scaler_ispu.pkl')
except:
    st.error("File model atau scaler belum ada di repositori.")

label_map = {0: 'BAIK', 1: 'SEDANG', 2: 'TIDAK SEHAT'}

st.title("🏙️ Dashboard Kualitas Udara (ISPU) DKI Jakarta")
st.markdown("Pemantauan historis dan prediksi kualitas udara menggunakan Machine Learning.")
st.divider()

# Bikin 4 Tab Utama Sekarang
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard Overview", 
    "📈 Analisis Tren Polutan", 
    "🔍 Prediksi ISPU", 
    "🧠 Interpretasi Model"
])

# ==========================================
# TAB 1: DASHBOARD OVERVIEW (ELEMEN 1 & 2)
# ==========================================
with tab1:
    if data_tersedia:
        st.subheader("1. Kondisi Terkini per Stasiun")
        st.markdown("Ringkasan kualitas udara dari pencatatan data terakhir pada masing-masing stasiun.")
        
        df_terkini = df_historis.sort_values(['periode_data', 'tanggal']).groupby('stasiun').tail(1).reset_index()
        cols = st.columns(len(df_terkini))
        
        for idx, row in enumerate(df_terkini.itertuples()):
            with cols[idx]:
                with st.container(border=True):
                    if row.kategori == 'BAIK':
                        st.success("🟢 BAIK")
                    elif row.kategori == 'SEDANG':
                        st.warning("🟡 SEDANG")
                    else:
                        st.error("🔴 TIDAK SEHAT")
                    
                    st.markdown(f"**{row.stasiun}**")
                    st.metric(label="PM2.5 (Dominan)", value=f"{row.pm_duakomalima}")
                    st.caption(f"PM10: {row.pm_sepuluh} | SO2: {row.sulfur_dioksida} | CO: {row.karbon_monoksida}")

        st.divider()

        st.subheader("2. Distribusi Kategori Kualitas Udara per Stasiun")
        st.markdown("Total jumlah hari berdasarkan kategori ISPU selama seluruh periode pengamatan.")
        
        df_dist = df_historis.groupby(['stasiun', 'kategori']).size().unstack(fill_value=0)
        for cat in ['BAIK', 'SEDANG', 'TIDAK SEHAT']:
            if cat not in df_dist.columns:
                df_dist[cat] = 0
        df_dist = df_dist[['BAIK', 'SEDANG', 'TIDAK SEHAT']]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(df_dist.index))
        width = 0.25
        
        rects1 = ax.bar(x - width, df_dist['BAIK'], width, label='BAIK', color='#2ca02c') 
        rects2 = ax.bar(x, df_dist['SEDANG'], width, label='SEDANG', color='#ff7f0e') 
        rects3 = ax.bar(x + width, df_dist['TIDAK SEHAT'], width, label='TIDAK SEHAT', color='#d62728') 
        
        ax.set_ylabel('Jumlah Hari')
        ax.set_title('Perbandingan Kategori Udara Antar Stasiun')
        ax.set_xticks(x)
        ax.set_xticklabels(df_dist.index)
        ax.legend()
        
        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)
        ax.bar_label(rects3, padding=3)
        
        st.pyplot(fig)

    else:
        st.warning("Data CSV tidak ditemukan.")

# ==========================================
# TAB 2: TREN POLUTAN (ELEMEN 3)
# ==========================================
with tab2:
    if data_tersedia:
        st.subheader("Tren Fluktuasi Polutan Harian")
        
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        
        with col_filter1:
            daftar_periode = sorted(df_historis['periode_data'].unique())
            def format_periode(periode):
                p_str = str(periode)
                tahun, bulan = p_str[:4], p_str[4:]
                map_bulan = {"01": "Januari", "02": "Februari", "03": "Maret", "04": "April", "05": "Mei", "06": "Juni", "07": "Juli", "08": "Agustus", "09": "September", "10": "Oktober", "11": "November", "12": "Desember"}
                return f"{map_bulan.get(bulan, bulan)} {tahun}"
                
            periode_terpilih = st.selectbox("1. Pilih Periode:", daftar_periode, format_func=format_periode)
        
        df_filter_periode = df_historis[df_historis['periode_data'] == periode_terpilih]
        
        with col_filter2:
            daftar_stasiun = sorted(df_filter_periode['stasiun'].unique())
            stasiun_terpilih = st.selectbox("2. Pilih Stasiun:", daftar_stasiun)
            
        with col_filter3:
            opsi_polutan = {
                'PM2.5': 'pm_duakomalima', 'PM10': 'pm_sepuluh', 'SO2 (Sulfur Dioksida)': 'sulfur_dioksida',
                'CO (Karbon Monoksida)': 'karbon_monoksida', 'O3 (Ozon)': 'ozon', 'NO2 (Nitrogen Dioksida)': 'nitrogen_dioksida'
            }
            label_terpilih = st.selectbox("3. Pilih Parameter Polutan:", list(opsi_polutan.keys()))
            kolom_aktual = opsi_polutan[label_terpilih]

        df_filter = df_filter_periode[df_filter_periode['stasiun'] == stasiun_terpilih]
        df_filter = df_filter.sort_values('tanggal').reset_index(drop=True)
        
        df_filter[kolom_aktual] = pd.to_numeric(df_filter[kolom_aktual], errors='coerce')
        df_filter[kolom_aktual] = df_filter[kolom_aktual].interpolate(method='linear')
        
        if not df_filter.empty:
            st.line_chart(
                df_filter, x='tanggal', y=kolom_aktual,
                x_label=f"Tanggal Pemantauan ({format_periode(periode_terpilih)})",
                y_label=f"Nilai Konsentrasi {label_terpilih}"
            )
            
            with st.expander("Lihat Data Tabel"):
                st.dataframe(df_filter[['tanggal', 'stasiun', 'periode_data
