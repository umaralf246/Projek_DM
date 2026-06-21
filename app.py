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

# Sekarang cuma jadi 2 Tab Utama
tab1, tab2 = st.tabs([
    "📊 Dashboard Overview", 
    "🔍 Prediksi ISPU"
])

# ==========================================
# TAB 1: DASHBOARD OVERVIEW (SEMUA ANALISIS DIGABUNG DI SINI)
# ==========================================
with tab1:
    if data_tersedia:
        # --- BAGIAN 1: KARTU RINGKASAN ---
        st.subheader("1. Kondisi Terkini per Stasiun")
        st.markdown("Pantau status kualitas udara terakhir di setiap stasiun. Kartu ini otomatis menampilkan data paling mutakhir untuk pengambilan keputusan cepat di lapangan.")
        
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

        # --- BAGIAN 2: BAR CHART DISTRIBUSI ---
        st.subheader("2. Distribusi Kategori Kualitas Udara per Stasiun")
        st.markdown("Bandingkan seberapa sering suatu wilayah masuk dalam kategori aman atau berbahaya. Grafik ini membantu memetakan stasiun mana yang paling rawan polusi secara historis.")
        
        df_dist = df_historis.groupby(['stasiun', 'kategori']).size().unstack(fill_value=0)
        for cat in ['BAIK', 'SEDANG', 'TIDAK SEHAT']:
            if cat not in df_dist.columns:
                df_dist[cat] = 0
        df_dist = df_dist[['BAIK', 'SEDANG', 'TIDAK SEHAT']]
        
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 6))
        
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        
        x = np.arange(len(df_dist.index))
        width = 0.25
        
        rects1 = ax.bar(x - width, df_dist['BAIK'], width, label='BAIK', color='#2ca02c') 
        rects2 = ax.bar(x, df_dist['SEDANG'], width, label='SEDANG', color='#ff7f0e') 
        rects3 = ax.bar(x + width, df_dist['TIDAK SEHAT'], width, label='TIDAK SEHAT', color='#d62728') 
        
        ax.set_ylabel('Jumlah Hari')
        ax.set_title('Perbandingan Kategori Udara Antar Stasiun')
        ax.set_xticks(x)
        ax.set_xticklabels(df_dist.index)
        
        ax.legend(frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax.bar_label(rects1, padding=3, color='white')
        ax.bar_label(rects2, padding=3, color='white')
        ax.bar_label(rects3, padding=3, color='white')
        
        st.pyplot(fig)
        
        st.divider()

        # --- BAGIAN 3: FEATURE IMPORTANCE ---
        st.subheader("3. Faktor Penentu Keputusan Model (Feature Importance)")
        st.markdown("Identifikasi polutan mana yang paling mendominasi penentuan status ISPU. Fokuskan mitigasi dan pemantauan pada polutan dengan skor tertinggi (berada di posisi teratas).")
        
        try:
            importances = model.feature_importances_
            feature_names = ['PM10', 'PM2.5', 'SO2', 'CO', 'O3', 'NO2']
            
            df_imp = pd.DataFrame({'Polutan': feature_names, 'Skor': importances})
            df_imp = df_imp.sort_values(by='Skor', ascending=True)
            
            fig_imp, ax_imp = plt.subplots(figsize=(10, 5))
            
            fig_imp.patch.set_alpha(0.0)
            ax_imp.patch.set_alpha(0.0)
            
            colors = ['#e74c3c' if val == df_imp['Skor'].max() else '#5a6268' for val in df_imp['Skor']]
            
            bars = ax_imp.barh(df_imp['Polutan'], df_imp['Skor'], color=colors)
            
            ax_imp.set_xlabel('Skor Importance (0 - 1)')
            ax_imp.set_xlim(0, df_imp['Skor'].max() + 0.15) 
            
            for bar in bars:
                skor_aktual = bar.get_width()
                ax_imp.text(
                    skor_aktual + 0.01, 
                    bar.get_y() + bar.get_height() / 2, 
                    f"{skor_aktual:.4f}", 
                    va='center', 
                    fontweight='bold',
                    color='white'
                )
                
            ax_imp.spines['top'].set_visible(False)
            ax_imp.spines['right'].set_visible(False)
            
            st.pyplot(fig_imp)
            
            st.info("💡 **Insight:** Berdasarkan arsitektur matematis model, terbukti bahwa **PM2.5** mendominasi keputusan klasifikasi kualitas udara secara signifikan (mendekati 91%). Itulah alasan mengapa PM2.5 selalu dipantau paling ketat sebagai parameter pencemar kritis.")
            
        except Exception as e:
            st.error("Gagal memuat visualisasi Feature Importance.")

        st.divider()

        # --- BAGIAN 4: TREN POLUTAN (PINDAHAN DARI TAB LAMA) ---
        st.subheader("4. Tren Fluktuasi Polutan Harian")
        st.markdown("Lacak pola pergerakan polusi secara spesifik. Gunakan filter di bawah untuk mengidentifikasi apakah terdapat lonjakan polutan abnormal pada tanggal-tanggal tertentu.")
        
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        
        with col_filter1:
            daftar_periode = sorted(df_historis['periode_data'].unique())
            def format_periode(periode):
                p_str = str(periode)
                tahun, bulan = p_str[:4], p_str[4:]
                map_bulan = {"01": "Januari", "02": "Februari", "03": "Maret", "04": "April", "05": "Mei", "06": "Juni", "07": "Juli", "08": "Agustus", "09": "September", "10": "Oktober", "11": "November", "12": "Desember"}
                return f"{map_bulan.get(bulan, bulan)} {tahun}"
                
            periode_terpilih = st.selectbox("Pilih Periode:", daftar_periode, format_func=format_periode)
        
        df_filter_periode = df_historis[df_historis['periode_data'] == periode_terpilih]
        
        with col_filter2:
            daftar_stasiun = sorted(df_filter_periode['stasiun'].unique())
            stasiun_terpilih = st.selectbox("Pilih Stasiun:", daftar_stasiun)
            
        with col_filter3:
            opsi_polutan = {
                'PM2.5': 'pm_duakomalima', 'PM10': 'pm_sepuluh', 'SO2 (Sulfur Dioksida)': 'sulfur_dioksida',
                'CO (Karbon Monoksida)': 'karbon_monoksida', 'O3 (Ozon)': 'ozon', 'NO2 (Nitrogen Dioksida)': 'nitrogen_dioksida'
            }
            label_terpilih = st.selectbox("Pilih Parameter Polutan:", list(opsi_polutan.keys()))
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
                st.dataframe(df_filter[['tanggal', 'stasiun', 'periode_data', kolom_aktual]])
        else:
            st.warning("Data tidak ditemukan.")

    else:
        st.warning("Data CSV tidak ditemukan.")


# ==========================================
# TAB 2: PREDIKSI ISPU (PINDAHAN DARI TAB 3 LAMA)
# ==========================================
with tab2:
    st.subheader("Prediksi Kategori ISPU Berdasarkan Input")
    st.markdown("Perhatikan batas visual ini sebelum melakukan prediksi. Batang grafik yang mendekati atau melewati angka 50 menandakan parameter tersebut butuh perhatian khusus.")
    
    with st.container(border=True):
        st.markdown("**📝 Form Input Parameter Polutan**")
        
        col_in1, col_in2, col_in3 = st.columns(3)
        
        with col_in1:
            pm_sepuluh = st.number_input("🌫️ PM10", min_value=0.0, value=50.0, step=1.0, help="Partikel udara berukuran lebih kecil dari 10 mikron.")
            karbon_monoksida = st.number_input("🚗 CO", min_value=0.0, value=15.0, step=1.0, help="Gas beracun tanpa warna dan bau, biasanya dari knalpot kendaraan.")
            
        with col_in2:
            pm_duakomalima = st.number_input("💨 PM2.5", min_value=0.0, value=70.0, step=1.0, help="Partikel udara sangat halus berukuran 2.5 mikron. (Paling berbahaya)")
            ozon = st.number_input("☀️ O3 (Ozon)", min_value=0.0, value=20.0, step=1.0, help="Gas Ozon di permukaan tanah.")
            
        with col_in3:
            sulfur_dioksida = st.number_input("🏭 SO2", min_value=0.0, value=30.0, step=1.0, help="Gas berbau menyengat, biasanya dari pembakaran batu bara/pabrik.")
            nitrogen_dioksida = st.number_input("🏭 NO2", min_value=0.0, value=25.0, step=1.0, help="Gas polusi dari kendaraan dan pabrik yang bisa menyebabkan hujan asam.")

    st.divider()

    st.subheader("📈 Grafik Perbandingan terhadap Batas Aman (ISPU = 50)")
    st.markdown("Membandingkan nilai input saat ini dengan batas maksimal kategori udara 'BAIK'.")

    data_grafik = pd.DataFrame({
        'Nilai Input': [pm_sepuluh, pm_duakomalima, sulfur_dioksida, karbon_monoksida, ozon, nitrogen_dioksida],
        'Batas Maksimal Sehat': [50.0, 50.0, 50.0, 50.0, 50.0, 50.0]
    }, index=['PM10', 'PM2.5', 'SO2', 'CO', 'O3', 'NO2'])

    st.bar_chart(data_grafik['Nilai Input'], color="#008CBA")

    st.divider()

    # Suntikan CSS buat ubah warna tombol
    st.markdown("""
    <style>
    button[kind="primary"] {
        background-color: #008CBA !important;
        color: white !important;
        border: none !important;
    }
    button[kind="primary"]:hover {
        background-color: #005f73 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    if st.button("🔍 Analisis & Prediksi Kualitas Udara", use_container_width=True, type="primary"):
        data_input = np.array([[pm_sepuluh, pm_duakomalima, sulfur_dioksida, karbon_monoksida, ozon, nitrogen_dioksida]])
        data_scaled = scaler.transform(data_input)
        prediksi = model.predict(data_scaled)
        hasil_kategori = label_map[prediksi[0]]
        
        st.subheader("💡 Hasil Analisis")
        
        if hasil_kategori == 'BAIK':
            st.success("🟢 **Kategori: BAIK** \n\nTingkat kualitas udara sangat baik, tidak memberikan efek negatif terhadap manusia ataupun hewan.")
        elif hasil_kategori == 'SEDANG':
            st.warning("🟡 **Kategori: SEDANG** \n\nTingkat kualitas udara masih dapat diterima, namun kelompok sensitif disarankan untuk mengurangi aktivitas berat di luar ruangan.")
        else:
            st.error("🔴 **Kategori: TIDAK SEHAT** \n\nTingkat kualitas udara bersifat merugikan pada manusia atau kelompok hewan yang sensitif. Segera gunakan masker jika harus keluar ruangan!")
