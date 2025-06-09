import streamlit as st
import pandas as pd
import joblib
import datetime

st.set_page_config(layout="wide")

st.title("🔮 Formulir Prediksi Permintaan Produk")
st.markdown("Isi formulir di bawah ini untuk mendapatkan prediksi level permintaan produk fashion.")
st.markdown("---")

# Fungsi untuk memuat model
@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        st.error(f"File model di {path} tidak ditemukan. Jalankan `train_model.py` terlebih dahulu.")
        return None

# Memuat model
knn_model = load_model('knn_model.joblib')
nb_model = load_model('nb_model.joblib')

# Opsi untuk input dari dataset asli
# (Ini memastikan input pengguna konsisten dengan data pelatihan)
df_options = pd.read_csv('inventory_data.csv')
category_options = df_options['Category'].unique()
region_options = df_options['Region'].unique()
weather_options = df_options['Weather Condition'].unique()


# Membuat form input
with st.form("prediction_form"):
    st.header("Masukkan Detail Produk dan Kondisi")

    # Layout kolom
    col1, col2 = st.columns(2)

    with col1:
        category = st.selectbox("Kategori Produk", options=category_options)
        region = st.selectbox("Wilayah Toko", options=region_options)
        inventory = st.number_input("Jumlah Inventaris Saat Ini", min_value=0, step=10)

    with col2:
        weather = st.selectbox("Kondisi Cuaca", options=weather_options)
        holiday = st.radio("Apakah Ada Hari Libur/Promosi?", options=[0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
        prediction_date = st.date_input("Tanggal Prediksi", value=datetime.date.today())

    submitted = st.form_submit_button("Dapatkan Prediksi")


if submitted and knn_model is not None and nb_model is not None:
    # 1. Mengumpulkan data input
    input_data = {
        'Category': [category],
        'Region': [region],
        'Inventory Level': [inventory],
        'Weather Condition': [weather],
        'Holiday/Promotion': [holiday],
        'Day': [prediction_date.day],
        'Month': [prediction_date.month],
        'Year': [prediction_date.year],
        'DayOfWeek': [prediction_date.weekday()]
    }
    input_df = pd.DataFrame(input_data)

    st.markdown("---")
    st.subheader("Hasil Prediksi")

    # 2. Melakukan prediksi
    try:
        prediction_knn = knn_model.predict(input_df)[0]
        prediction_proba_knn = knn_model.predict_proba(input_df)

        prediction_nb = nb_model.predict(input_df)[0]
        prediction_proba_nb = nb_model.predict_proba(input_df)

        # Menampilkan hasil dengan layout menarik
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.info("Prediksi dari Model KNN")
            if prediction_knn == 'Permintaan Tinggi':
                st.success(f"**Prediksi: {prediction_knn}** 📈")
            elif prediction_knn == 'Permintaan Sedang':
                st.warning(f"**Prediksi: {prediction_knn}** 📊")
            else:
                st.error(f"**Prediksi: {prediction_knn}** 📉")
            
            st.write("Probabilitas:")
            st.dataframe(pd.DataFrame(prediction_proba_knn, columns=knn_model.classes_, index=['Prob.']))


        with col_res2:
            st.info("Prediksi dari Model Naive Bayes")
            if prediction_nb == 'Permintaan Tinggi':
                st.success(f"**Prediksi: {prediction_nb}** 📈")
            elif prediction_nb == 'Permintaan Sedang':
                st.warning(f"**Prediksi: {prediction_nb}** 📊")
            else:
                st.error(f"**Prediksi: {prediction_nb}** 📉")

            st.write("Probabilitas:")
            st.dataframe(pd.DataFrame(prediction_proba_nb, columns=nb_model.classes_, index=['Prob.']))

    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
