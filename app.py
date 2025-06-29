import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import joblib

st.set_page_config(
    page_title="Prediksi Permintaan Fashion",
    page_icon="👕",
    layout="wide"
)

st.title("👕 Selamat Datang di Dasbor Prediksi Permintaan Produk Fashion")
st.markdown("---")

st.markdown("""
Dasbor ini dirancang untuk membantu Anda memahami dan memprediksi permintaan produk fashion berdasarkan data historis.

**Apa yang bisa Anda lakukan di sini?**

1.  **📊 Analisis Data Eksplorasi (EDA)**: Lihat dataset, visualisasi tren penjualan, dan pahami karakteristik data.
2.  **🤖 Hasil Pelatihan Model**: Tinjau seberapa baik performa model Machine Learning (KNN & Naive Bayes) yang telah kami latih.
3.  **🔮 Prediksi Permintaan**: Gunakan formulir interaktif untuk memasukkan data baru dan dapatkan prediksi permintaan secara *real-time*.

Silakan pilih halaman yang ingin Anda kunjungi dari **sidebar di sebelah kiri**.
""")

st.info("Navigasi ke halaman lain menggunakan menu di sidebar.", icon="👈")

st.markdown("---")
st.header("Tentang Proyek")
st.markdown("""
- **Dataset**: Menggunakan data inventaris toko ritel dari Kaggle.
- **Model**: Prediksi dilakukan menggunakan **K-Nearest Neighbors (KNN)** dan **Gaussian Naive Bayes**.
- **Tools**: Dibangun dengan **Streamlit** dan **Scikit-learn**.
""")

print("Memulai proses pelatihan model...")

# 1. Muat Dataset
try:
    df = pd.read_csv('inventory_data.csv')
    print("Dataset berhasil dimuat.")
except FileNotFoundError:
    print("Error: File 'inventory_data.csv' tidak ditemukan. Pastikan file ada di direktori yang sama.")
    exit()


# 2. Pra-pemrosesan Data
# Mengubah kolom tanggal menjadi tipe datetime
df['Date'] = pd.to_datetime(df['Date'])

# Ekstraksi fitur dari tanggal
df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['DayOfWeek'] = df['Date'].dt.dayofweek # Senin=0, Minggu=6

# Membuat Target Variabel Kategorikal (Klasifikasi)
# Kita akan mengkategorikan 'Units Sold' menjadi 3 level permintaan
def create_demand_level(units_sold):
    if units_sold <= 20:
        return 'Permintaan Rendah'
    elif units_sold <= 50:
        return 'Permintaan Sedang'
    else:
        return 'Permintaan Tinggi'

df['Demand_Level'] = df['Units Sold'].apply(create_demand_level)
print("Variabel target 'Demand_Level' berhasil dibuat.")


# 3. Mendefinisikan Fitur (X) dan Target (y)
# Fitur yang akan digunakan untuk prediksi
features = [
    'Category', 'Region', 'Inventory Level', 'Weather Condition',
    'Holiday/Promotion', 'Day', 'Month', 'Year', 'DayOfWeek'
]
target = 'Demand_Level'

X = df[features]
y = df[target]

# Memisahkan fitur numerik dan kategorikal
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns


# 4. Membuat Pipeline Pra-pemrosesan
# OneHotEncoder akan mengubah data kategori menjadi angka
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

print("Pipeline pra-pemrosesan siap.")

# 5. Membagi Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# 6. Melatih Model K-Nearest Neighbors (KNN)
print("Melatih model KNN...")
knn_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', KNeighborsClassifier(n_neighbors=5))])

knn_pipeline.fit(X_train, y_train)
print("Model KNN berhasil dilatih.")


# 7. Melatih Model Naive Bayes
print("Melatih model Naive Bayes...")
nb_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', GaussianNB())])

nb_pipeline.fit(X_train, y_train)
print("Model Naive Bayes berhasil dilatih.")


# 8. Menyimpan Model dan Data Uji
# Model yang sudah dilatih disimpan agar bisa digunakan di aplikasi Streamlit
joblib.dump(knn_pipeline, 'knn_model.joblib')
joblib.dump(nb_pipeline, 'nb_model.joblib')
# Simpan juga data test untuk ditampilkan di halaman hasil model
X_test.to_csv('X_test.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print("\nPelatihan selesai! Model KNN, Naive Bayes, dan data uji telah disimpan.")
