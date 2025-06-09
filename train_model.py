import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import joblib

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
