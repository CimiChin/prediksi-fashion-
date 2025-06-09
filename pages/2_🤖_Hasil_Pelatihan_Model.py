import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.title("🤖 Hasil dan Performa Pelatihan Model")
st.markdown("Halaman ini menunjukkan performa dari model KNN dan Naive Bayes setelah dilatih menggunakan data uji.")
st.markdown("---")

# Fungsi untuk memuat model dan data uji
@st.cache_data
def load_artifacts():
    try:
        knn_model = joblib.load('knn_model.joblib')
        nb_model = joblib.load('nb_model.joblib')
        X_test = pd.read_csv('X_test.csv')
        y_test = pd.read_csv('y_test.csv').squeeze() # squeeze untuk mengubahnya menjadi Series
        return knn_model, nb_model, X_test, y_test
    except FileNotFoundError:
        st.error("File model atau data uji tidak ditemukan. Jalankan `train_model.py` terlebih dahulu.")
        return None, None, None, None

knn_model, nb_model, X_test, y_test = load_artifacts()

if knn_model is not None:
    # Prediksi menggunakan model yang sudah dilatih
    y_pred_knn = knn_model.predict(X_test)
    y_pred_nb = nb_model.predict(X_test)

    # Menghitung metrik evaluasi
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    accuracy_nb = accuracy_score(y_test, y_pred_nb)
    report_knn = classification_report(y_test, y_pred_knn, output_dict=True)
    report_nb = classification_report(y_test, y_pred_nb, output_dict=True)
    cm_knn = confusion_matrix(y_test, y_pred_knn)
    cm_nb = confusion_matrix(y_test, y_pred_nb)

    st.subheader("Perbandingan Performa Model")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Akurasi Model KNN", value=f"{accuracy_knn:.2%}")
    with col2:
        st.metric(label="Akurasi Model Naive Bayes", value=f"{accuracy_nb:.2%}")

    st.markdown("---")

    tab1, tab2 = st.tabs(["K-Nearest Neighbors (KNN)", "Naive Bayes"])

    with tab1:
        st.header("Laporan Performa KNN")
        st.text("Classification Report:")
        st.dataframe(pd.DataFrame(report_knn).transpose())

        st.text("Confusion Matrix:")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=knn_model.classes_, yticklabels=knn_model.classes_)
        plt.xlabel('Prediksi')
        plt.ylabel('Aktual')
        st.pyplot(fig)

    with tab2:
        st.header("Laporan Performa Naive Bayes")
        st.text("Classification Report:")
        st.dataframe(pd.DataFrame(report_nb).transpose())

        st.text("Confusion Matrix:")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Greens', ax=ax,
                    xticklabels=nb_model.classes_, yticklabels=nb_model.classes_)
        plt.xlabel('Prediksi')
        plt.ylabel('Aktual')
        st.pyplot(fig)
