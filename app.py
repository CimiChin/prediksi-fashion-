import streamlit as st

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
