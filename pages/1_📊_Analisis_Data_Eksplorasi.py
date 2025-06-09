import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")

st.title("📊 Analisis Data Eksplorasi (EDA)")
st.markdown("Halaman ini menampilkan analisis dari dataset inventaris. Anda bisa melihat data mentah dan beberapa visualisasi kunci.")
st.markdown("---")

# Fungsi untuk memuat data dengan caching agar lebih cepat
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('inventory_data.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except FileNotFoundError:
        st.error("File 'inventory_data.csv' tidak ditemukan. Pastikan file tersebut ada di direktori utama proyek.")
        return None

df = load_data()

if df is not None:
    # Tampilkan data mentah
    st.subheader("Tampilan Dataset")
    st.dataframe(df.head())

    # Tampilkan statistik deskriptif
    st.subheader("Statistik Deskriptif")
    st.write(df.describe())
    st.markdown("---")

    st.subheader("Visualisasi Data")

    # Kolom untuk layout
    col1, col2 = st.columns(2)

    with col1:
        # 1. Tren Penjualan Harian
        st.write("**Tren Penjualan Harian**")
        daily_sales = df.groupby('Date')['Units Sold'].sum().reset_index()
        fig_daily_sales = px.line(daily_sales, x='Date', y='Units Sold', title='Total Unit Terjual per Hari')
        st.plotly_chart(fig_daily_sales, use_container_width=True)

    with col2:
        # 2. Penjualan per Kategori Produk
        st.write("**Total Penjualan per Kategori Produk**")
        sales_by_category = df.groupby('Category')['Units Sold'].sum().sort_values(ascending=False).reset_index()
        fig_cat_sales = px.bar(sales_by_category, x='Category', y='Units Sold', title='Unit Terjual Berdasarkan Kategori', color='Category')
        st.plotly_chart(fig_cat_sales, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # 3. Penjualan per Wilayah
        st.write("**Total Penjualan per Wilayah**")
        sales_by_region = df.groupby('Region')['Units Sold'].sum().sort_values(ascending=False).reset_index()
        fig_region_sales = px.pie(sales_by_region, names='Region', values='Units Sold', title='Distribusi Penjualan per Wilayah', hole=0.3)
        st.plotly_chart(fig_region_sales, use_container_width=True)

    with col4:
        # 4. Pengaruh Cuaca pada Penjualan
        st.write("**Pengaruh Kondisi Cuaca pada Penjualan**")
        sales_by_weather = df.groupby('Weather Condition')['Units Sold'].mean().sort_values(ascending=False).reset_index()
        fig_weather_sales = px.bar(sales_by_weather, x='Weather Condition', y='Units Sold', title='Rata-rata Penjualan Berdasarkan Cuaca', color='Weather Condition')
        st.plotly_chart(fig_weather_sales, use_container_width=True)
