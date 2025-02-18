import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

# 🔹 Fungsi Ambil Data Saham dari Yahoo Finance
@st.cache_data
def get_stock_data(symbol):
    try:
        data = yf.download(symbol, start="2019-01-01", end="2024-01-01")
        if data.empty:
            st.error("Data saham tidak tersedia. Coba simbol saham lain.")
            return None
        return data
    except Exception as e:
        st.error(f"Gagal mengambil data saham: {e}")
        return None

# 🔹 Streamlit App
st.title("📈 AI Stock Prediction App")
st.write("Selamat datang di aplikasi prediksi saham berbasis AI!")

# 🔹 Input Simbol Saham
symbol = st.text_input("Masukkan simbol saham (contoh: TSLA, AAPL, BTC-USD, BBRI.JK)", "BBRI.JK")

# 🔹 Contoh Saham Indonesia
st.write("📌 Contoh simbol saham Indonesia: **BBRI.JK, BBCA.JK, TLKM.JK, ANTM.JK**")

# 🔹 Ambil Data Saham
data = get_stock_data(symbol)

if data is not None:
    st.write("📊 **Data Saham Terbaru:**")
    st.write(data.tail())  # 🔍 Menampilkan 5 data terakhir untuk debugging

    # 🔹 Tampilkan Grafik Harga Historis
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["Close"], mode="lines", name="Closing Price"))
    fig.update_layout(title=f"{symbol} Stock Price", xaxis_title="Date", yaxis_title="Price (IDR)")
    st.plotly_chart(fig)
