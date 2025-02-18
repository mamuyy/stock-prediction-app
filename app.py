import streamlit as st
from tensorflow.keras.models import load_model

# Coba load model LSTM
try:
    model = load_model("model_lstm.h5")
    st.success("✅ Model berhasil dimuat!")
except Exception as e:
    st.error(f"❌ Error saat memuat model: {e}")
    model = None

import streamlit as st
import pandas as pd  # ✅ Pastikan `pandas` diimpor!
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# 🔹 Load Model LSTM (Pastikan file `model_lstm.h5` ada di repo!)
try:
    model = load_model("model_lstm.h5")
except:
    model = None
    st.error("❌ Model LSTM tidak ditemukan! Pastikan `model_lstm.h5` sudah diupload.")

# 🔹 Fungsi Ambil Data Saham dari Yahoo Finance
@st.cache_data
def get_stock_data(symbol):
    try:
        data = yf.download(symbol, period="5y", interval="1d")  # Ambil 5 tahun terakhir, harian
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
    st.write(data.tail())  # 🔍 Menampilkan 5 data terakhir

    # 🔹 Tampilkan Grafik Harga Historis
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["Close"], mode="lines", name="Closing Price"))
    fig.update_layout(title=f"{symbol} Stock Price", xaxis_title="Date", yaxis_title="Price (IDR)")
    st.plotly_chart(fig)

    # 🔹 Normalisasi Data untuk Prediksi
    scaler = MinMaxScaler(feature_range=(0,1))
    prices_scaled = scaler.fit_transform(data[['Close']])

    # 🔹 Persiapan Data LSTM
    X_test = []
    for i in range(60, len(prices_scaled)):
        X_test.append(prices_scaled[i-60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # 🔹 Prediksi 30 Hari ke Depan
    if model:
        future_prices = []
        test_data = prices_scaled[-60:]
        for i in range(30):  # Prediksi 30 hari ke depan
            X_input = np.array(test_data[-60:]).reshape(1, 60, 1)
            predicted_price = model.predict(X_input)[0,0]
            future_prices.append(predicted_price)
            test_data = np.append(test_data, predicted_price).reshape(-1, 1)

        future_prices = scaler.inverse_transform(np.array(future_prices).reshape(-1, 1))

        # 🔹 Pastikan `pd` Sudah Dipanggil & Buat Rentang Tanggal
        future_dates = pd.date_range(start=data.index[-1], periods=31, freq='D')[1:]

        # 🔹 Grafik Prediksi
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=data.index, y=data["Close"], mode="lines", name="Historical Prices"))
        fig_pred.add_trace(go.Scatter(x=future_dates, y=future_prices.flatten(), mode="lines", name="Predicted Prices", line=dict(dash="dash", color="red")))
        fig_pred.update_layout(title=f"{symbol} Stock Price Prediction", xaxis_title="Date", yaxis_title="Price (IDR)")
        st.plotly_chart(fig_pred)
    else:
        st.warning("⚠️ Model belum dimuat, prediksi tidak tersedia.")
