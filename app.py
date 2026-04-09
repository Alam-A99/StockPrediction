import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go

# ML Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

# Boosting
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# DL Models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from datetime import date

# =========================
# TITLE
# =========================
st.title("📊 Stock Prediction: ML & DL Models")

# =========================
# SIDEBAR
# =========================
st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload Excel / CSV",
    type=["xlsx", "csv"]
)

st.sidebar.header("🤖 Pilih Model")

model_type = st.sidebar.selectbox(
    "Kategori Model",
    ["Machine Learning", "Deep Learning"]
)

if model_type == "Machine Learning":
    model_choice = st.sidebar.selectbox(
        "Pilih Model ML",
        ["Linear Regression", "Random Forest", "Gradient Boosting", "XGBoost", "SVR", "LightGBM"]
    )
else:
    model_choice = st.sidebar.selectbox(
        "Pilih Model DL",
        ["LSTM", "GRU"]
    )

n_days = st.sidebar.slider("Hari Prediksi", 30, 365)

# =========================
# LOAD DATA
# =========================
if uploaded_file is None:
    st.warning("Silakan upload dataset terlebih dahulu!")
    st.stop()

try:
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    if 'Date' not in data.columns or 'Close' not in data.columns:
        st.error("Dataset harus memiliki kolom: Date dan Close")
        st.stop()

    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date')

    st.success("Dataset berhasil diupload!")

except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# =========================
# PREVIEW
# =========================
st.subheader("Preview Data")
st.write(data.head())

# =========================
# VISUALISASI
# =========================
st.subheader("Grafik Harga")
st.line_chart(data.set_index('Date')['Close'])

# =========================
# PREPROCESSING
# =========================
df = data[['Close']].copy()

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# =========================
# CREATE DATASET
# =========================
def create_dataset(data, time_step=10):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 10
X, y = create_dataset(scaled_data, time_step)

# =========================
# SPLIT DATA
# =========================
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# =========================
# MACHINE LEARNING
# =========================
if model_type == "Machine Learning":

    X_train_ml = X_train.reshape(X_train.shape[0], -1)
    X_test_ml = X_test.reshape(X_test.shape[0], -1)

    if model_choice == "Linear Regression":
        model = LinearRegression()

    elif model_choice == "Random Forest":
        model = RandomForestRegressor()

    elif model_choice == "Gradient Boosting":
        model = GradientBoostingRegressor()

    elif model_choice == "XGBoost":
        model = XGBRegressor()

    elif model_choice == "SVR":
        model = SVR()

    elif model_choice == "LightGBM":
        model = LGBMRegressor()

    # TRAIN
    model.fit(X_train_ml, y_train)

    # PREDICT
    y_pred = model.predict(X_test_ml)

# =========================
# DEEP LEARNING
# =========================
else:

    X_train_dl = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_dl = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = Sequential()

    if model_choice == "LSTM":
        model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(50))

    else:
        model.add(GRU(50, return_sequences=True, input_shape=(time_step, 1)))
        model.add(Dropout(0.2))
        model.add(GRU(50))

    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    model.fit(X_train_dl, y_train, epochs=10, batch_size=32, verbose=1)

    y_pred = model.predict(X_test_dl)

# =========================
# EVALUASI
# =========================
mse = mean_squared_error(y_test, y_pred)
st.subheader("Evaluasi Model")
st.write(f"MSE: {mse:.6f}")

# =========================
# INVERSE SCALE
# =========================
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))

# =========================
# PLOT HASIL
# =========================
st.subheader("Perbandingan Aktual vs Prediksi")

fig = go.Figure()
fig.add_trace(go.Scatter(y=y_test_inv.flatten(), name="Actual"))
fig.add_trace(go.Scatter(y=y_pred_inv.flatten(), name="Predicted"))

st.plotly_chart(fig)

# =========================
# FUTURE PREDICTION
# =========================
def predict_future(model, data, n_days):
    temp_input = list(data[-time_step:].flatten())
    output = []

    for _ in range(n_days):
        x_input = np.array(temp_input[-time_step:]).reshape(1, -1)

        if model_type == "Deep Learning":
            x_input = x_input.reshape(1, time_step, 1)

        yhat = model.predict(x_input)
        temp_input.append(yhat[0])
        output.append(yhat[0])

    return scaler.inverse_transform(np.array(output).reshape(-1, 1))

future = predict_future(model, scaled_data, n_days)

# =========================
# PLOT FUTURE
# =========================
st.subheader("Prediksi Masa Depan")

fig2 = go.Figure()
fig2.add_trace(go.Scatter(y=df['Close'], name="History"))
fig2.add_trace(go.Scatter(
    x=list(range(len(df), len(df) + n_days)),
    y=future.flatten(),
    name="Future"
))

st.plotly_chart(fig2)

# =========================
# DOWNLOAD
# =========================
future_df = pd.DataFrame({"Prediksi": future.flatten()})

st.download_button(
    "Download Hasil Prediksi",
    future_df.to_csv(index=False),
    "prediksi.csv"
)
