import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

st.set_page_config(layout="wide")
st.title("📊 Stock Analytics Dashboard")


st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV / Excel", type=["csv", "xlsx"])


def generate_default_data():
    dates = pd.date_range(start="2022-01-01", end="2024-12-31")
    np.random.seed(42)
    n = len(dates)

    trend = np.linspace(4000, 9000, n)
    noise = np.random.normal(0, 80, n)

    df = pd.DataFrame({
        "Date": dates,
        "Close": trend + noise,
        "Volume": np.random.randint(1_000_000, 7_000_000, n),
        "PER": np.random.uniform(8, 30, n),
        "PBV": np.random.uniform(0.8, 6, n),
        "ROA": np.random.uniform(0.5, 6, n),
        "ROE": np.random.uniform(8, 30, n),
        "EPS": np.random.uniform(30, 400, n)
    })

    return df


if uploaded_file:
    data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    st.sidebar.success("Upload berhasil")
else:
    data = generate_default_data()
    st.sidebar.info("Menggunakan dataset default")

data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')


numeric_cols = data.select_dtypes(include=np.number).columns.tolist()

st.sidebar.header("📊 Feature Selection")

selected_features = st.sidebar.multiselect(
    "Pilih fitur",
    numeric_cols,
    default=["Close"] if "Close" in numeric_cols else numeric_cols[:1]
)

if "Close" not in selected_features:
    st.error("Close wajib dipilih")
    st.stop()


st.sidebar.header("🤖 Model Selection")

model_option = st.sidebar.selectbox(
    "Pilih Model",
    ["All Models (Benchmark)",
     "Linear Regression",
     "Random Forest",
     "Gradient Boosting",
     "XGBoost",
     "SVR",
     "LightGBM"]
)


st.sidebar.header("🔗 Korelasi Fitur")

col1, col2 = st.sidebar.columns(2)

feat1 = col1.selectbox("Fitur 1", numeric_cols)
feat2 = col2.selectbox("Fitur 2", numeric_cols, index=1)


st.subheader("📊 KPI")

latest = data['Close'].iloc[-1]
returns = data['Close'].pct_change().mean()*100
vol = data['Close'].pct_change().std()*100

c1, c2, c3 = st.columns(3)
c1.metric("Price", f"{latest:.2f}")
c2.metric("Return %", f"{returns:.2f}")
c3.metric("Volatility %", f"{vol:.2f}")


tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Model", "Forecast", "Insight"])


with tab1:
    st.line_chart(data.set_index("Date")["Close"])

    st.subheader("🔗 Korelasi Fitur")
    corr = data[feat1].corr(data[feat2])
    st.write(f"Korelasi {feat1} vs {feat2}: {corr:.4f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data[feat1],
        y=data[feat2],
        mode='markers'
    ))
    st.plotly_chart(fig)


df = data[selected_features]

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)

def create_dataset(data, time_step=10):
    X, y = [], []
    for i in range(len(data)-time_step-1):
        X.append(data[i:(i+time_step)])
        y.append(data[i+time_step, 0])
    return np.array(X), np.array(y)

X, y = create_dataset(scaled)
split = int(len(X)*0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)


# MODELS

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "XGBoost": XGBRegressor(),
    "SVR": SVR(),
    "LightGBM": LGBMRegressor()
}


with tab2:
    results = []

    if model_option == "All Models (Benchmark)":

        for name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, pred))
            mae = mean_absolute_error(y_test, pred)
            mape = np.mean(np.abs((y_test - pred)/y_test))*100

            results.append([name, rmse, mae, mape, model])

        df_results = pd.DataFrame(results, columns=["Model","RMSE","MAE","MAPE","Obj"])
        df_results = df_results.sort_values("RMSE")

        st.dataframe(df_results[["Model","RMSE","MAE","MAPE"]])

        best = df_results.iloc[0]
        best_model = best["Obj"]

        st.success(f"Best Model: {best['Model']}")

        st.info(f"""
Model {best['Model']} dipilih karena memiliki error terendah.
RMSE: {best['RMSE']:.4f}
MAE: {best['MAE']:.4f}
MAPE: {best['MAPE']:.2f}%
""")

    else:
        model = models[model_option]
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, pred))
        st.write(f"RMSE: {rmse:.4f}")

        best_model = model

with tab3:
    def forecast(model, data, n_days=90):
        temp = list(data[-10:].flatten())
        out = []

        for _ in range(n_days):
            x = np.array(temp[-len(selected_features)*10:]).reshape(1, -1)
            yhat = model.predict(x)
            temp.append(yhat[0])
            out.append(yhat[0])

        return scaler.inverse_transform(np.array(out).reshape(-1,1))

    future = forecast(best_model, scaled)

    st.line_chart(future)


with tab4:
    if hasattr(best_model, "feature_importances_"):
        st.bar_chart(best_model.feature_importances_)
    else:
        st.info("Model tidak mendukung feature importance")
