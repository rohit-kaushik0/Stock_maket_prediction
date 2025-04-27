import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import plotly.graph_objects as go
from model import preprocess_data, build_lstm_model, train_model, predict_future_prices, calculate_rmse

# Page configuration
st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced and responsive CSS for sidebar and main UI
sidebar_style = """
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #232526 0%, #414345 100%);
        color: #fff;
    }
    .sidebar-title {
        color: #FFD700;
        font-size: 1.4em;
        font-weight: bold;
        margin-bottom: 0.5em;
    }
    .sidebar-section {
        background: rgba(255,255,255,0.07);
        border-radius: 10px;
        padding: 1em;
        margin-bottom: 1em;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .sidebar-highlight {
        color: #00FFCC;
        font-size: 1.2em;
        font-weight: bold;
        background: #222;
        border-radius: 8px;
        padding: 0.5em 0.8em;
        margin-bottom: 0.5em;
        display: inline-block;
    }
    .sidebar-link a {
        color: #FFD700 !important;
        text-decoration: underline;
    }
    .main-header {
        background: linear-gradient(90deg, #232526 0%, #00c6ff 100%);
        color: #fff;
        padding: 1.5em 1em 1em 1em;
        border-radius: 12px;
        margin-bottom: 1.5em;
        text-align: center;
        font-size: 2.2em;
        font-weight: bold;
        letter-spacing: 1px;
        box-shadow: 0 2px 16px rgba(0,0,0,0.08);
    }
    .info-card {
        background: #f7f7fa;
        border-radius: 10px;
        padding: 1.2em 1em;
        margin-bottom: 1.2em;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        color: #232526;
        font-size: 1.1em;
    }
    .info-card strong {
        color: #00b894;
    }
    .feature-badge {
        display: inline-block;
        background: #00c6ff;
        color: #fff;
        border-radius: 8px;
        padding: 0.2em 0.7em;
        margin: 0.2em 0.2em 0.2em 0;
        font-size: 0.95em;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .footer {
        margin-top:2em;
        margin-bottom:1em;
        text-align:center;
        color:#888;
        font-size:1.1em;
    }
    .footer a {
        color:#0984e3;
        text-decoration: none;
        font-weight: bold;
    }
    </style>
"""
st.markdown(sidebar_style, unsafe_allow_html=True)

ALL_STOCKS = [
    'AMZN', 'DPZ', 'BTC-USD', 'NFLX',
    'RELIANCE.NS', 'INFY.NS', 'TCS.NS', 'HDFCBANK.NS'
]

# Main header
st.markdown('<div class="main-header">📈 Stock Price Prediction Using Machine Learning</div>', unsafe_allow_html=True)

# Info cards row with badges
col1, col2 = st.columns(2)
with col1:
    st.markdown(
        '<div class="info-card">'
        '<strong>Welcome!</strong> This app allows you to visualize and predict stock prices for major global and Indian stocks using multiple machine learning models.<br>'
        'Choose your model and stock in the sidebar to get started.<br>'
        '<span class="feature-badge">AMZN</span>'
        '<span class="feature-badge">DPZ</span>'
        '<span class="feature-badge">BTC-USD</span>'
        '<span class="feature-badge">NFLX</span>'
        '<span class="feature-badge">RELIANCE.NS</span>'
        '<span class="feature-badge">INFY.NS</span>'
        '<span class="feature-badge">TCS.NS</span>'
        '<span class="feature-badge">HDFCBANK.NS</span>'
        '</div>',
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        '<div class="info-card">'
        '⭐ <strong>Features:</strong><br>'
        '- Multiple ML models (Random Forest, LSTM, Linear Regression, XGBoost)<br>'
        '- Interactive charts and data<br>'
        '- Real-time predictions<br>'
        '- Raw data preview<br>'
        '- Clean, modern UI'
        '</div>',
        unsafe_allow_html=True
    )

# Expander for project details and instructions
with st.expander("ℹ️ About this App & Instructions", expanded=False):
    st.markdown("""
    - **Select a stock and model** from the sidebar to view predictions.
    - **Adjust the prediction window** using the slider.
    - **Model Used** is shown above the chart.
    - **Scroll down** for historical data and raw data preview.
    - For best results, try different models and compare their performance!
    - <b>Note:</b> MSE is not shown as it is not comparable across stocks with different price scales.
    """, unsafe_allow_html=True)

# Load and clean data
data = pd.read_csv('stock_data.csv', parse_dates=['date'])
data.set_index('date', inplace=True)
data = data.dropna()
data = data.astype(float)

# Sidebar content
with st.sidebar:
    st.markdown('<div class="sidebar-title">📋 Project Information</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">Stock Price Prediction Using Machine Learning<br>This app predicts stock prices for the next 30 days using historical data and machine learning models.</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-title">👤 Developer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section"><span class="sidebar-highlight">Rohit Kaushik</span><br>Welcome to the Stock Price Prediction App! 🚀</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-title">⚙️ Settings</div>', unsafe_allow_html=True)
    with st.container():
        model_type = st.selectbox(
            "Select Training Model",
            options=["Random Forest", "LSTM", "Linear Regression", "XGBoost"]
        )
        target_column = st.selectbox(
            "Select the stock for prediction",
            options=ALL_STOCKS
        )
        feature_columns = [col for col in ALL_STOCKS if col != target_column]
        future_days = st.slider(
            "Select Future Prediction Days (1-30)",
            min_value=1, max_value=30, value=7
        )

    st.markdown('<div class="sidebar-title">🔗 Useful Links</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sidebar-section sidebar-link">
        <a href="https://github.com/rohit-kaushik0/Stock_maket_prediction" target="_blank">- GitHub Repository</a><br>
        <a href="https://www.linkedin.com/in/rohit-hrk-517809231/" target="_blank">- LinkedIn Profile</a>
        </div>
        """,
        unsafe_allow_html=True
    )

# Data preparation
X = data[feature_columns]
y = data[target_column]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

if model_type == "Random Forest":
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    last_known_date = data.index[-1]
    future_dates = [last_known_date + timedelta(days=i) for i in range(1, future_days + 1)]
    future_predictions = model.predict(X.tail(future_days))
elif model_type == "LSTM":
    X_lstm, y_lstm, scaler = preprocess_data(data, target_column)
    split_idx = int(len(X_lstm) * 0.8)
    X_train_lstm, X_val_lstm = X_lstm[:split_idx], X_lstm[split_idx:]
    y_train_lstm, y_val_lstm = y_lstm[:split_idx], y_lstm[split_idx:]
    lstm_model = train_model(X_train_lstm, y_train_lstm, X_val_lstm, y_val_lstm)
    last_60_days = data[target_column].values[-60:]
    last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, 1)).flatten()
    future_pred_scaled = predict_future_prices(lstm_model, last_60_days_scaled, days=future_days)
    future_predictions = scaler.inverse_transform(future_pred_scaled.reshape(-1, 1)).flatten()
    last_known_date = data.index[-1]
    future_dates = [last_known_date + timedelta(days=i) for i in range(1, future_days + 1)]
elif model_type == "Linear Regression":
    model = LinearRegression()
    model.fit(X_train, y_train)
    last_known_date = data.index[-1]
    future_dates = [last_known_date + timedelta(days=i) for i in range(1, future_days + 1)]
    future_predictions = model.predict(X.tail(future_days))
elif model_type == "XGBoost":
    model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    model.fit(X_train, y_train)
    last_known_date = data.index[-1]
    future_dates = [last_known_date + timedelta(days=i) for i in range(1, future_days + 1)]
    future_predictions = model.predict(X.tail(future_days))

# Show model used
st.markdown(f"**Model Used:** <span style='color:#00b894;font-weight:bold'>{model_type}</span>", unsafe_allow_html=True)

# Main content
st.markdown(f"### {target_column} Stock Price Prediction")
st.write(f"Forecasting the next **{future_days} days** for **{target_column}** using **{model_type}**.")

# Create figure
fig = go.Figure()
years_back = 3
start_date = last_known_date - timedelta(days=365 * years_back)
historical_data = data[target_column].loc[start_date:]
fig.add_trace(go.Scatter(
    x=historical_data.index,
    y=historical_data,
    mode='lines',
    name='Historical Data',
    line=dict(color='blue')
))
fig.add_trace(go.Scatter(
    x=future_dates,
    y=future_predictions,
    mode='lines+markers',
    name='Future Prediction',
    line=dict(color='red', dash='dash')
))
fig.update_layout(
    title=f"{target_column} Stock Price Prediction (Next {future_days} Days)",
    xaxis_title="Date",
    yaxis_title="Price",
    template="plotly_white"
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("📈 Advanced Chart Options", expanded=False):
    st.write("You can zoom, pan, and save the chart using the Plotly toolbar above.")

st.markdown("### 📊 Historical Data Overview")
st.line_chart(data[target_column].tail(365), use_container_width=True)

with st.expander("📂 Show Raw Data (last 10 rows)", expanded=False):
    st.dataframe(data.tail(10))

# Final footer
st.markdown(
    "<hr class='footer'>"
    "<div class='footer'>"
    "Made with ❤️ by <b style='color:#00FFCC;'>Rohit Kaushik</b> | "
    "<a href='https://github.com/rohit-kaushik0/Stock_maket_prediction' target='_blank'>GitHub</a> | "
    "<a href='https://www.linkedin.com/in/rohit-hrk-517809231/' target='_blank'>LinkedIn</a>"
    "</div>",
    unsafe_allow_html=True
)
