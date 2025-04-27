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
    layout="wide"
)

# Title and description
st.title("📈 Stock Price Prediction Using Machine Learning")
st.markdown("""
Welcome to the **Stock Price Prediction** app!  
This tool allows you to visualize historical stock prices and predict future trends using machine learning models.  
Select a stock from the sidebar to get started.
""")

# Load data
data = pd.read_csv('stock_data.csv', parse_dates=['Date'])
data.set_index('Date', inplace=True)

# Sidebar for user input
st.sidebar.header("📋 Project Information")
st.sidebar.markdown("""
**Stock Price Prediction Using Machine Learning**  
This app predicts stock prices for the next 30 days using historical data and machine learning models.  
""")

st.sidebar.header("👤 Developer")
st.sidebar.markdown("""
**Rohit Kaushik**  
Welcome to the Stock Price Prediction App!  
""")

st.sidebar.header("⚙️ Settings")
model_type = st.sidebar.selectbox(
    "Select Training Model",
    options=["Random Forest", "LSTM", "Linear Regression", "XGBoost"]
)
target_column = st.sidebar.selectbox(
    "Select the stock for prediction",
    options=['AMZN', 'DPZ', 'BTC', 'NFLX']
)
future_days = st.sidebar.slider(
    "Select Future Prediction Days (1-30)",
    min_value=1, max_value=30, value=7
)

# Data preparation
X = data[['DPZ', 'BTC', 'NFLX']]
y = data[target_column]

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

if model_type == "Random Forest":
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    val_predictions = model.predict(X_val)
    mse = mean_squared_error(y_val, val_predictions)
    # Predict future stock prices
    last_known_date = data.index[-1]
    future_dates = [last_known_date + timedelta(days=i) for i in range(1, future_days + 1)]
    future_predictions = model.predict(X.tail(future_days))
elif model_type == "LSTM":
    # LSTM expects only the target column, not other features
    X_lstm, y_lstm, scaler = preprocess_data(data, target_column)
    split_idx = int(len(X_lstm) * 0.8)
    X_train_lstm, X_val_lstm = X_lstm[:split_idx], X_lstm[split_idx:]
    y_train_lstm, y_val_lstm = y_lstm[:split_idx], y_lstm[split_idx:]
    lstm_model = train_model(X_train_lstm, y_train_lstm, X_val_lstm, y_val_lstm)
    val_pred_scaled = lstm_model.predict(X_val_lstm)
    val_pred = scaler.inverse_transform(val_pred_scaled)
    y_val_true = scaler.inverse_transform(y_val_lstm.reshape(-1, 1))
    mse = calculate_rmse(y_val_true, val_pred)
    # Predict future prices
    last_60_days = data[target_column].values[-60:]
    last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, 1)).flatten()
    future_pred_scaled = predict_future_prices(lstm_model, last_60_days_scaled, days=future_days)
    future_predictions = scaler.inverse_transform(future_pred_scaled.reshape(-1, 1)).flatten()
    last_known_date = data.index[-1]
    future_dates = [last_known_date + timedelta(days=i) for i in range(1, future_days + 1)]
elif model_type == "Linear Regression":
    model = LinearRegression()
    model.fit(X_train, y_train)
    val_predictions = model.predict(X_val)
    mse = mean_squared_error(y_val, val_predictions)
    last_known_date = data.index[-1]
    future_dates = [last_known_date + timedelta(days=i) for i in range(1, future_days + 1)]
    future_predictions = model.predict(X.tail(future_days))
elif model_type == "XGBoost":
    model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    model.fit(X_train, y_train)
    val_predictions = model.predict(X_val)
    mse = mean_squared_error(y_val, val_predictions)
    last_known_date = data.index[-1]
    future_dates = [last_known_date + timedelta(days=i) for i in range(1, future_days + 1)]
    future_predictions = model.predict(X.tail(future_days))

st.sidebar.header("📊 Model Performance")
if mse is not None:
    st.sidebar.write(f"📊 Validation MSE: {mse:.2f}")
else:
    st.sidebar.write("📊 Validation MSE: N/A")

st.sidebar.header("🔗 Useful Links")
st.sidebar.markdown("""
- [GitHub Repository](https://github.com/your-username/Stock-Price-Prediction-Using-Machine-Learning)
- [LinkedIn Profile](https://www.linkedin.com/in/rohit-hrk-517809231/ )
""")

# Main content
st.header(f"{target_column} Stock Price Prediction")
st.write(f"Forecasting the next {future_days} days")

# Create figure
fig = go.Figure()

# Plot historical data
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

# Plot future predictions
fig.add_trace(go.Scatter(
    x=future_dates,
    y=future_predictions,
    mode='lines+markers',
    name='Future Prediction',
    line=dict(color='red', dash='dash')
))

# Customize layout
fig.update_layout(
    title=f"{target_column} Stock Price Prediction (Next {future_days} Days)",
    xaxis_title="Date",
    yaxis_title="Price",
    template="plotly_white"
)

# Show plot
st.plotly_chart(fig, use_container_width=True)

# Additional visualizations
st.subheader("📊 Historical Data Overview")
st.line_chart(data[target_column].tail(365), use_container_width=True)

st.subheader("📂 Raw Data")
st.dataframe(data.tail(10))
