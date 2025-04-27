# 📈 Stock Price Prediction Using Machine Learning

**Live DEMO**  
 [Link](https://hrkstockmarketpridictiction.streamlit.app/)

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [Setup and Installation](#setup-and-installation)
5. [Project Structure](#project-structure)
6. [How to Use](#how-to-use)
7. [Dataset](#dataset)
8. [Future Scope](#future-scope)
9. [Developer](#developer)
10. [License](#license)

---

## Overview

This project predicts stock prices for major global and Indian stocks using advanced machine learning models. It features an interactive Streamlit web app for visualization and forecasting, supporting multiple models and a modern UI.

---

## Features

- Predict stock prices for the next 30 days.
- Supports multiple stocks: **AMZN, DPZ, BTC-USD, NFLX, RELIANCE.NS, INFY.NS, TCS.NS, HDFCBANK.NS**
- Choose from multiple ML models: Random Forest, LSTM, Linear Regression, XGBoost.
- Interactive charts and real-time prediction visualization.
- Clean, modern, and responsive UI.
- Raw data preview and advanced chart options.

---

## Technologies Used

- **Python** (backend and ML)
- **Streamlit** (web app)
- **Pandas, NumPy** (data processing)
- **Scikit-learn** (ML utilities)
- **TensorFlow/Keras** (LSTM model)
- **Matplotlib, Plotly** (visualization)
- **XGBoost** (advanced regression)

---

## Setup and Installation

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/rohit-kaushik0/Stock_maket_prediction.git
    cd Stock-Price-Prediction-Using-Machine-Learning
    ```

2. **Create a Virtual Environment:**
    ```bash
    python -m venv env
    # On Windows:
    env\Scripts\activate
    # On Mac/Linux:
    source env/bin/activate
    ```

3. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Streamlit Application:**
    ```bash
    streamlit run main.py
    ```

---

## Project Structure

```plaintext
Stock-Price-Prediction-Using-Machine-Learning/
│
├── stock_data.csv               # Dataset used for training and prediction
├── model.py                     # Model training and utility functions
├── main.py                      # Streamlit app script
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
└── .gitignore                   # Git ignore rules
```

---

## How to Use

- Select a stock and model from the sidebar.
- Adjust the prediction window (1-30 days).
- View interactive charts and raw data.
- Try different models and compare results.
- Use advanced chart options for zooming and saving.

---

## Dataset

- The dataset (`stock_data.csv`) contains daily prices for multiple stocks and cryptocurrencies.
- Columns: `date`, `AMZN`, `DPZ`, `BTC-USD`, `NFLX`, `RELIANCE.NS`, `INFY.NS`, `TCS.NS`, `HDFCBANK.NS`

---

## Future Scope

- Add more stocks and data sources.
- Integrate technical indicators and news sentiment.
- Deploy as a public web service with user authentication.
- Add model explainability and comparison dashboards.

---

## Developer

**Rohit Kaushik**  
 [LinkedIn](https://www.linkedin.com/in/rohit-hrk-517809231/) | [Portfolio](https://rohit-kaushik0.github.io/MyPorfolio/)

---

## License

This project is licensed under the MIT License.

---

