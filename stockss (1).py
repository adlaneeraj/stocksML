# streamlit_stock_trend_app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from pandas.tseries.offsets import BDay
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="Stock Trend Prediction", layout="wide")
st.title("📈 Stock Price & Trend Prediction Dashboard")

# ------------------------------
# COMPANY SELECTION
# ------------------------------
tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']
company = st.selectbox("Select a company to analyze:", tech_list)

# ------------------------------
# HISTORICAL DATA DOWNLOAD
# ------------------------------
end = datetime.now()
start = datetime(end.year - 5, end.month, end.day)

@st.cache_data(show_spinner=True)
def load_stock_data(ticker):
    df = yf.download(ticker, start=start, end=end)

    if df.empty:
        return pd.DataFrame()

    df = df.reset_index()

    # Flatten MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    # Ensure Close column exists and numeric
    if 'Close' not in df.columns:
        st.error(f"'Close' column not found for {ticker}")
        return pd.DataFrame()

    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close'])

    # Ensure Date column exists
    if 'Date' not in df.columns:
        df['Date'] = df.index

    return df

df = load_stock_data(company)

if df.empty or len(df) < 60:
    st.error(f"No sufficient historical data for {company}. Cannot make predictions.")
    st.stop()

# ------------------------------
# MOVING AVERAGES & TRENDS
# ------------------------------
df['MA_10'] = df['Close'].rolling(10).mean()
df['MA_50'] = df['Close'].rolling(50).mean()

latest_close = df['Close'].iloc[-1]
prev_close = df['Close'].iloc[-2]

# Trend based on MA10 vs MA50
if df['MA_10'].iloc[-1] > df['MA_50'].iloc[-1]:
    trend = "Uptrend 📈"
elif df['MA_10'].iloc[-1] < df['MA_50'].iloc[-1]:
    trend = "Downtrend 📉"
else:
    trend = "Sideways ⚖"

momentum = latest_close - prev_close
momentum_pct = (momentum / prev_close) * 100

st.subheader(f"📊 Current Trend for {company}")
st.write(f"Latest Close Price: **{latest_close:.2f}**")
st.write(f"Yesterday Close Price: **{prev_close:.2f}**")
st.write(f"Trend (MA10 vs MA50): **{trend}**")
st.write(f"Momentum (change from yesterday): **{momentum:.2f} ({momentum_pct:.2f}%)**")

# ------------------------------
# PLOT HISTORICAL PRICE + MA
# ------------------------------
st.subheader(f"Historical Price & Moving Averages for {company}")
fig, ax = plt.subplots(figsize=(12,5))
df.plot(x='Date', y=['Close','MA_10','MA_50'], ax=ax)
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.set_title(f"{company} Stock Price & Moving Averages")
st.pyplot(fig)

# ------------------------------
# LSTM PREPARATION
# ------------------------------
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df[['Close']].values)

train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size - 60:]

def create_sequences(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_data)
X_test, y_test = create_sequences(test_data)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1],1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1],1)

# ------------------------------
# TRAIN & PREDICT BUTTON
# ------------------------------
if st.button(f"Train LSTM & Predict Next 30 Days for {company}"):
    st.info("Training LSTM model... this may take a few minutes.")

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(60,1)))
    model.add(LSTM(50))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")

    # Train model
    model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)

    # Predict test data
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    train_df = df[:train_size]
    valid_df = df[train_size:].copy()
    valid_df['Predicted'] = predictions

    # ------------------------------
    # PLOT ACTUAL VS PREDICTED
    # ------------------------------
    st.subheader(f"Train vs Actual vs Predicted for {company}")
    fig2, ax2 = plt.subplots(figsize=(12,5))
    ax2.plot(train_df['Date'], train_df['Close'], label='Train')
    ax2.plot(valid_df['Date'], valid_df['Close'], label='Actual')
    ax2.plot(valid_df['Date'], valid_df['Predicted'], label='Predicted')
    ax2.set_title(f"{company} Stock Price Prediction")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Close Price")
    ax2.legend()
    st.pyplot(fig2)

    # ------------------------------
    # PREDICT NEXT 30 TRADING DAYS
    # ------------------------------
    last_60_days = scaled_data[-60:]
    future_input = last_60_days.reshape(1,60,1)
    future_prices = []

    for _ in range(30):
        pred = model.predict(future_input)
        future_prices.append(pred[0][0])
        pred_reshaped = pred.reshape(1,1,1)
        future_input = np.concatenate((future_input[:,1:,:], pred_reshaped), axis=1)

    future_prices = scaler.inverse_transform(np.array(future_prices).reshape(-1,1))

    # Generate real trading dates
    last_date = df['Date'].iloc[-1]
    future_dates = [last_date + BDay(i) for i in range(1,31)]

    future_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted_Close": future_prices.flatten()
    })

    st.subheader("🔮 Next 30 Trading Days Predictions")
    st.dataframe(future_df)

    # Summary print
    st.write("First 5 predicted prices:")
    for i in range(5):
        st.write(f"{future_df['Date'].iloc[i].date()}: {future_df['Predicted_Close'].iloc[i]:.2f}")

    st.write(f"Predicted Price Range (Next 30 days): {future_df['Predicted_Close'].min():.2f} - {future_df['Predicted_Close'].max():.2f}")
    st.write(f"Predicted Average Price: {future_df['Predicted_Close'].mean():.2f}")

    # Plot future prediction trend
    st.subheader("📈 Predicted Next 30 Trading Days Trend")
    fig3, ax3 = plt.subplots(figsize=(12,5))
    ax3.plot(future_df['Date'], future_df['Predicted_Close'], marker='o')
    ax3.set_title(f"{company} Next 30 Trading Days Prediction")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Predicted Close Price")
    ax3.grid(True)
    st.pyplot(fig3)
