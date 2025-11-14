import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

st.title("📈 Stock Price Forecasting Dashboard (ARIMA)")
st.write("Enter any stock ticker (Example: 7203.T for Toyota, 6758.T for Sony)")

# -----------------------
# User Input Section
# -----------------------
ticker = st.text_input("Enter Stock Ticker:", value="7203.T")
start_date = st.date_input("Start Date", dt.date(2024, 1, 1))
end_date = st.date_input("End Date", dt.date.today())
forecast_days = st.slider("Forecast days", 5, 60, 10)

if st.button("Generate Forecast"):
    # -----------------------
    # Download Data
    # -----------------------
    st.write(f"Downloading data for **{ticker}**...")
    data = yf.download(ticker, start=start_date, end=end_date)

    if data.empty:
        st.error("Invalid ticker or no data available.")
        st.stop()

    data.columns = data.columns.get_level_values(0)
    data = data.reset_index()
    data = data.set_index("Date")
    df = data[["Close"]]
    df["Returns"] = df["Close"].pct_change()
    df = df.dropna()

    # -----------------------
    # Stationarity Check
    # -----------------------
    def check_stationarity(series):
        result = adfuller(series.dropna())
        return result[1] < 0.05

    is_stationary = check_stationarity(df["Close"])

    st.subheader("ADF Test Result")
    if is_stationary:
        st.success("✔ The series is stationary")
    else:
        st.warning("✖ The series is NOT stationary — applying differencing")

    # Apply first difference
    df["Close_Diff"] = df["Close"].diff()
    df = df.dropna()

    # -----------------------
    # ARIMA Model
    # -----------------------
    st.write("Fitting ARIMA model (5,1,0)... please wait")

    model = ARIMA(df["Close"], order=(5, 1, 0))
    model_fit = model.fit()

    # Forecast
    forecast = model_fit.forecast(steps=forecast_days)
    forecast_dates = pd.date_range(start=df.index[-1], periods=forecast_days + 1, freq="B")[1:]

    # -----------------------
    # Plotting
    # -----------------------
    st.subheader(f"Actual vs Predicted Prices for {ticker}")

    plt.figure(figsize=(12, 5))
    plt.plot(df["Close"], label="Actual Prices")
    plt.plot(forecast_dates, forecast, label="Predicted Prices", linestyle="dashed")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(plt)

    # -----------------------
    # Show Forecast Table
    # -----------------------
    forecast_df = pd.DataFrame({"Date": forecast_dates, "Forecast": forecast})
    forecast_df = forecast_df.set_index("Date")

    st.subheader("Forecast Values")
    st.dataframe(forecast_df)

    # download forecast button
    st.download_button(
        label="Download Forecast CSV",
        data=forecast_df.to_csv().encode("utf-8"),
        file_name=f"{ticker}_forecast.csv",
        mime="text/csv",
    )
