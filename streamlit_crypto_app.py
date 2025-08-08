import streamlit as st
import pandas as pd
import numpy as np
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import google.generativeai as genai
from datetime import datetime, timedelta

# -----------------------------
# API KEYS (HARD-CODED)
# -----------------------------
CRYPTOCOMPARE_API_KEY = "4ba44372fbe4f9a4e338d7c72908b6c6b4e838b9aa09e4b34763e89a26417b7d"
GEMINI_API_KEY = "AIzaSyD8dbzMGmUYkuK2nXSO8zJsMyho1t6onfk"

# -----------------------------
# Configure Gemini
# -----------------------------
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-pro")

# -----------------------------
# Sentiment Analysis
# -----------------------------
def fetch_sentiment_data(symbols):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_records = []
    for symbol in symbols:
        url = f"https://min-api.cryptocompare.com/data/v2/news/?categories={symbol}&api_key={CRYPTOCOMPARE_API_KEY}"
        try:
            response = requests.get(url)
            articles = response.json().get("Data", [])
            for article in articles:
                text = article.get("title", "")
                date = datetime.utcfromtimestamp(article["published_on"]).date()
                score = analyzer.polarity_scores(text)["compound"]
                sentiment_records.append({"date": date, "symbol": symbol, "sentiment": score})
        except Exception:
            continue
    df = pd.DataFrame(sentiment_records)
    if not df.empty:
        df = df.groupby(["date", "symbol"]).mean().reset_index()
    return df

# -----------------------------
# Fetch price data
# -----------------------------
def fetch_price_data(symbols, limit=30):
    records = []
    for symbol in symbols:
        url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={symbol}&tsym=USD&limit={limit}&api_key={CRYPTOCOMPARE_API_KEY}"
        r = requests.get(url)
        data = r.json().get("Data", {}).get("Data", [])
        for entry in data:
            records.append({
                "date": datetime.utcfromtimestamp(entry["time"]).date(),
                "symbol": symbol,
                "close": entry["close"]
            })
    df = pd.DataFrame(records)
    return df

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Crypto Portfolio Optimisation", layout="wide")
st.title("🧠 Crypto Portfolio Optimisation")

symbols = st.multiselect("Select cryptocurrencies:", ["BTC", "ETH", "XRP"], default=["BTC", "ETH"])

if not symbols:
    st.warning("Please select at least one cryptocurrency.")
    st.stop()

with st.spinner("Fetching data..."):
    price_df = fetch_price_data(symbols)
    sentiment_df = fetch_sentiment_data(symbols)

# Ensure datetime format matches
price_df["date"] = pd.to_datetime(price_df["date"])
sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])

# Merge data
merged = pd.merge(price_df, sentiment_df, on=["date", "symbol"], how="left")
merged["sentiment"].fillna(0, inplace=True)

# Compute indicators
merged.sort_values(["symbol", "date"], inplace=True)
merged["daily_return"] = merged.groupby("symbol")["close"].pct_change()
merged["volatility"] = merged.groupby("symbol")["daily_return"].rolling(window=7).std().reset_index(0, drop=True)
merged["momentum"] = merged.groupby("symbol")["close"].pct_change(periods=7)

# Dashboard
for symbol in symbols:
    st.subheader(f"📊 {symbol} Metrics")

    coin_data = merged[merged["symbol"] == symbol].set_index("date")

    st.line_chart(coin_data[["close"]], height=200, use_container_width=True)
    st.line_chart(coin_data[["daily_return", "volatility", "momentum"]], height=250, use_container_width=True)
    st.line_chart(coin_data[["sentiment"]], height=150, use_container_width=True)

    # Gemini interpretation
    with st.expander(f"💬 Gemini Summary for {symbol}"):
        latest_sentiment = coin_data["sentiment"].dropna().iloc[-7:].mean()
        prompt = f"Summarise the recent 7-day sentiment trend for {symbol} which has an average score of {latest_sentiment:.2f}. What should an investor know?"
        gemini_response = gemini_model.generate_content(prompt)
        st.write(gemini_response.text)

st.success("Dashboard loaded successfully.")
