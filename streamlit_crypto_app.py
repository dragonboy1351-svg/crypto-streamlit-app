import streamlit as st
import pandas as pd
import numpy as np
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ----------------------------
# App Title
# ----------------------------
st.title("🔐 Crypto Portfolio Optimisation")
st.markdown("### 📊 Sentiment-Enhanced Crypto Analysis Dashboard")

# ----------------------------
# Config
# ----------------------------
API_KEY = "ca28d0c8038e074b58ba188a33bdefad11bf7dbbfc739fe5942f8a3323ee075a " 
coins = ["BTC", "ETH", "XRP"]
analyzer = SentimentIntensityAnalyzer()

# ----------------------------
# Functions
# ----------------------------

def get_daily_price_data(symbol: str, api_key: str, limit: int = 90):
    url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={symbol}&tsym=USD&limit={limit}&api_key={api_key}"
    response = requests.get(url)
    data = response.json()
    if data["Response"] == "Success":
        df = pd.DataFrame(data["Data"]["Data"])
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        df = df[["close"]]
        df.rename(columns={"close": symbol}, inplace=True)
        return df
    else:
        return pd.DataFrame()

def get_sentiment_scores(coins, api_key):
    url = f"https://min-api.cryptocompare.com/data/v2/news/?lang=EN&api_key={api_key}"
    response = requests.get(url)
    news_data = response.json()
    sentiment_scores = {coin: [] for coin in coins}

    for article in news_data["Data"]:
        title = article["title"]
        score = analyzer.polarity_scores(title)["compound"]
        for coin in coins:
            if coin in article["categories"]:
                sentiment_scores[coin].append(score)

    avg_scores = {coin: np.mean(scores) if scores else 0 for coin, scores in sentiment_scores.items()}
    return avg_scores

# ----------------------------
# Sidebar Settings
# ----------------------------
st.sidebar.header("Settings")
limit = st.sidebar.slider("Days of historical data", 30, 120, 90)

# ----------------------------
# Load Price Data
# ----------------------------
dfs = [get_daily_price_data(coin, API_KEY, limit) for coin in coins]
price_data = pd.concat(dfs, axis=1).dropna()

# ----------------------------
# Main App
# ----------------------------
if not price_data.empty:
    st.subheader("📈 Historical Prices (USD)")
    st.line_chart(price_data)

    # Calculate Daily Returns
    returns = price_data.pct_change().dropna()
    st.subheader("📉 Daily Returns")
    st.line_chart(returns)

    # Volatility (Rolling Std Dev)
    st.subheader("📊 Volatility (Rolling 7-Day Std Dev)")
    volatility = returns.rolling(window=7).std()
    st.line_chart(volatility)

    # Momentum (Rolling Return)
    st.subheader("🚀 Momentum (Rolling 7-Day Return)")
    momentum = price_data.pct_change(periods=7)
    st.line_chart(momentum)

    # Sentiment Scores
    st.subheader("🧠 Sentiment Scores from News Headlines")
    sentiment_scores = get_sentiment_scores(coins, API_KEY)
    st.write(sentiment_scores)

    # Market-Wide Sentiment
    avg_sentiment = np.mean(list(sentiment_scores.values()))
    st.write(f"**Average Market Sentiment**: {avg_sentiment:.4f}")

else:
    st.error("❌ Failed to load crypto price data.")
