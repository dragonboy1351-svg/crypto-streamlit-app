import streamlit as st
import pandas as pd
import numpy as np
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import cvxpy as cp

# ----------------------------
# App Title
# ----------------------------
st.title("Crypto Portfolio Optimisation")
st.markdown("### 📊 Sentiment-Enhanced Crypto Analysis Dashboard")

# ----------------------------
# Config
# ----------------------------
API_KEY = "ca28d0c8038e074b58ba188a33bdefad11bf7dbbfc739fe5942f8a3323ee075a"  
coins = ["BTC", "ETH", "XRP"]
analyzer = SentimentIntensityAnalyzer()

# ----------------------------
# Function: Fetch Price Data
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

# ----------------------------
# Function: Fetch News + Sentiment
# ----------------------------
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
# Function: Portfolio Optimisation
# ----------------------------
def optimize_portfolio(returns, cov_matrix, sentiment_score):
    n = len(returns)
    w = cp.Variable(n)
    risk_aversion = max(0.01, 1 - sentiment_score)
    objective = cp.Maximize(returns @ w - risk_aversion * cp.quad_form(w, cov_matrix))
    constraints = [cp.sum(w) == 1, w >= 0, w <= 0.5]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return pd.Series(w.value, index=returns.index)

# ----------------------------
# Main App
# ----------------------------
st.sidebar.header("Settings")
limit = st.sidebar.slider("Days of historical data", 30, 120, 90)

# Load Data
dfs = [get_daily_price_data(coin, API_KEY, limit) for coin in coins]
price_data = pd.concat(dfs, axis=1).dropna()

if not price_data.empty:
    st.subheader("📈 Historical Prices")
    st.line_chart(price_data)

    # Calculate Returns
    returns = price_data.pct_change().dropna()

    st.subheader("📉 Daily Returns")
    st.line_chart(returns)

    # Sentiment Analysis
    st.subheader("🧠 Sentiment Scores")
    sentiment_scores = get_sentiment_scores(coins, API_KEY)
    st.write(sentiment_scores)

    avg_sentiment = np.mean(list(sentiment_scores.values()))
    st.write(f"**Average Market Sentiment**: {avg_sentiment:.4f}")

    # Portfolio Optimisation
    st.subheader("📊 Portfolio Optimisation (Simulated)")

    if not returns.empty:
        returns_series = returns.mean()       # expected return per coin
        cov_matrix = returns.cov()            # real covariance matrix
        weights = optimize_portfolio(returns_series, cov_matrix, avg_sentiment)
        st.write("**Simulated Portfolio Weights:**")
        st.write(weights.round(4))
    else:
        st.warning("Not enough return data to optimize portfolio.")
else:
    st.error("Failed to load crypto price data.")
