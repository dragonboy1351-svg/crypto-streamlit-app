import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import cvxpy as cp

# ----------------------------
# Functions
# ----------------------------

def get_daily_price_data(symbol: str, api_key: str, limit: int = 2000, currency: str = "USD") -> pd.DataFrame:
    url = f"https://data-api.coindesk.com/index/cc/v1/historical/days?market=cadli&instrument={symbol}-{currency}&limit={limit}&aggregate=1&fill=true&apply_mapping=true"
    headers = {"authorization": f"Apikey {api_key}"}
    response = requests.get(url, headers=headers, timeout=30)
    data = response.json()

    if "Data" not in data:
        return pd.DataFrame()

    df = pd.DataFrame(data["Data"])
    df["date"] = pd.to_datetime(df["TIMESTAMP"], unit="s")
    df = df.rename(columns={
        "OPEN": "open", "HIGH": "high", "LOW": "low", "CLOSE": "close",
        "VOLUME": "btc_volume", "QUOTE_VOLUME": "usd_volume"
    })
    df["symbol"] = symbol
    return df[["symbol", "date", "open", "high", "low", "close", "usd_volume", "btc_volume"]]

def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["symbol", "date"]).copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["volatility_7d"] = df["log_return"].rolling(7).std() * np.sqrt(365)
    df["return_7d"] = df["close"].pct_change(periods=7)
    return df

def get_crypto_news(api_key: str, limit: int = 100) -> pd.DataFrame:
    url = f"https://min-api.cryptocompare.com/data/v2/news/?lang=EN&api_key={api_key}&limit={limit}"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if "Data" not in data:
            return pd.DataFrame()

        rows = []
        for article in data["Data"]:
            rows.append({
                "published_on": pd.to_datetime(article.get("published_on", 0), unit="s"),
                "title": article.get("title", ""),
                "body": article.get("body", ""),
                "source": article.get("source", ""),
                "categories": article.get("categories", ""),
                "url": article.get("url", "")
            })
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Error downloading news: {e}")
        return pd.DataFrame()

def clean_news_text(df: pd.DataFrame) -> pd.DataFrame:
    def clean(text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    df["clean_title"] = df["title"].apply(clean)
    df["clean_body"] = df["body"].apply(clean)
    return df

def add_sentiment_scores(df: pd.DataFrame) -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer()
    df["sentiment_score"] = df["clean_title"].apply(lambda t: analyzer.polarity_scores(t)["compound"])
    return df

def optimize_portfolio(returns: pd.Series, cov_matrix: pd.DataFrame, sentiment_score: float) -> pd.Series:
    n = len(returns)
    w = cp.Variable(n)
    risk_aversion = max(0.01, 1 - sentiment_score)  # lower risk aversion if sentiment is positive
    objective = cp.Maximize(returns @ w - risk_aversion * cp.quad_form(w, cov_matrix))
    constraints = [cp.sum(w) == 1, w >= 0, w <= 0.5]  # no shorting, max 50% per asset
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return pd.Series(w.value, index=returns.index)

# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="Crypto Dashboard", layout="wide")
st.title("🔐 Crypto Portfolio Optimisation")
st.markdown("### 📊 Sentiment-Enhanced Crypto Analysis Dashboard")

# ✅ Hardcoded API Key (safe only for private/local use)
api_key = "ca28d0c8038e074b58ba188a33bdefad11bf7dbbfc739fe5942f8a3323ee075a"

symbol = st.selectbox("Choose Cryptocurrency Symbol", ["BTC", "ETH", "XRP", "LTC"])
limit_days = st.slider("Number of Days for Price Data", 30, 2000, 365)
news_limit = st.slider("Number of News Articles", 10, 200, 100)

if st.button("Run Analysis"):
    with st.spinner("Fetching price data..."):
        df_price = get_daily_price_data(symbol, api_key, limit=limit_days)
        if df_price.empty:
            st.error("No price data available.")
        else:
            df_price = add_price_features(df_price)
            st.subheader("📈 Price Data")
            st.dataframe(df_price.tail())
            st.line_chart(df_price.set_index("date")[["close", "return_7d"]])

    with st.spinner("Fetching crypto news..."):
        df_news = get_crypto_news(api_key, limit=news_limit)
        if df_news.empty:
            st.warning("No news data available.")
        else:
            df_news = clean_news_text(df_news)
            df_news = add_sentiment_scores(df_news)
            st.subheader("📰 News Sentiment")
            st.dataframe(df_news[["published_on", "title", "sentiment_score"]].head())
            st.line_chart(df_news.set_index("published_on")["sentiment_score"])

    if not df_price.empty and not df_news.empty:
        avg_sentiment = df_news["sentiment_score"].mean()
        latest_return = df_price["return_7d"].iloc[-1]

        st.subheader("📌 Summary")
        st.write(f"**7-Day Return:** {latest_return:.2%}")
        st.write(f"**Average Sentiment Score:** {avg_sentiment:.3f}")

        if avg_sentiment > 0.05:
            st.success("Recommendation: Consider increasing allocation due to positive sentiment.")
        elif avg_sentiment < -0.05:
            st.error("Recommendation: Consider reducing exposure due to negative sentiment.")
        else:
            st.info("Recommendation: Neutral sentiment – hold current position.")

        # Simulated portfolio optimization
        st.subheader("📊 Portfolio Optimization (Simulated)")
        returns = df_price["return_7d"].dropna().tail(10)
        cov_matrix = df_price[["log_return"]].dropna().cov()
        if not returns.empty:
            returns_series = pd.Series([returns.mean()] * 3, index=["BTC", "ETH", "XRP"])
            cov_matrix = pd.DataFrame(np.eye(3) * cov_matrix.values[0][0], index=returns_series.index, columns=returns_series.index)
            weights = optimize_portfolio(returns_series, cov_matrix, avg_sentiment)
            st.write("**Simulated Portfolio Weights:**")
            st.write(weights.round(4))

        # Download buttons
        st.download_button("⬇️ Download Price Data", df_price.to_csv(index=False), file_name="price_data.csv")
        st.download_button("⬇️ Download News Sentiment Data", df_news.to_csv(index=False), file_name="news_sentiment.csv")
else:
    st.info("Click 'Run Analysis' to fetch and analyze crypto data.")
