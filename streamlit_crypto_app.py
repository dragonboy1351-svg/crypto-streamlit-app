import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime

# ----------------------------------------
# 🔐 Hardcoded CryptoCompare API Key
API_KEY = "ca28d0c8038e074b58ba188a33bdefad11bf7dbbfc739fe5942f8a3323ee075a "  # ← Replace with your actual API key
# ----------------------------------------

# ----------------------------
# 📥 Data Fetching
# ----------------------------
@st.cache_data
def get_price_data(symbol: str, api_key: str, limit: int = 200) -> pd.DataFrame:
    url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={symbol}&tsym=USD&limit={limit}&api_key={api_key}"
    response = requests.get(url)
    data = response.json()["Data"]["Data"]
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    return df

@st.cache_data
def get_crypto_news(api_key: str, categories="BTC,ETH,XRP") -> list:
    url = f"https://min-api.cryptocompare.com/data/v2/news/?categories={categories}&api_key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()["Data"]
    return []

# ----------------------------
# 🧠 Sentiment Scoring
# ----------------------------
def clean_text(text: str) -> str:
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s]+", "", text)
    return text.strip()

def analyze_sentiment(news: list) -> dict:
    analyzer = SentimentIntensityAnalyzer()
    scores = {"BTC": [], "ETH": [], "XRP": []}

    for article in news:
        text = clean_text(article.get("title", ""))
        score = analyzer.polarity_scores(text)["compound"]
        for coin in scores:
            if coin in article.get("categories", "") or coin in article.get("title", ""):
                scores[coin].append(score)
    
    return {coin: np.mean(vals) if vals else 0 for coin, vals in scores.items()}

# ----------------------------
# 💬 Gemini Mock Assistant
# ----------------------------
def mock_gemini_response(coin: str, sentiment_score: float) -> str:
    if sentiment_score < -0.5:
        return f"{coin} is experiencing very negative sentiment. This may reflect bearish news, regulatory fears, or a drop in investor confidence."
    elif sentiment_score < -0.1:
        return f"{coin} has mildly negative sentiment. There might be cautious news or market hesitation."
    elif sentiment_score < 0.1:
        return f"{coin} sentiment is neutral. Market participants seem uncertain or are waiting for stronger signals."
    elif sentiment_score < 0.5:
        return f"{coin} has positive sentiment. Possibly driven by optimistic headlines or market momentum."
    else:
        return f"{coin} sentiment is strongly positive. Likely influenced by bullish news, positive partnerships, or strong community sentiment."

# ----------------------------
# 🖥️ Streamlit UI
# ----------------------------
st.set_page_config(page_title="Crypto Portfolio Optimisation", layout="wide")
st.title("📊 Crypto Portfolio Optimisation Dashboard")

st.sidebar.header("Dashboard Settings")
selected_symbols = st.sidebar.multiselect("Select Cryptocurrencies", ["BTC", "ETH", "XRP"], default=["BTC", "ETH", "XRP"])
rolling_window = st.sidebar.slider("Rolling Window (days)", 3, 30, 7)

# Load Data
price_data = {symbol: get_price_data(symbol, API_KEY) for symbol in selected_symbols}
news_articles = get_crypto_news(API_KEY)
sentiment_scores = analyze_sentiment(news_articles)

# ----------------------------
# 📈 Charts & Metrics
# ----------------------------
st.subheader("📉 Market Data Overview")

for symbol in selected_symbols:
    df = price_data[symbol]
    st.markdown(f"### {symbol} Historical Price")
    st.line_chart(df["close"], height=200)

    df["daily_return"] = df["close"].pct_change()
    df["volatility"] = df["daily_return"].rolling(window=rolling_window).std()
    df["momentum"] = df["close"].pct_change(periods=rolling_window)

    st.markdown(f"**{symbol} - Daily Return (%):**")
    st.line_chart(df["daily_return"] * 100)

    st.markdown(f"**{symbol} - {rolling_window}-Day Rolling Volatility (%):**")
    st.line_chart(df["volatility"] * 100)

    st.markdown(f"**{symbol} - {rolling_window}-Day Momentum (%):**")
    st.line_chart(df["momentum"] * 100)

# ----------------------------
# 🧠 Sentiment Analysis
# ----------------------------
st.subheader("📰 Sentiment Analysis")
st.markdown("Sentiment is computed using VADER on recent news headlines.")

sentiment_df = pd.DataFrame.from_dict(sentiment_scores, orient="index", columns=["Sentiment Score"])
st.table(sentiment_df.style.format("{:.2f}"))

# ----------------------------
# 💬 Gemini (Mock) Assistant
# ----------------------------
st.subheader("💬 Gemini Assistant (Mock AI Insight)")
selected_coin = st.selectbox("Select a coin for insight:", sentiment_scores.keys())

if selected_coin:
    score = sentiment_scores[selected_coin]
    insight = mock_gemini_response(selected_coin, score)
    st.info(f"**Gemini says:** {insight}")

st.markdown("---")
st.caption("🚀 Built with VADER, CryptoCompare, and Streamlit")
