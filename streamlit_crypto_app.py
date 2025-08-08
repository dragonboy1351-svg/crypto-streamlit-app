import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import altair as alt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import google.generativeai as genai

# -------------------------------
# --- Hardcoded API Key (Crypto)
# -------------------------------
CRYPTO_API_KEY = "your_crypto_compare_api_key"  # replace this with your actual key

# -------------------------------
# --- Gemini API Key (Hardcoded)
# -------------------------------
GEMINI_API_KEY = "AIzaSyD8dbzMGmUYkuK2nXSO8zJsMyho1t6onfk"

# -------------------------------
# --- Streamlit Layout
# -------------------------------
st.set_page_config(page_title="Crypto Portfolio Optimisation", layout="wide")
st.title("📊 Crypto Portfolio Optimisation")

# -------------------------------
# --- User Settings
# -------------------------------
coins = st.multiselect("Select cryptocurrencies", ["BTC", "ETH", "XRP"], default=["BTC", "ETH"])
days = st.slider("How many days of data?", min_value=30, max_value=365, value=90)

# -------------------------------
# --- Helper Functions
# -------------------------------
def get_crypto_data(symbol):
    url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={symbol}&tsym=USD&limit={days}&api_key={CRYPTO_API_KEY}"
    response = requests.get(url).json()
    if response["Response"] == "Success":
        df = pd.DataFrame(response["Data"]["Data"])
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df["symbol"] = symbol
        return df[["time", "close", "symbol"]]
    else:
        return pd.DataFrame()

def get_sentiment_scores(coins):
    url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN&api_key=" + CRYPTO_API_KEY
    response = requests.get(url).json()
    analyzer = SentimentIntensityAnalyzer()

    sentiment_data = {coin: [] for coin in coins}
    for article in response["Data"]:
        title = article.get("title", "")
        for coin in coins:
            if coin in title:
                score = analyzer.polarity_scores(title)["compound"]
                published = datetime.datetime.fromtimestamp(article["published_on"])
                sentiment_data[coin].append((published.date(), score))

    sentiment_df = []
    for coin, entries in sentiment_data.items():
        if entries:
            df = pd.DataFrame(entries, columns=["date", "sentiment"])
            df = df.groupby("date").mean().reset_index()
            df["symbol"] = coin
            sentiment_df.append(df)

    if sentiment_df:
        return pd.concat(sentiment_df)
    else:
        return pd.DataFrame()

# -------------------------------
# --- Main App Logic
# -------------------------------
price_data = pd.concat([get_crypto_data(c) for c in coins])
sentiment_data = get_sentiment_scores(coins)

# Compute metrics
price_data["return"] = price_data.groupby("symbol")["close"].pct_change()
price_data["volatility"] = price_data.groupby("symbol")["return"].rolling(7).std().reset_index(level=0, drop=True)
price_data["momentum"] = price_data.groupby("symbol")["close"].pct_change(7)

# Join with sentiment
price_data["date"] = price_data["time"].dt.date
merged = pd.merge(price_data, sentiment_data, on=["date", "symbol"], how="left')

# -------------------------------
# --- Dashboard
# -------------------------------
st.subheader("📈 Price Chart")
line = alt.Chart(price_data).mark_line().encode(
    x="time", y="close", color="symbol"
).properties(width=700, height=400)
st.altair_chart(line, use_container_width=True)

st.subheader("📊 Volatility & Momentum (7-day)")
metrics = price_data.groupby("symbol").agg({
    "return": "mean",
    "volatility": "mean",
    "momentum": "mean"
}).rename(columns={
    "return": "Avg Daily Return",
    "volatility": "7d Volatility",
    "momentum": "7d Momentum"
})
st.dataframe(metrics.round(4))

if not sentiment_data.empty:
    st.subheader("🧠 Average Sentiment Score")
    avg_sentiment = sentiment_data.groupby("symbol")["sentiment"].mean().round(4)
    st.write(avg_sentiment)

# -------------------------------
# --- Gemini Integration
# -------------------------------
st.markdown("---")
st.subheader("🔮 Ask Gemini About the Market")

try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-pro")

    prompt = st.text_area("Ask a question about crypto sentiment (e.g., 'Explain today's sentiment on BTC'):")

    if st.button("Ask Gemini"):
        with st.spinner("Gemini is thinking..."):
            response = model.generate_content(prompt)
            st.success("✅ Gemini's Response:")
            st.write(response.text)
except Exception as e:
    st.error(f"❌ Gemini API Error: {e}")
