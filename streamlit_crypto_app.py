import streamlit as st
import pandas as pd
import numpy as np
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import google.generativeai as genai

# --- API Keys ---
CRYPTO_API_KEY = "ca28d0c8038e074b58ba188a33bdefad11bf7dbbfc739fe5942f8a3323ee075a"
GEMINI_API_KEY = "AIzaSyCFWIl2SrnRo7T25G4vp4O-CPy-O7UpuzY"

# --- Configure Gemini ---
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-pro")

# --- App Layout ---
st.set_page_config(page_title="Simple Crypto Dashboard", layout="wide")
st.title("📊 Simple Crypto Dashboard")

# --- Get Price Data ---
def get_price_data(symbol: str, limit: int = 30) -> pd.DataFrame:
    url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={symbol}&tsym=USD&limit={limit}&api_key={CRYPTO_API_KEY}"
    res = requests.get(url)
    data = res.json().get("Data", {}).get("Data", [])
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.rename(columns={"time": "date", "close": symbol}, inplace=True)
    return df[["date", symbol]]

# --- Get Sentiment ---
def get_sentiment(symbol: str) -> float:
    url = f"https://min-api.cryptocompare.com/data/v2/news/?categories={symbol}&api_key={CRYPTO_API_KEY}"
    res = requests.get(url)
    articles = res.json().get("Data", [])
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(article["title"])['compound'] for article in articles[:10]]
    return np.mean(scores) if scores else 0

# --- Gemini Chat Summary ---
def get_gemini_summary(symbol: str, avg_sentiment: float):
    prompt = f"The average sentiment for {symbol} is {avg_sentiment:.2f}. Explain this in simple terms and whether it's positive or negative."
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini error: {str(e)}"

# --- Sidebar ---
st.sidebar.header("Select Coins")
symbols = st.sidebar.multiselect("Cryptos", ["BTC", "ETH", "XRP", "LTC"], default=["BTC", "ETH"])

# --- Show Data ---
all_prices = []
sentiments = {}

for symbol in symbols:
    df = get_price_data(symbol)
    all_prices.append(df.set_index("date"))
    sentiments[symbol] = get_sentiment(symbol)

if all_prices:
    price_df = pd.concat(all_prices, axis=1)
    st.subheader("📈 Price Chart")
    st.line_chart(price_df)

    st.subheader("🧠 Sentiment Scores")
    st.write(pd.DataFrame.from_dict(sentiments, orient='index', columns=["Sentiment"]))

    st.subheader("🔮 Gemini Summary")
    for symbol in symbols:
        st.markdown(f"**{symbol}**")
        st.write(get_gemini_summary(symbol, sentiments[symbol]))

st.caption("Simplified crypto dashboard using CryptoCompare, VADER and Gemini API.")
