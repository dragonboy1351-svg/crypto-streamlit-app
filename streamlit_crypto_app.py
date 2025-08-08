import streamlit as st
import pandas as pd
import numpy as np
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import google.generativeai as genai
import cvxpy as cp
import datetime

# -----------------------------
# HARDCODED API KEYS
# -----------------------------
CRYPTO_API_KEY = "your_crypto_api_key_here"  # replace with your actual CryptoCompare key
GEMINI_API_KEY = "AIzaSyD8dbzMGmUYkuK2nXSO8zJsMyho1t6onfk"

# -----------------------------
# Configure Gemini
# -----------------------------
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-pro")

# -----------------------------
# App Layout
# -----------------------------
st.set_page_config(page_title="Crypto Portfolio Optimisation", layout="wide")
st.title("📊 Crypto Portfolio Optimisation")

# -----------------------------
# Functions
# -----------------------------

def get_price_data(symbol: str, limit: int = 100) -> pd.DataFrame:
    url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={symbol}&tsym=USD&limit={limit}&api_key={CRYPTO_API_KEY}"
    res = requests.get(url)
    data = res.json().get("Data", {}).get("Data", [])
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.rename(columns={"time": "date"}, inplace=True)
    df["symbol"] = symbol
    return df[["date", "symbol", "close"]]

def get_sentiment_scores(symbol: str, limit: int = 30) -> pd.DataFrame:
    url = f"https://min-api.cryptocompare.com/data/v2/news/?categories={symbol}&api_key={CRYPTO_API_KEY}"
    res = requests.get(url)
    articles = res.json().get("Data", [])
    analyzer = SentimentIntensityAnalyzer()
    rows = []
    for article in articles[:limit]:
        score = analyzer.polarity_scores(article["title"])["compound"]
        date = datetime.datetime.utcfromtimestamp(article["published_on"]).date()
        rows.append({"date": date, "symbol": symbol, "sentiment": score})
    return pd.DataFrame(rows)

def optimise_portfolio(returns: pd.Series, cov_matrix: pd.DataFrame, sentiment_score: float):
    n = len(returns)
    w = cp.Variable(n)
    risk_aversion = max(0.01, 1 - sentiment_score)
    objective = cp.Maximise(returns.values @ w - risk_aversion * cp.quad_form(w, cov_matrix.values))
    constraints = [cp.sum(w) == 1, w >= 0, w <= 0.5]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return w.value

def get_gemini_summary(symbol: str, avg_sentiment: float):
    prompt = f"The average sentiment for {symbol} is {avg_sentiment:.2f}. Explain this in simple terms and whether it's positive or negative sentiment."
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini error: {str(e)}"

# -----------------------------
# App Logic
# -----------------------------

# Step 1: User selection
st.sidebar.header("Select Cryptocurrencies")
selected_symbols = st.sidebar.multiselect("Pick coins", ["BTC", "ETH", "XRP", "LTC", "ADA"], default=["BTC", "ETH"])

# Step 2: Load data
price_data_all = []
sentiment_data_all = []

for sym in selected_symbols:
    price_data_all.append(get_price_data(sym))
    sentiment_data_all.append(get_sentiment_scores(sym))

price_df = pd.concat(price_data_all)
sentiment_df = pd.concat(sentiment_data_all)

# Merge price and sentiment data
merged = pd.merge(price_df, sentiment_df, on=["date", "symbol"], how="left")
merged["sentiment"].fillna(0, inplace=True)

# Step 3: Show data
st.subheader("📈 Market Data + Sentiment")
st.dataframe(merged.tail(10), use_container_width=True)

# Step 4: Compute average sentiment and show Gemini explanation
sentiment_summary = merged.groupby("symbol")["sentiment"].mean().to_dict()

st.subheader("🧠 Sentiment Interpretation (Gemini)")
for sym in selected_symbols:
    avg_s = sentiment_summary.get(sym, 0)
    gemini_text = get_gemini_summary(sym, avg_s)
    st.markdown(f"**{sym}**: {gemini_text}")

# Step 5: Portfolio Optimisation
st.subheader("📊 Portfolio Allocation")

# Pivot prices for return calculation
pivot = price_df.pivot(index="date", columns="symbol", values="close")
returns = pivot.pct_change().dropna()

# Calculate mean returns & cov matrix
mean_returns = returns.mean()
cov_matrix = returns.cov()

# Get average market sentiment
market_sentiment = np.mean(list(sentiment_summary.values()))

# Run optimisation
weights = optimise_portfolio(mean_returns, cov_matrix, market_sentiment)

# Show weights
weight_df = pd.DataFrame({
    "Symbol": mean_returns.index,
    "Weight": np.round(weights, 4)
})
st.dataframe(weight_df, use_container_width=True)

# Optional chart
st.bar_chart(weight_df.set_index("Symbol"))

# -----------------------------
# Done
# -----------------------------
st.caption("Built with sentiment + Gemini-powered insights 🔮")
