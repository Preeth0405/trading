import streamlit as st
import pandas as pd
import os
import requests
from dotenv import load_dotenv
from alpha_vantage.timeseries import TimeSeries
import ta
from openai import OpenAI
import plotly.graph_objects as go
import feedparser

# Load API keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY") or "demo"  # Replace "demo" with real key

client = OpenAI(api_key=OPENAI_API_KEY)
ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')

# Stock symbols with descriptions
STOCK_LABELS = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corp.",
    "NVDA": "NVIDIA Corp.",
    "GOOGL": "Alphabet Inc.",
    "AMD": "Advanced Micro Devices",
    "QCOM": "Qualcomm",
    "TSLA": "Tesla Inc.",
    "AVGO": "Broadcom",
    "SAGE.L": "Sage Group (UK)",
    "SOPH.L": "Sophos Group (UK)",
    "AVST.L": "Avast (UK)",
    "NOK": "Nokia",
    "ERIC": "Ericsson",
    "F": "Ford Motor Co.",
    "GM": "General Motors",
    "TM": "Toyota Motor Corp.",
    "HMC": "Honda Motor Co.",
    "RACE": "Ferrari NV",
    "JPM": "JPMorgan Chase",
    "BAC": "Bank of America",
    "C": "Citigroup",
    "HSBC": "HSBC Holdings",
    "WFC": "Wells Fargo",
    "BARC.L": "Barclays (UK)",
    "HSBA.L": "HSBC Holdings (UK)",
    "BLK": "BlackRock",
    "GS": "Goldman Sachs",
    "MS": "Morgan Stanley",
    "SCHW": "Charles Schwab",
    "BRK.B": "Berkshire Hathaway",
    "NEE": "NextEra Energy",
    "DUK": "Duke Energy",
    "AES": "AES Corp.",
    "EXC": "Exelon Corp.",
    "NRG": "NRG Energy",
    "NG.L": "National Grid (UK)",
    "SSE.L": "SSE Plc (UK)",
    "DRX.L": "Drax Group (UK)"
}

# Sector and region-wise stock map
STOCK_MAP = {
    "Technology": {
        "US": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMD", "QCOM", "TSLA", "AVGO"],
        "UK": ["SAGE.L", "SOPH.L", "AVST.L"]
    },
    "Phones": {
        "US": ["AAPL", "QCOM", "NOK", "ERIC"]
    },
    "Cars": {
        "US": ["TSLA", "F", "GM", "TM", "HMC", "RACE"]
    },
    "Banks": {
        "US": ["JPM", "BAC", "C", "HSBC", "WFC"],
        "UK": ["BARC.L", "HSBA.L"]
    },
    "Investment": {
        "US": ["BLK", "GS", "MS", "SCHW", "BRK.B"]
    },
    "Power": {
        "US": ["NEE", "DUK", "AES", "EXC", "NRG"],
        "UK": ["NG.L", "SSE.L", "DRX.L"]
    }
}

# Streamlit UI setup
st.set_page_config(page_title="GPT Trade Assistant", layout="centered")
st.title("📈 GPT Swing Trade Assistant")

sector = st.selectbox("Select Sector", list(STOCK_MAP.keys()))
region = st.selectbox("Select Region", list(STOCK_MAP[sector].keys()))
symbol = st.selectbox("Choose Stock", STOCK_MAP[sector][region], format_func=lambda x: f"{x} - {STOCK_LABELS.get(x, '')}")

# Analyze and Suggest
if st.button("📊 Analyze & Suggest Trade"):
    try:
        df, _ = ts.get_daily(symbol=symbol, outputsize="compact")
        df = df.sort_index(ascending=True)
        df["SMA_5"] = ta.trend.sma_indicator(df["4. close"], window=5)
        df["RSI_14"] = ta.momentum.rsi(df["4. close"], window=14)

        st.subheader("Last 5 Days Data")
        st.dataframe(df.tail(5)[["4. close", "SMA_5", "RSI_14"]])

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index[-20:], y=df["4. close"][-20:], name="Close Price"))
        fig.add_trace(go.Scatter(x=df.index[-20:], y=df["SMA_5"][-20:], name="SMA-5"))
        fig.update_layout(title=f"{symbol} Price Trend", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig)

        # GPT Prompt
        last_close = df["4. close"].iloc[-1]
        sma_val = df["SMA_5"].iloc[-1]
        rsi_val = df["RSI_14"].iloc[-1]
        last_5 = df["4. close"].tail(5).to_string()

        prompt = f"""
You are an expert trader.

Stock: {symbol}
- Last Close: {last_close:.2f}
- SMA (5-day): {sma_val:.2f}
- RSI (14-day): {rsi_val:.2f}
- Last 5 closes: {last_5}

Suggest a 2–3 day swing trade strategy with entry, stop-loss, target and your reasoning.
"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )

        st.subheader("💡 GPT Trade Suggestion")
        st.write(response.choices[0].message.content)

    except Exception as e:
        st.error(f"Error fetching or processing data: {e}")

# News Summary
st.divider()
st.subheader("📰 Latest News Summary")

try:
    feed = feedparser.parse(f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US")
    headlines = "\n".join(f"- {entry.title}" for entry in feed.entries[:10])

    if headlines:
        for entry in feed.entries[:5]:
            st.write(f"- {entry.title}")

        summary_prompt = f"""
You are a financial news summarizer.

Based on these headlines for {symbol}:
{headlines}

Give:
1. What happened recently?
2. What’s happening now?
3. What could happen next quarter or year?
"""

        summary = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=500
        )

        st.subheader("📘 GPT News Summary")
        st.write(summary.choices[0].message.content)
    else:
        st.warning("No news found for this stock.")

except Exception as e:
    st.error(f"Failed to fetch or summarize news: {e}")

# Profit / Loss Calculator
st.divider()
st.subheader("💰 Profit / Loss Calculator")

with st.form("pnl_calc"):
    buy_price = st.number_input("Buy Price", value=100.0)
    sell_price = st.number_input("Sell Price", value=110.0)
    qty = st.number_input("Quantity", value=10, step=1)
    calc = st.form_submit_button("Calculate")

    if calc:
        profit = (sell_price - buy_price) * qty
        result = "Profit" if profit > 0 else "Loss" if profit < 0 else "Break-even"
        st.metric(label=f"📈 {result}", value=f"${profit:.2f}")
