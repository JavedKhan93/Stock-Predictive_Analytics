import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import os
import yfinance as yf

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    layout="wide",
    page_title="FinTech Pro Analytics",
    page_icon="📈"
)

# --- 2. ADVANCED CSS STYLING (THE "PRO" LOOK) ---
st.markdown("""
<style>
    /* Main Background - Soft Gradient */
    .stApp {
        background: linear-gradient(to right, #f8f9fa, #e9ecef);
        color: #212529;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
        box-shadow: 2px 0 5px rgba(0,0,0,0.05);
    }

    /* Card Container Style (Glassmorphism Light) */
    .css-card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 25px;
        border: 1px solid #f0f0f0;
    }

    /* Metric Box Styling */
    .metric-box {
        background: #f8f9fa;
        border-left: 4px solid #4CAF50;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    /* Headlines */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        color: #1a1a1a;
        font-weight: 700;
    }

    /* Custom Badge for AI */
    .ai-badge {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
        border: 1px solid #a5d6a7;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 4px;
        padding: 10px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50 !important;
        color: white !important;
    }

</style>
""", unsafe_allow_html=True)


# --- 3. HELPER FUNCTIONS ---
@st.cache_data
def load_local_data():
    file_path = "dataset/stock_market_data_2014_2024.csv"
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, format='mixed')
    return df


def get_live_company_info(ticker):
    """Fetches live info from Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info
    except:
        return None


# --- 4. SIDEBAR & NAVIGATION ---
st.sidebar.markdown("## 💎 FinTech Pro")
st.sidebar.markdown("---")
st.sidebar.header("🎯 Asset Selection")

ticker_options = {
    'JPM': 'JP Morgan Chase',
    'GS': 'Goldman Sachs',
    'RELIANCE.NS': 'Reliance Industries',
    'HDFCBANK.NS': 'HDFC Bank'
}
ticker = st.sidebar.selectbox("Choose Ticker", list(ticker_options.keys()),
                              format_func=lambda x: f"{x} - {ticker_options[x]}")

# Load Data
df = load_local_data()
if df is not None:
    stock_df = df[df['Ticker'] == ticker].copy()
    stock_df.set_index('Date', inplace=True)
    stock_df.sort_index(inplace=True)

# Fetch Live Info
info = get_live_company_info(ticker)

if info:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🏢 Company Profile")

    # Safe Logo Logic
    logo_url = info.get('logo_url')
    if logo_url and logo_url != '':
        st.sidebar.image(logo_url, width=120)

    st.sidebar.markdown(f"**Sector:** {info.get('sector', 'N/A')}")
    st.sidebar.markdown(f"**Industry:** {info.get('industry', 'N/A')}")

    with st.sidebar.expander("📖 Business Summary"):
        st.write(info.get('longBusinessSummary', 'No summary available.'))

# --- 5. MAIN DASHBOARD HEADER ---
st.markdown(f"<h1>📈 {ticker_options[ticker]} <span style='font-size: 20px; color: gray;'>({ticker})</span></h1>",
            unsafe_allow_html=True)
st.markdown("**Real-Time AI Predictive Analytics & Technical Insights**")
st.markdown("---")

# --- KPI ROW (Using Custom CSS Cards) ---
if df is not None:
    last_close = stock_df['Close'].iloc[-1]
    prev_close = stock_df['Close'].iloc[-2]
    change = last_close - prev_close
    pct_change = (change / prev_close) * 100
    volume = stock_df['Volume'].iloc[-1]

    # Use Streamlit columns but wrap contents in styling
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    kpi1.metric(label="📉 Last Close", value=f"${last_close:,.2f}", delta=f"{pct_change:.2f}%")
    kpi2.metric(label="🆙 Day High", value=f"${stock_df['High'].iloc[-1]:,.2f}")
    kpi3.metric(label="⬇️ Day Low", value=f"${stock_df['Low'].iloc[-1]:,.2f}")
    kpi4.metric(label="📊 Volume", value=f"{volume:,.0f}")

st.markdown("---")

# --- 6. CORE TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Technical Analysis", "🧠 AI Forecast", "🚦 Signals", "📰 News"])

# === TAB 1: PRO CHARTS ===
with tab1:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)  # Start Card
    st.subheader("Interactive Market Chart")

    if 'stock_df' not in locals():
    st.error("⚠️ critical error: 'stock_df' is not defined. The data failed to load.")
    st.info("Check if 'stock_market_data_2014_2024.csv' is uploaded and the path is correct.")
    st.stop()  # Stop the script here so it doesn't crash below

    col_ctrl1, col_ctrl2 = st.columns([1, 3])
    with col_ctrl1:
        chart_type = st.radio("Chart Type:", ["Candlestick", "Line"], horizontal=True)
    with col_ctrl2:
        indicators = st.multiselect("Overlay Indicators:", ["SMA 50", "SMA 200", "Bollinger Bands"], default=["SMA 50"])

    fig = go.Figure()

    if chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(x=stock_df.index,
                                     open=stock_df['Open'], high=stock_df['High'],
                                     low=stock_df['Low'], close=stock_df['Close'], name='OHLC'))
    else:
        fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['Close'],
                                 mode='lines', name='Close Price', line=dict(width=2)))

    # Dynamic Indicators
    if "SMA 50" in indicators:
        sma50 = stock_df['Close'].rolling(50).mean()
        fig.add_trace(go.Scatter(x=stock_df.index, y=sma50, line=dict(color='orange', width=1.5), name='SMA 50'))

    if "SMA 200" in indicators:
        sma200 = stock_df['Close'].rolling(200).mean()
        fig.add_trace(go.Scatter(x=stock_df.index, y=sma200, line=dict(color='#008000', width=1.5), name='SMA 200'))

    if "Bollinger Bands" in indicators:
        sma20 = stock_df['Close'].rolling(20).mean()
        std = stock_df['Close'].rolling(20).std()
        upper = sma20 + (std * 2)
        lower = sma20 - (std * 2)
        fig.add_trace(
            go.Scatter(x=stock_df.index, y=upper, line=dict(color='gray', width=1, dash='dot'), name='Upper BB'))
        fig.add_trace(
            go.Scatter(x=stock_df.index, y=lower, line=dict(color='gray', width=1, dash='dot'), name='Lower BB'))

    fig.update_layout(height=550, template="plotly_white", xaxis_rangeslider_visible=False,
                      margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)  # End Card

# === TAB 2: AI PREDICTION ===
with tab2:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.subheader("🤖 LSTM Deep Learning Prediction")
    st.markdown("<span class='ai-badge'>Running on TensorFlow Backend</span>", unsafe_allow_html=True)
    st.write("")  # Spacer

    model_path = "lstm_stock_model.h5"
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            data = stock_df.filter(['Close']).dropna()
            dataset = data.values
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(dataset)

            if len(scaled_data) >= 60:
                last_60_days = scaled_data[-60:]
                X_test = np.array([last_60_days])
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

                pred = model.predict(X_test)
                pred = scaler.inverse_transform(pred)

                curr = data['Close'].iloc[-1]
                future = pred[0][0]
                diff = future - curr

                c1, c2, c3 = st.columns(3)
                c1.metric("Current Price", f"${curr:.2f}")
                c2.metric("AI Predicted (Tomorrow)", f"${future:.2f}")
                c3.metric("Expected Move", f"${diff:.2f}", delta_color="normal")

                st.write("---")
                st.write("#### 🧠 AI Trend Visualizer")
                st.caption("Comparing recent trend vs AI Forecast")
                st.line_chart(data['Close'].tail(50))

                if diff > 0:
                    st.success(f"✅ **BULLISH Signal:** The AI model expects the price to RISE by ${diff:.2f}")
                else:
                    st.error(f"🔻 **BEARISH Signal:** The AI model expects the price to FALL by ${abs(diff):.2f}")
        except Exception as e:
            st.error(f"Prediction Error: {e}")
    else:
        st.warning("⚠️ Model file not found.")
    st.markdown('</div>', unsafe_allow_html=True)

# === TAB 3: TRADING SIGNALS ===
with tab3:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.subheader("⚡ Technical Signals Dashboard")

    # Calculations
    delta = stock_df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    stock_df['RSI'] = 100 - (100 / (1 + rs))
    rsi = stock_df['RSI'].iloc[-1]

    sma50 = stock_df['Close'].rolling(50).mean().iloc[-1]
    price = stock_df['Close'].iloc[-1]

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"### RSI Momentum: **{rsi:.2f}**")
        if rsi > 70:
            st.error("🔴 Overbought (Sell Risk High)")
        elif rsi < 30:
            st.success("🟢 Oversold (Buy Opportunity)")
        else:
            st.info("🔵 Neutral Zone")

    with col2:
        st.write(f"### Trend Strength")
        if price > sma50:
            st.success(f"✅ Price is ABOVE 50-Day SMA (Uptrend)")
        else:
            st.error(f"🔻 Price is BELOW 50-Day SMA (Downtrend)")

    st.markdown('</div>', unsafe_allow_html=True)

# === TAB 4: NEWS ===
with tab4:
    st.subheader(f"📰 Latest News for {ticker}")

    # Robust News Logic
    try:
        stock = yf.Ticker(ticker)
        news_list = stock.news

        if not news_list:
            st.info(f"Direct news feed temporarily unavailable for {ticker}.")
            st.markdown(
                f"👉 **[Click here to read real-time news on Yahoo Finance](https://finance.yahoo.com/quote/{ticker}/news)**")
        else:
            for item in news_list[:5]:
                # Card style for news
                st.markdown(f"""
                <div style="background-color: white; padding: 15px; border-radius: 10px; margin-bottom: 10px; border: 1px solid #eee;">
                    <h5><a href="{item.get('link')}" target="_blank" style="text-decoration: none; color: #1a1a1a;">{item.get('title')}</a></h5>
                    <small style="color: gray;">{item.get('publisher')} • {pd.to_datetime(item.get('providerPublishTime'), unit='s').strftime('%Y-%m-%d %H:%M')}</small>
                </div>
                """, unsafe_allow_html=True)

    except:
        st.warning("Could not fetch live news at this moment.")

        st.markdown(f"👉 **[Click here to read {ticker} News directly](https://finance.yahoo.com/quote/{ticker}/news)**")
