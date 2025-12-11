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

# --- 2. ADVANCED CSS STYLING ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: linear-gradient(to right, #f8f9fa, #e9ecef);
        color: #212529;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    /* Card Styling */
    .css-card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 25px;
        border: 1px solid #f0f0f0;
    }
    /* AI Badge */
    .ai-badge {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
        border: 1px solid #a5d6a7;
    }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
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


# --- 3. ROBUST DATA LOADING (FIXED) ---
@st.cache_data
def load_data(ticker_symbol):
    """
    Tries to load from local CSV. 
    If CSV is missing or ticker not found, falls back to Yahoo Finance live download.
    """
    # 1. Try Local CSV
    file_path = "dataset/stock_market_data_2014_2024.csv"
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            # Ensure Date parsing is robust
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            
            # Filter for specific ticker
            mask = df['Ticker'] == ticker_symbol
            if mask.any():
                local_df = df[mask].copy()
                local_df.set_index('Date', inplace=True)
                local_df.sort_index(inplace=True)
                return local_df
        except Exception as e:
            st.warning(f"⚠️ Error reading local CSV: {e}")

    # 2. Fallback: Download from Yahoo Finance
    # st.toast(f"Downloading live data for {ticker_symbol}...", icon="☁️")
    try:
        live_df = yf.download(ticker_symbol, period="5y", auto_adjust=False, progress=False)
        
        # Fix for yfinance v0.2+ returning MultiIndex columns
        if isinstance(live_df.columns, pd.MultiIndex):
            live_df.columns = live_df.columns.get_level_values(0)
            
        # Ensure columns map correctly
        live_df = live_df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        live_df.dropna(inplace=True)
        return live_df
    except Exception as e:
        st.error(f"❌ Failed to download data for {ticker_symbol}: {e}")
        return None

def get_live_company_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        return stock.info
    except:
        return None


# --- 4. SIDEBAR & NAVIGATION ---
st.sidebar.markdown("## 💎 FinTech Pro")
st.sidebar.markdown("---")

ticker_options = {
    'JPM': 'JP Morgan Chase',
    'GS': 'Goldman Sachs',
    'RELIANCE.NS': 'Reliance Industries',
    'HDFCBANK.NS': 'HDFC Bank',
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corp.'
}
ticker = st.sidebar.selectbox(
    "Choose Ticker", 
    list(ticker_options.keys()), 
    format_func=lambda x: f"{x} - {ticker_options.get(x, x)}"
)

# === LOAD DATA GLOBALLY ===
stock_df = load_data(ticker)

# === CRITICAL SAFETY CHECK ===
if stock_df is None or stock_df.empty:
    st.error(f"⛔ No data found for {ticker}. Please check your dataset or internet connection.")
    st.stop()  # Stops execution here so the app doesn't crash below

# Sidebar Info
info = get_live_company_info(ticker)
if info:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🏢 Company Profile")
    
    logo_url = info.get('logo_url')
    if logo_url:
        st.sidebar.image(logo_url, width=120)

    st.sidebar.markdown(f"**Sector:** {info.get('sector', 'N/A')}")
    st.sidebar.markdown(f"**Industry:** {info.get('industry', 'N/A')}")
    
    with st.sidebar.expander("📖 Business Summary"):
        st.write(info.get('longBusinessSummary', 'No summary available.'))


# --- 5. MAIN DASHBOARD ---
st.markdown(f"<h1>📈 {ticker_options.get(ticker, ticker)} <span style='font-size: 20px; color: gray;'>({ticker})</span></h1>", unsafe_allow_html=True)
st.markdown("**Real-Time AI Predictive Analytics & Technical Insights**")
st.markdown("---")

# --- KPI ROW ---
# Safe calculation of metrics
try:
    last_close = stock_df['Close'].iloc[-1]
    prev_close = stock_df['Close'].iloc[-2]
    change = last_close - prev_close
    pct_change = (change / prev_close) * 100
    volume = stock_df['Volume'].iloc[-1]
    high_val = stock_df['High'].iloc[-1]
    low_val = stock_df['Low'].iloc[-1]

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric(label="📉 Last Close", value=f"${last_close:,.2f}", delta=f"{pct_change:.2f}%")
    kpi2.metric(label="🆙 Day High", value=f"${high_val:,.2f}")
    kpi3.metric(label="⬇️ Day Low", value=f"${low_val:,.2f}")
    kpi4.metric(label="📊 Volume", value=f"{volume:,.0f}")
except:
    st.warning("Not enough data to calculate daily changes.")

st.markdown("---")

# --- 6. CORE TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Technical Analysis", "🧠 AI Forecast", "🚦 Signals", "📰 News"])

# === TAB 1: PRO CHARTS ===
with tab1:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.subheader("Interactive Market Chart")
    
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
        fig.add_trace(go.Scatter(x=stock_df.index, y=upper, line=dict(color='gray', width=1, dash='dot'), name='Upper BB'))
        fig.add_trace(go.Scatter(x=stock_df.index, y=lower, line=dict(color='gray', width=1, dash='dot'), name='Lower BB'))

    fig.update_layout(height=550, template="plotly_white", xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# === TAB 2: AI PREDICTION ===
with tab2:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.subheader("🤖 LSTM Deep Learning Prediction")
    st.markdown("<span class='ai-badge'>Running on TensorFlow Backend</span>", unsafe_allow_html=True)
    st.write("") 

    model_path = "lstm_stock_model.h5"
    
    if os.path.exists(model_path):
        try:
            # We already have stock_df here, no need to check locals()
            data = stock_df.filter(['Close']).dropna()
            
            if len(data) < 60:
                st.warning("Not enough data points (need 60+) to run the AI model.")
            else:
                model = tf.keras.models.load_model(model_path)
                dataset = data.values
                scaler = MinMaxScaler(feature_range=(0, 1))
                # Note: fitting on whole dataset just for demo scaling
                scaler.fit(dataset) 
                
                # Get last 60 days
                last_60_days = dataset[-60:]
                scaled_last_60 = scaler.transform(last_60_days)
                
                X_test = np.array([scaled_last_60])
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

                pred_scaled = model.predict(X_test)
                pred_price = scaler.inverse_transform(pred_scaled)

                curr_price = data['Close'].iloc[-1]
                future_price = pred_price[0][0]
                diff = future_price - curr_price

                # Metrics
                c1, c2, c3 = st.columns(3)
                c1.metric("Current Price", f"${curr_price:.2f}")
                c2.metric("AI Predicted (Tomorrow)", f"${future_price:.2f}")
                c3.metric("Expected Move", f"${diff:.2f}", delta_color="normal")

                st.write("---")
                st.write("#### 🧠 AI Trend Visualizer")
                st.caption("Recent Price Action")
                st.line_chart(data['Close'].tail(50))

                if diff > 0:
                    st.success(f"✅ **BULLISH Signal:** The AI model expects the price to RISE by ${diff:.2f}")
                else:
                    st.error(f"🔻 **BEARISH Signal:** The AI model expects the price to FALL by ${abs(diff):.2f}")
                    
        except Exception as e:
            st.error(f"Prediction logic error: {e}")
    else:
        st.info("ℹ️ LSTM Model file (`lstm_stock_model.h5`) not found. Upload it to enable AI predictions.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# === TAB 3: TRADING SIGNALS ===
with tab3:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.subheader("⚡ Technical Signals Dashboard")

    try:
        # RSI Calculation
        delta = stock_df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        # Handle division by zero edge case
        rs.replace([np.inf, -np.inf], 0, inplace=True) 
        
        rsi_series = 100 - (100 / (1 + rs))
        rsi = rsi_series.iloc[-1]

        sma50_val = stock_df['Close'].rolling(50).mean().iloc[-1]
        current_price = stock_df['Close'].iloc[-1]

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
            # Check if SMA is NaN (not enough data)
            if pd.isna(sma50_val):
                st.warning("Not enough data for SMA 50")
            elif current_price > sma50_val:
                st.success(f"✅ Price is ABOVE 50-Day SMA (Uptrend)")
            else:
                st.error(f"🔻 Price is BELOW 50-Day SMA (Downtrend)")
    except Exception as e:
        st.error(f"Error calculating signals: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

# === TAB 4: NEWS ===
with tab4:
    st.subheader(f"📰 Latest News for {ticker}")

    try:
        stock_obj = yf.Ticker(ticker)
        news_list = stock_obj.news

        if not news_list:
            st.info(f"Direct news feed temporarily unavailable for {ticker}.")
            st.markdown(f"👉 **[Click here to read real-time news on Yahoo Finance](https://finance.yahoo.com/quote/{ticker}/news)**")
        else:
            count = 0
            for item in news_list:
                if count >= 5: break
                
                title = item.get('title')
                link = item.get('link')
                publisher = item.get('publisher')
                pub_time = item.get('providerPublishTime')
                
                if title and link:
                    # Formatting time
                    time_str = "Recent"
                    if pub_time:
                        time_str = pd.to_datetime(pub_time, unit='s').strftime('%Y-%m-%d %H:%M')

                    st.markdown(f"""
                    <div style="background-color: white; padding: 15px; border-radius: 10px; margin-bottom: 10px; border: 1px solid #eee;">
                        <h5><a href="{link}" target="_blank" style="text-decoration: none; color: #1a1a1a;">{title}</a></h5>
                        <small style="color: gray;">{publisher} • {time_str}</small>
                    </div>
                    """, unsafe_allow_html=True)
                    count += 1

    except Exception as e:
        st.warning(f"Could not fetch live news: {e}")
        st.markdown(f"👉 **[Click here to read {ticker} News directly](https://finance.yahoo.com/quote/{ticker}/news)**")
