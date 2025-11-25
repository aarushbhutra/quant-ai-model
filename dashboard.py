import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from datetime import datetime, timedelta

# ==========================================
# CONFIGURATION
# ==========================================
st.set_page_config(page_title="Quant AI Backtest & Signals", layout="wide", page_icon="📈")

NIFTY_250 = [
    'RELIANCE.NS', 'HDFCBANK.NS', 'BHARTIARTL.NS', 'ICICIBANK.NS', 'INFY.NS', 'BAJFINANCE.NS',
    'HINDUNILVR.NS', 'LICI.NS', 'LT.NS', 'ITC.NS', 'MARUTI.NS', 'M&M.NS',
    'HCLTECH.NS', 'KOTAKBANK.NS', 'AXISBANK.NS', 'BAJAJFINSV.NS', 'ADANIPORTS.NS', 'NTPC.NS',
    'ONGC.NS', 'HAL.NS', 'BEL.NS', 'ADANIENT.NS', 'ADANIPOWER.NS', 'JSWSTEEL.NS',
    'ASIANPAINT.NS', 'ETERNAL.NS', 'DMART.NS', 'POWERGRID.NS', 'BAJAJ-AUTO.NS', 'NESTLEIND.NS',
    'IOC.NS', 'COALINDIA.NS', 'INDIGO.NS', 'EICHERMOT.NS', 'HINDZINC.NS', 'JIOFIN.NS',
    'HYUNDAI.NS', 'GRASIM.NS', 'DLF.NS', 'LTIM.NS', 'HINDALCO.NS', 'ADANIGREEN.NS',
    'DIVISLAB.NS', 'HDFCLIFE.NS', 'BPCL.NS', 'IRFC.NS', 'PIDILITIND.NS', 'BANKBARODA.NS',
    'MUTHOOTFIN.NS', 'PNB.NS', 'BRITANNIA.NS', 'CHOLAFIN.NS', 'AMBUJACEM.NS', 'CANBK.NS',
    'BAJAJHLDNG.NS', 'CIPLA.NS', 'PFC.NS', 'GAIL.NS', 'CUMMINSIND.NS', 'HEROMOTOCO.NS',
    'LODHA.NS', 'ADANIENSOL.NS', 'BSE.NS', 'HDFCAMC.NS', 'GODREJCP.NS', 'MAXHEALTH.NS',
    'INDIANB.NS', 'POLYCAB.NS', 'CGPOWER.NS', 'MAZDOCK.NS', 'GMRAIRPORT.NS', 'ABB.NS',
    'IDBI.NS', 'BOSCHLTD.NS', 'APOLLOHOSP.NS', 'JINDALSTEL.NS', 'INDUSTOWER.NS', 'INDHOTEL.NS',
    'DRREDDY.NS', 'HINDPETRO.NS', 'ICICIGI.NS', 'BHEL.NS', 'PERSISTENT.NS', 'POWERINDIA.NS',
    'MARICO.NS', 'RECLTD.NS', 'LUPIN.NS', 'MANKIND.NS', 'DABUR.NS', 'DIXON.NS',
    'HAVELLS.NS', 'ICICIPRULI.NS', 'BHARTIHEXA.NS', 'BAJAJHFL.NS', 'NAUKRI.NS', 'ABCAPITAL.NS',
    'ASHOKLEY.NS', 'JSWENERGY.NS', 'POLICYBZR.NS', 'NTPCGREEN.NS', 'PAYTM.NS', 'NHPC.NS',
    'NYKAA.NS', 'IOB.NS', 'GVT&D.NS', 'PRESTIGE.NS', 'LTF.NS', 'OFSS.NS',
    'AUROPHARMA.NS', 'FORTIS.NS', 'OIL.NS', 'AUBANK.NS', 'ALKEM.NS', 'BERGEPAINT.NS',
    'BHARATFORG.NS', 'IDFCFIRSTB.NS', 'COROMANDEL.NS', 'ATGL.NS', 'GICRE.NS', 'BANKINDIA.NS',
    'INDUSINDBK.NS', 'RVNL.NS', 'LLOYDSME.NS', 'MRF.NS', 'NMDC.NS', 'PATANJALI.NS',
    'GODREJPROP.NS', 'ABBOTINDIA.NS', 'JSL.NS', 'PHOENIXLTD.NS', 'FEDERALBNK.NS', 'OBEROIRLTY.NS',
    'COFORGE.NS', 'COLPAL.NS', 'MFSL.NS', 'FACT.NS', 'MOTILALOFS.NS', 'JSWINFRA.NS',
    'NAM-INDIA.NS', 'BDL.NS', 'IRCTC.NS', 'BIOCON.NS', 'LAURUSLABS.NS', 'MPHASIS.NS',
    'GLENMARK.NS', 'PIIND.NS', 'KALYANKJIL.NS', 'LINDEINDIA.NS', 'MCX.NS', 'APLAPOLLO.NS',
    'M&MFIN.NS', '360ONE.NS', 'NATIONALUM.NS', 'HUDCO.NS', 'AIIL.NS', 'LTTS.NS',
    'MAHABANK.NS', 'GODFRYPHLP.NS', 'BALKRISIND.NS', 'COCHINSHIP.NS', 'PREMIERENE.NS', 'RADICO.NS',
    'HEXT.NS', 'ITCHOTELS.NS', 'JKCEMENT.NS', 'PAGEIND.NS', 'GLAXO.NS', 'PGHH.NS',
    'NH.NS', 'PETRONET.NS', 'IREDA.NS', '3MINDIA.NS', 'ESCORTS.NS', 'KAYNES.NS',
    'KEI.NS', 'ASTRAL.NS', 'CONCOR.NS', 'JUBLFOOD.NS', 'FLUOROCHEM.NS', 'ENDURANCE.NS',
    'POONAWALLA.NS', 'DALBHARAT.NS', 'KPRMILL.NS', 'APARINDS.NS', 'IPCALAB.NS', 'BLUESTARCO.NS',
    'AWL.NS', 'GODREJIND.NS', 'CHOLAHLDNG.NS', 'AIAENG.NS', 'CENTRALBK.NS', 'ACC.NS',
    'NLCINDIA.NS', 'CDSL.NS', 'ASTERDM.NS', 'CRISIL.NS', 'MEDANTA.NS', 'APOLLOTYRE.NS',
    'GODIGIT.NS', 'GRSE.NS', 'MSUMI.NS', 'EXIDEIND.NS', 'KPITTECH.NS', 'DELHIVERY.NS',
    'AJANTPHARM.NS', 'NAVINFLUOR.NS', 'NBCC.NS', 'HINDCOPPER.NS', 'HONAUT.NS', 'LICHSGFIN.NS',
    'MRPL.NS', 'GLAND.NS', 'AEGISVOPAK.NS', 'ITI.NS', 'IGL.NS', 'GUJGASLTD.NS',
    'JBCHEPHARM.NS', 'GILLETTE.NS', 'AEGISLOG.NS', 'KIMS.NS', 'ATHERENERG.NS', 'IRB.NS',
    'NUVAMA.NS', 'IKS.NS', 'LALPATHLAB.NS', 'PTCIL.NS', 'EMCURE.NS', 'AMBER.NS',
    'ASAHIINDIA.NS', 'PPLPHARMA.NS', 'ANGELONE.NS', 'HBLENGINE.NS', 'FSL.NS', 'AFFLE.NS',
    'BANDHANBNK.NS', 'KARURVYSYA.NS', 'INOXWIND.NS', 'ANANDRATHI.NS', 'MANAPPURAM.NS', 'EIHOTEL.NS',
    'PNBHOUSING.NS', 'DEEPAKNTR.NS', 'IIFL.NS', 'PFIZER.NS', 'CESC.NS', 'ERIS.NS',
    'JYOTICNC.NS', 'COHANCE.NS', 'ASTRAZEN.NS', 'EMAMILTD.NS'
]


MODEL_PATH = "lstm_trading_model.pth"
SCALER_PATH = "scaler.pkl"
SIGNALS_FILE = "weekly_signals.csv"
PORTFOLIO_FILE = "portfolio.csv"
LOOKBACK_WINDOW = 60
INITIAL_CAPITAL = 100000

# Device Config
device = torch.device("cpu")

# ==========================================
# MODEL CLASS (Must match training)
# ==========================================
class LSTMModel(nn.Module):
    def __init__(self, input_size=7, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# ==========================================
# HELPER FUNCTIONS
# ==========================================
@st.cache_resource
def load_resources():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, None
        
    model = LSTMModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

def get_data(ticker, period=None, start=None):
    for attempt in range(3):
        try:
            if start:
                df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
            else:
                df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            
            if df.empty:
                continue
                
            if isinstance(df.columns, pd.MultiIndex):
                if 'Close' in df.columns.get_level_values(0):
                    df.columns = df.columns.get_level_values(0)
                else:
                    df.columns = df.columns.get_level_values(1)
            
            df['RSI'] = df.ta.rsi(length=14)
            df['EMA_20'] = df.ta.ema(length=20)
            df['EMA_50'] = df.ta.ema(length=50)
            df['ATR'] = df.ta.atr(length=14)
            
            # MACD
            macd = df.ta.macd(fast=12, slow=26, signal=9)
            if macd is not None:
                df['MACD'] = macd.iloc[:, 0]
                
            # CCI
            df['CCI'] = df.ta.cci(length=20)
            
            df.dropna(inplace=True)
            if not df.empty:
                return df
        except Exception:
            pass
    return None

def prepare_sequences(df, scaler):
    df_proc = df.copy()
    df_proc['Log_Ret'] = np.log(df_proc['Close'] / df_proc['Close'].shift(1)) * 100
    df_proc['RSI_Norm'] = df_proc['RSI'] / 100.0
    df_proc['Price_EMA20'] = (df_proc['Close'] / df_proc['EMA_20']) - 1
    df_proc['Price_EMA50'] = (df_proc['Close'] / df_proc['EMA_50']) - 1
    df_proc['ATR_Price'] = df_proc['ATR'] / df_proc['Close']
    df_proc['MACD_Norm'] = df_proc['MACD'] / df_proc['Close']
    df_proc['CCI_Norm'] = df_proc['CCI'] / 100.0
    
    df_proc.dropna(inplace=True)
    
    feature_cols = ['Log_Ret', 'RSI_Norm', 'Price_EMA20', 'Price_EMA50', 'ATR_Price', 'MACD_Norm', 'CCI_Norm']
    
    # Scale
    df_proc[feature_cols] = scaler.transform(df_proc[feature_cols])
    
    X = []
    indices = []
    data_values = df_proc[feature_cols].values
    
    for i in range(LOOKBACK_WINDOW, len(df_proc)):
        X.append(data_values[i-LOOKBACK_WINDOW:i])
        indices.append(df_proc.index[i])
        
    if not X:
        return None, None
        
    return np.array(X, dtype=np.float32), df_proc.loc[indices]

def run_backtest_logic(df, predictions, initial_capital=100000):
    capital = initial_capital
    shares = 0
    in_position = False
    entry_cost = 0
    
    portfolio_values = []
    trades = []
    
    df['Predicted_Log_Ret'] = predictions
    
    for date, row in df.iterrows():
        price = row['Close']
        pred_ret = row['Predicted_Log_Ret']
        
        # Simple Strategy: Buy if predicted return > 0, Sell if < 0
        if pred_ret > 0.0 and not in_position:
            shares = int(capital // price)
            if shares > 0:
                cost = shares * price
                capital -= cost
                in_position = True
                entry_cost = cost
                trades.append({'Date': date, 'Type': 'BUY', 'Price': price, 'Balance': capital})
            
        elif pred_ret < 0 and in_position:
            revenue = shares * price
            capital += revenue
            
            pnl = revenue - entry_cost
            pnl_pct = (pnl / entry_cost) * 100
            
            shares = 0
            in_position = False
            trades.append({'Date': date, 'Type': 'SELL', 'Price': price, 'Balance': capital, 'PnL': pnl, 'PnL_Pct': pnl_pct})
            
        current_val = capital + (shares * price)
        portfolio_values.append(current_val)
        
    return portfolio_values, trades

# ==========================================
# UI LAYOUT
# ==========================================
st.title("🇮🇳 Quant AI: Backtest & Signals")

model, scaler = load_resources()

if model is None:
    st.error("Model or Scaler not found! Please run `train_model.py` first.")
    st.stop()

# Sidebar
st.sidebar.header("Settings")
initial_capital = st.sidebar.number_input("Initial Capital (₹)", min_value=10000, value=100000, step=5000, help="Starting portfolio value for backtesting.")
selected_ticker = st.sidebar.selectbox("Select Stock", ["Custom"] + NIFTY_250)
if selected_ticker == "Custom":
    selected_ticker = st.sidebar.text_input("Enter Ticker", "ITC.NS")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Backtest Analysis", "📢 Weekly Signals", "💼 My Portfolio", "🤖 Bot Simulation"])

# ==========================================
# TAB 1: BACKTEST
# ==========================================
with tab1:
    st.header(f"Backtest Results: {selected_ticker}")
    
    if st.button("Run Backtest"):
        with st.spinner(f"Running simulation for {selected_ticker}..."):
            # Use fixed start date to match backtest_model.py
            df = get_data(selected_ticker, start="2020-01-01") 
            
            if df is not None and len(df) > LOOKBACK_WINDOW:
                # Prepare Data
                X, df_sim = prepare_sequences(df, scaler)
                
                if X is not None:
                    # Predict
                    with torch.no_grad():
                        X_tensor = torch.from_numpy(X).to(device)
                        predictions = model(X_tensor).cpu().numpy().flatten()
                    
                    # Run Logic
                    port_vals, trades = run_backtest_logic(df_sim, predictions, initial_capital)
                    
                    # Calculate Metrics
                    final_val = port_vals[-1]
                    net_profit = final_val - initial_capital
                    ret_pct = (net_profit / initial_capital) * 100
                    
                    bnh_shares = initial_capital / df_sim['Close'].iloc[0]
                    bnh_values = df_sim['Close'] * bnh_shares
                    bnh_ret = ((bnh_values.iloc[-1] - initial_capital) / initial_capital) * 100
                    
                    # Display Metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("AI Strategy Return", f"{ret_pct:.2f}%", f"₹{net_profit:,.0f}")
                    col2.metric("Buy & Hold Return", f"{bnh_ret:.2f}%", f"₹{(bnh_values.iloc[-1] - initial_capital):,.0f}")
                    col3.metric("Total Trades", len(trades))
                    
                    # Plot 0: Stock Price History
                    fig_price = go.Figure()
                    fig_price.add_trace(go.Scatter(x=df_sim.index, y=df_sim['Close'], name="Close Price", line=dict(color='#1f77b4', width=2)))
                    fig_price.update_layout(title=f"{selected_ticker} Price History", xaxis_title="Date", yaxis_title="Price (₹)", height=400)
                    st.plotly_chart(fig_price, use_container_width=True)

                    # Plot 1: Equity Curve
                    fig_equity = go.Figure()
                    fig_equity.add_trace(go.Scatter(x=df_sim.index, y=port_vals, name="AI Strategy", line=dict(color='#00ff00', width=2)))
                    fig_equity.add_trace(go.Scatter(x=df_sim.index, y=bnh_values, name="Buy & Hold", line=dict(color='gray', dash='dash')))
                    fig_equity.update_layout(title="Equity Curve Comparison", xaxis_title="Date", yaxis_title="Portfolio Value (₹)", height=400)
                    st.plotly_chart(fig_equity, use_container_width=True)
                    
                    # Plot 2: Trade Execution
                    fig_trades = go.Figure()
                    fig_trades.add_trace(go.Scatter(x=df_sim.index, y=df_sim['Close'], name="Price", line=dict(color='gray', width=1)))
                    
                    buy_dates = [t['Date'] for t in trades if t['Type'] == 'BUY']
                    buy_prices = [t['Price'] for t in trades if t['Type'] == 'BUY']
                    fig_trades.add_trace(go.Scatter(x=buy_dates, y=buy_prices, mode='markers', name="Buy", marker=dict(color='green', symbol='triangle-up', size=10)))
                    
                    sell_dates = [t['Date'] for t in trades if t['Type'] == 'SELL']
                    sell_prices = [t['Price'] for t in trades if t['Type'] == 'SELL']
                    sell_pnl = [f"PnL: ₹{t['PnL']:.2f} ({t['PnL_Pct']:.2f}%)" for t in trades if t['Type'] == 'SELL']
                    
                    fig_trades.add_trace(go.Scatter(
                        x=sell_dates, 
                        y=sell_prices, 
                        mode='markers', 
                        name="Sell", 
                        marker=dict(color='red', symbol='triangle-down', size=10),
                        text=sell_pnl,
                        hoverinfo='text+x+y'
                    ))
                    
                    fig_trades.update_layout(title="Trade Execution Points (with P&L)", xaxis_title="Date", yaxis_title="Price (₹)", height=400)
                    st.plotly_chart(fig_trades, use_container_width=True)
                    
                    # Plot 3: Technical Indicators (RSI & MACD)
                    st.subheader("Technical Indicators")
                    col_tech1, col_tech2 = st.columns(2)
                    
                    with col_tech1:
                        fig_rsi = go.Figure()
                        fig_rsi.add_trace(go.Scatter(x=df_sim.index, y=df_sim['RSI'], name="RSI", line=dict(color='purple')))
                        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                        fig_rsi.update_layout(title="RSI (14)", height=300, yaxis_range=[0, 100])
                        st.plotly_chart(fig_rsi, use_container_width=True)
                        
                    with col_tech2:
                        fig_macd = go.Figure()
                        # MACD is already in df_sim if we fetched it, but let's re-calc to be sure or use what we have
                        # prepare_sequences doesn't return MACD columns in df_sim usually, it returns the original df slice
                        # df_sim is df.loc[indices], so it HAS the columns from get_data
                        
                        if 'MACD' in df_sim.columns:
                            fig_macd.add_trace(go.Scatter(x=df_sim.index, y=df_sim['MACD'], name="MACD", line=dict(color='blue')))
                            fig_macd.add_hline(y=0, line_dash="dash", line_color="gray")
                            fig_macd.update_layout(title="MACD", height=300)
                            st.plotly_chart(fig_macd, use_container_width=True)
                    
                    # Current Status & Forecast
                    st.subheader("🔮 Forecast for Tomorrow")
                    
                    # Prepare data for next day prediction
                    df_proc = df.copy()
                    df_proc['Log_Ret'] = np.log(df_proc['Close'] / df_proc['Close'].shift(1)) * 100
                    df_proc['RSI_Norm'] = df_proc['RSI'] / 100.0
                    df_proc['Price_EMA20'] = (df_proc['Close'] / df_proc['EMA_20']) - 1
                    df_proc['Price_EMA50'] = (df_proc['Close'] / df_proc['EMA_50']) - 1
                    df_proc['ATR_Price'] = df_proc['ATR'] / df_proc['Close']
                    df_proc['MACD_Norm'] = df_proc['MACD'] / df_proc['Close']
                    df_proc['CCI_Norm'] = df_proc['CCI'] / 100.0
                    
                    df_proc.fillna(method='bfill', inplace=True)
                    df_proc.fillna(0, inplace=True)
                    
                    last_window_raw = df_proc.iloc[-LOOKBACK_WINDOW:].copy()
                    feature_cols = ['Log_Ret', 'RSI_Norm', 'Price_EMA20', 'Price_EMA50', 'ATR_Price', 'MACD_Norm', 'CCI_Norm']
                    last_window_raw[feature_cols] = scaler.transform(last_window_raw[feature_cols])
                    
                    X_next = last_window_raw[feature_cols].values.reshape(1, LOOKBACK_WINDOW, len(feature_cols))
                    X_tensor_next = torch.from_numpy(X_next.astype(np.float32)).to(device)
                    
                    with torch.no_grad():
                        pred_next = model(X_tensor_next).item()
                        
                    # Determine Current Position from Backtest
                    currently_holding = False
                    if trades and trades[-1]['Type'] == 'BUY':
                        currently_holding = True
                        
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Current Position", "HOLDING 🟢" if currently_holding else "FLAT ⚪")
                    c2.metric("Predicted Return (Tomorrow)", f"{pred_next:.4f}%")
                    
                    action = "WAIT"
                    if pred_next > 0:
                        action = "HOLD" if currently_holding else "BUY"
                    else:
                        action = "SELL" if currently_holding else "WAIT"
                        
                    c3.metric("Recommended Action", action)
                    
                    # Trade Log
                    with st.expander("View Trade Log"):
                        st.dataframe(pd.DataFrame(trades))
                else:
                    st.error("Not enough data to generate sequences.")
            else:
                st.error("Could not fetch data or data too short.")

# ==========================================
# TAB 2: WEEKLY SIGNALS
# ==========================================
with tab2:
    st.header("📢 Weekly Market Signals (Nifty 100)")
    
    # Load existing signals if available
    if os.path.exists(SIGNALS_FILE):
        try:
            existing_signals = pd.read_csv(SIGNALS_FILE)
            st.info(f"Loaded cached signals from {SIGNALS_FILE}")
            st.dataframe(existing_signals)
        except:
            pass
            
    st.write("Scanning Nifty 100 for active Buy/Sell signals based on the latest data.")
    
    if st.button("Scan Market for Signals (Force Refresh)"):
        signals = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, ticker in enumerate(NIFTY_250):
            status_text.text(f"Scanning {ticker}...")
            # Use start date to ensure consistent indicators, though we only need recent data
            df = get_data(ticker, start="2020-01-01") 
            
            if df is not None and len(df) > LOOKBACK_WINDOW:
                # 1. Run Backtest Logic to determine CURRENT Position
                # We need to know if we are currently holding the stock to give a valid signal
                
                # Prepare sequences for the whole history (fast enough)
                X_full, df_sim = prepare_sequences(df, scaler)
                
                if X_full is not None:
                    with torch.no_grad():
                        X_tensor_full = torch.from_numpy(X_full).to(device)
                        predictions_full = model(X_tensor_full).cpu().numpy().flatten()
                    
                    # Run logic to find end state
                    _, trades = run_backtest_logic(df_sim, predictions_full)
                    
                    # Determine if currently in position
                    in_position = False
                    if trades:
                        last_trade = trades[-1]
                        if last_trade['Type'] == 'BUY':
                            in_position = True
                            
                    # 2. Get Prediction for TOMORROW (Next Day)
                    # This uses the very last window of data
                    last_window = df_sim.iloc[-LOOKBACK_WINDOW:].copy() # Use df_sim which is processed
                    # Actually prepare_sequences returns X aligned with df_sim.
                    # The last X in X_full predicts the last row of df_sim.
                    # We need to predict the NEXT step, which is not in df_sim yet.
                    
                    # Re-construct the last window from raw df to include the very last day
                    # prepare_sequences cuts off the first 60 days.
                    # We need the window [T-60 : T] to predict T+1.
                    
                    df_proc = df.copy()
                    df_proc['Log_Ret'] = np.log(df_proc['Close'] / df_proc['Close'].shift(1)) * 100
                    df_proc['RSI_Norm'] = df_proc['RSI'] / 100.0
                    df_proc['Price_EMA20'] = (df_proc['Close'] / df_proc['EMA_20']) - 1
                    df_proc['Price_EMA50'] = (df_proc['Close'] / df_proc['EMA_50']) - 1
                    df_proc['ATR_Price'] = df_proc['ATR'] / df_proc['Close']
                    df_proc['MACD_Norm'] = df_proc['MACD'] / df_proc['Close']
                    df_proc['CCI_Norm'] = df_proc['CCI'] / 100.0
                    
                    df_proc.fillna(method='bfill', inplace=True)
                    df_proc.fillna(0, inplace=True)
                    
                    last_window_raw = df_proc.iloc[-LOOKBACK_WINDOW:].copy()
                    feature_cols = ['Log_Ret', 'RSI_Norm', 'Price_EMA20', 'Price_EMA50', 'ATR_Price', 'MACD_Norm', 'CCI_Norm']
                    last_window_raw[feature_cols] = scaler.transform(last_window_raw[feature_cols])
                    
                    X_next = last_window_raw[feature_cols].values.reshape(1, LOOKBACK_WINDOW, len(feature_cols))
                    X_tensor_next = torch.from_numpy(X_next.astype(np.float32)).to(device)
                    
                    with torch.no_grad():
                        pred_next = model(X_tensor_next).item()
                        
                    # 3. Determine Signal based on State + Prediction
                    # Logic must match backtest: Buy if > 0 and NOT in position.
                    
                    signal_type = "NEUTRAL"
                    status = "FLAT"
                    
                    if in_position:
                        status = "HOLDING"
                        if pred_next < 0.0:
                            signal_type = "SELL" # Exit signal
                        else:
                            signal_type = "HOLD" # Stay in
                    else:
                        status = "FLAT"
                        if pred_next > 0.0:
                            signal_type = "BUY" # Entry signal
                        else:
                            signal_type = "WAIT" # Stay out
                            
                    # Filter: Show actionable signals (BUY/SELL)
                    if signal_type in ["BUY", "SELL"]:
                        signals.append({
                            "Date": datetime.now().strftime("%Y-%m-%d"),
                            "Ticker": ticker,
                            "Price": f"₹{df['Close'].iloc[-1]:.2f}",
                            "Forecast": f"{pred_next:.4f}%",
                            "Current Pos": status,
                            "Action": signal_type
                        })
                        
            progress_bar.progress((i + 1) / len(NIFTY_250))
            
        status_text.text("Scan Complete!")
        
        if signals:
            st.success(f"Found {len(signals)} active signals!")
            df_signals = pd.DataFrame(signals)
            st.dataframe(df_signals)
            df_signals.to_csv(SIGNALS_FILE, index=False)
        else:
            st.info("No significant Buy/Sell signals found for this week.")

# ==========================================
# TAB 3: PORTFOLIO
# ==========================================
with tab3:
    st.header("💼 My Portfolio Tracker")
    
    INITIAL_CASH_BALANCE = 1200000
    
    # Load Portfolio
    if os.path.exists(PORTFOLIO_FILE):
        portfolio_df = pd.read_csv(PORTFOLIO_FILE)
        # Ensure Buy_Date exists (backward compatibility)
        if 'Buy_Date' not in portfolio_df.columns:
            portfolio_df['Buy_Date'] = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    else:
        portfolio_df = pd.DataFrame(columns=["Ticker", "Quantity", "Buy_Price", "Buy_Date"])
        
    # Add New Position
    with st.expander("Add New Position"):
        with st.form("add_pos_form"):
            c1, c2 = st.columns(2)
            with c1:
                p_ticker = st.selectbox("Ticker", NIFTY_250)
                p_qty = st.number_input("Quantity", min_value=1, value=10)
            with c2:
                p_price = st.number_input("Avg Buy Price", min_value=0.0, value=1000.0)
                p_date = st.date_input("Buy Date", datetime.now())
            
            if st.form_submit_button("Add to Portfolio"):
                new_row = pd.DataFrame({
                    "Ticker": [p_ticker], 
                    "Quantity": [p_qty], 
                    "Buy_Price": [p_price],
                    "Buy_Date": [p_date.strftime('%Y-%m-%d')]
                })
                portfolio_df = pd.concat([portfolio_df, new_row], ignore_index=True)
                portfolio_df.to_csv(PORTFOLIO_FILE, index=False)
                st.success(f"Added {p_ticker} to portfolio!")
                st.rerun()
                
    # Display Portfolio
    if not portfolio_df.empty:
        st.subheader("Current Holdings")
        
        # Calculate Live Values & History
        total_invested = 0.0
        total_current_val = 0.0
        
        portfolio_data = []
        
        # Initialize Time Series for Graphs (Last 6 Months)
        # We need a reference date range first. Let's fetch one stock to get the index.
        ref_ticker = "RELIANCE.NS" # Proxy
        ref_dat = yf.Ticker(ref_ticker)
        ref_hist = ref_dat.history(period="6mo", auto_adjust=True)
        
        if not ref_hist.empty:
            # Ensure timezone naive for consistency
            ref_hist.index = ref_hist.index.tz_localize(None)
            full_date_index = ref_hist.index
            # Initialize Series with Initial Cash
            daily_portfolio_value = pd.Series(INITIAL_CASH_BALANCE, index=full_date_index)
            daily_cash_balance = pd.Series(INITIAL_CASH_BALANCE, index=full_date_index)
        else:
            daily_portfolio_value = None
            daily_cash_balance = None
        
        individual_histories = {} # {ticker: df_history}
        
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        # Iterate with index to allow deletion
        rows_to_drop = []
        
        for index, row in portfolio_df.iterrows():
            ticker = row['Ticker']
            qty = float(row['Quantity'])
            buy_price = float(row['Buy_Price'])
            buy_date_str = str(row['Buy_Date'])
            
            try:
                buy_date_ts = pd.Timestamp(buy_date_str).tz_localize(None)
            except:
                buy_date_ts = pd.Timestamp(datetime.now().date()).tz_localize(None)

            progress_text.text(f"Fetching data for {ticker}...")
            progress_bar.progress((index + 1) / len(portfolio_df))
            
            try:
                # Fetch 6 months history for graphs
                dat = yf.Ticker(ticker)
                df_live = dat.history(period="6mo", auto_adjust=True)
                
                if df_live.empty:
                    continue
                
                # Ensure timezone naive for comparison
                df_live.index = df_live.index.tz_localize(None)
                
                current_price = float(df_live['Close'].iloc[-1])
                
                invested_val = qty * buy_price
                current_val = qty * current_price
                pl = current_val - invested_val
                pl_pct = (pl / invested_val) * 100
                
                total_invested += invested_val
                total_current_val += current_val
                
                # --- Graph Logic ---
                if daily_portfolio_value is not None:
                    # Align df_live to master index
                    aligned_close = df_live['Close'].reindex(daily_portfolio_value.index, method='ffill').fillna(0)
                    
                    # Create mask for dates >= Buy_Date
                    mask = daily_portfolio_value.index >= buy_date_ts
                    
                    # 1. Update Cash Balance: Subtract Cost for days we held the stock
                    # Cash = Initial - Cost_of_Held_Stocks
                    cost_series = pd.Series(0.0, index=daily_portfolio_value.index)
                    cost_series[mask] = invested_val
                    daily_cash_balance = daily_cash_balance - cost_series
                    
                    # 2. Update Portfolio Value: Add (Current_Val - Cost) to Base (which is Cash)
                    # Actually: Portfolio_Value = Cash_Balance + Stock_Value
                    # We can just sum Stock Values separately and add to Cash Balance later?
                    # Let's do: Total_Value = Initial_Cash + Sum(PnL_of_each_stock)
                    # PnL = (Price - Buy_Price) * Qty (only after Buy_Date)
                    
                    pnl_series = (aligned_close - buy_price) * qty
                    pnl_series[~mask] = 0.0 # 0 PnL before buying
                    
                    # Add this stock's PnL to the total portfolio value (which started at 1.2M)
                    daily_portfolio_value = daily_portfolio_value + pnl_series
                    
                    # Store individual PnL for expander
                    individual_histories[ticker] = pnl_series

                portfolio_data.append({
                    "Index": index, # For deletion
                    "Ticker": ticker,
                    "Quantity": qty,
                    "Buy Price": f"₹{buy_price:.2f}",
                    "Current Price": f"₹{current_price:.2f}",
                    "Invested": f"₹{invested_val:,.0f}",
                    "Current Value": f"₹{current_val:,.0f}",
                    "P/L": f"₹{pl:,.0f}",
                    "P/L %": f"{pl_pct:.2f}%",
                    "Buy Date": buy_date_str
                })
                    
            except Exception as e:
                st.error(f"Error fetching {ticker}: {e}")
        
        progress_text.empty()
        progress_bar.empty()
        
        # 1. Overall P/L Graph (Cash + Holdings)
        if daily_portfolio_value is not None:
            # Filter graph to start from the first buy date
            start_date_ts = None
            if not portfolio_df.empty and 'Buy_Date' in portfolio_df.columns:
                try:
                    # Convert to datetime if not already
                    portfolio_df['Buy_Date'] = pd.to_datetime(portfolio_df['Buy_Date'])
                    min_date = portfolio_df['Buy_Date'].min()
                    # Localize/De-localize to match index
                    start_date_ts = pd.Timestamp(min_date).tz_localize(None)
                except Exception as e:
                    st.error(f"Date parsing error: {e}")
            
            # Apply Filter
            if start_date_ts is not None:
                filtered_val = daily_portfolio_value[daily_portfolio_value.index >= start_date_ts]
                if not filtered_val.empty:
                    daily_portfolio_value = filtered_val
                    daily_cash_balance = daily_cash_balance[daily_cash_balance.index >= start_date_ts]
                else:
                    # Fallback if filter removes everything (e.g. Buy Date > Last Market Date)
                    # This happens if bought on weekend/today and data isn't updated yet.
                    # We'll show the last 5 days to avoid an empty graph.
                    daily_portfolio_value = daily_portfolio_value.tail(5)
                    daily_cash_balance = daily_cash_balance.tail(5)

            st.subheader("Overall Portfolio Performance")
            fig_overall = go.Figure()
            
            # Total Value
            fig_overall.add_trace(go.Scatter(
                x=daily_portfolio_value.index, 
                y=daily_portfolio_value.values, 
                mode='lines', 
                name='Total Portfolio Value',
                line=dict(color='#00ff00', width=2),
                fill=None # Explicitly no fill
            ))
            
            # Cash Balance
            fig_overall.add_trace(go.Scatter(
                x=daily_cash_balance.index, 
                y=daily_cash_balance.values, 
                mode='lines', 
                name='Cash Balance',
                line=dict(color='gray', dash='dash')
            ))
            
            final_val = daily_portfolio_value.iloc[-1] if not daily_portfolio_value.empty else INITIAL_CASH_BALANCE
            
            fig_overall.update_layout(
                title=f"Total Value: ₹{final_val:,.0f} (Initial: ₹{INITIAL_CASH_BALANCE:,.0f})",
                xaxis_title="Date",
                yaxis_title="Value (₹)",
                height=400,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_overall, use_container_width=True)

        # 2. Summary Table & Sell Buttons
        # Custom table with Sell buttons
        st.write("### Holdings")
        
        # Header
        cols = st.columns([1.5, 1, 1.5, 1.5, 1.5, 1.5, 1.5, 1])
        headers = ["Ticker", "Qty", "Buy Price", "Cur Price", "Invested", "Cur Value", "P/L", "Action"]
        for col, h in zip(cols, headers):
            col.markdown(f"**{h}**")
            
        for item in portfolio_data:
            cols = st.columns([1.5, 1, 1.5, 1.5, 1.5, 1.5, 1.5, 1])
            cols[0].write(item['Ticker'])
            cols[1].write(item['Quantity'])
            cols[2].write(item['Buy Price'])
            cols[3].write(item['Current Price'])
            cols[4].write(item['Invested'])
            cols[5].write(item['Current Value'])
            
            # Color P/L
            pl_val = float(item['P/L'].replace('₹', '').replace(',', ''))
            color = "green" if pl_val >= 0 else "red"
            cols[6].markdown(f":{color}[{item['P/L']}]")
            
            if cols[7].button("Sell", key=f"sell_{item['Index']}"):
                # Remove from dataframe
                portfolio_df = portfolio_df.drop(item['Index'])
                portfolio_df.to_csv(PORTFOLIO_FILE, index=False)
                st.success(f"Sold {item['Ticker']}!")
                st.rerun()
        
        # Summary Metrics
        current_cash = INITIAL_CASH_BALANCE - total_invested
        total_portfolio_val = current_cash + total_current_val
        total_pl = total_portfolio_val - INITIAL_CASH_BALANCE
        total_pl_pct = (total_pl / INITIAL_CASH_BALANCE) * 100
        
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cash Balance", f"₹{current_cash:,.0f}")
        c2.metric("Invested Amount", f"₹{total_invested:,.0f}")
        c3.metric("Total Portfolio Value", f"₹{total_portfolio_val:,.0f}")
        c4.metric("Net Profit/Loss", f"₹{total_pl:,.0f}", f"{total_pl_pct:.2f}%")
        
        if st.button("Clear Entire Portfolio"):
            if os.path.exists(PORTFOLIO_FILE):
                os.remove(PORTFOLIO_FILE)
                st.rerun()
                
        # 3. Individual Stock Graphs (Dropdowns)
        st.subheader("Individual Stock Performance")
        for p_item in portfolio_data:
            ticker = p_item['Ticker']
            if ticker in individual_histories:
                pnl_series = individual_histories[ticker]
                curr_pnl = pnl_series.iloc[-1]
                color_emoji = "🟢" if curr_pnl >= 0 else "🔴"
                
                with st.expander(f"{ticker} {color_emoji} P/L: {p_item['P/L']} ({p_item['P/L %']})"):
                    fig_ind = go.Figure()
                    color = 'green' if curr_pnl >= 0 else 'red'
                    
                    fig_ind.add_trace(go.Scatter(
                        x=pnl_series.index,
                        y=pnl_series.values,
                        mode='lines',
                        name='P/L',
                        line=dict(color=color)
                    ))
                    
                    fig_ind.update_layout(
                        title=f"{ticker} Profit/Loss Trend (Since {p_item['Buy Date']})",
                        xaxis_title="Date",
                        yaxis_title="P/L (₹)",
                        height=300,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig_ind, use_container_width=True)

    else:
        st.info("Your portfolio is empty. Add stocks above.")

# ==========================================
# TAB 4: BOT SIMULATION
# ==========================================
with tab4:
    st.header("🤖 Autonomous AI Bot Strategy")
    st.markdown("""
    This bot autonomously trades the entire **NIFTY 250** universe. 
    It dynamically manages risk, position sizing, and portfolio rebalancing based on AI confidence and market volatility.
    **Self-Correction:** The bot adjusts its risk appetite based on its recent win rate.
    """)
    
    col_b1, col_b2 = st.columns(2)
    
    with col_b1:
        bot_capital = st.number_input("Initial Bot Capital (₹)", value=500000, step=50000)
        
    with col_b2:
        bot_start_date = st.date_input("Start Date", datetime(2023, 1, 1))
        bot_end_date = st.date_input("End Date", datetime.now())

    if st.button("🚀 Run Autonomous Bot"):
        status_area = st.empty()
        progress_bar = st.progress(0)
        
        # 1. Pre-fetch and Pre-predict (NIFTY 250)
        status_area.text("Scanning NIFTY 250 Universe... This may take a moment.")
        
        market_data = {} # {ticker: df}
        predictions_map = {} # {ticker: Series of predictions}
        valid_tickers = []
        
        # Limit to top 50 for speed if needed, but user asked for ALL. 
        # We will try all but handle errors gracefully.
        universe = NIFTY_250 
        
        for i, t in enumerate(universe):
            # Optimization: Only fetch if we don't have it or if it's fast
            # We use a shorter lookback to speed up if possible, but we need history for indicators
            df = get_data(t, start=str(bot_start_date - timedelta(days=150))) 
            
            if df is not None and len(df) > LOOKBACK_WINDOW:
                X, df_sim = prepare_sequences(df, scaler)
                if X is not None:
                    with torch.no_grad():
                        X_tensor = torch.from_numpy(X).to(device)
                        preds = model(X_tensor).cpu().numpy().flatten()
                    
                    # Align predictions with dates
                    pred_series = pd.Series(preds, index=df_sim.index)
                    
                    # Store
                    market_data[t] = df_sim
                    predictions_map[t] = pred_series
                    valid_tickers.append(t)
            
            progress_bar.progress((i + 1) / len(universe) * 0.3)
        
        if not valid_tickers:
            st.error("No valid data found for NIFTY 250 tickers.")
            st.stop()
            
        # 2. Simulation Loop
        status_area.text("Simulating trading days with Adaptive Risk Management...")
        
        # Create a master timeline
        all_dates = sorted(list(set().union(*[df.index for df in market_data.values()])))
        sim_dates = [d for d in all_dates if pd.Timestamp(bot_start_date) <= d <= pd.Timestamp(bot_end_date)]
        
        cash = bot_capital
        portfolio = {} # {ticker: {'qty': int, 'entry_price': float, 'stop_loss': float}}
        history = [] 
        trade_log = []
        
        # Adaptive Risk Parameters
        current_risk_per_trade = 0.02 # Start with 2% risk
        win_history = [] # Track last 20 trades [1, 0, 1, 1...]
        
        for i, current_date in enumerate(sim_dates):
            # --- ADAPTIVE LOGIC ---
            # Calculate Win Rate of last 20 trades
            if len(win_history) > 20:
                win_history = win_history[-20:]
            
            if len(win_history) >= 5:
                win_rate = sum(win_history) / len(win_history)
                # Self-Correction:
                if win_rate > 0.6: # Winning streak -> Increase aggression
                    current_risk_per_trade = min(0.05, current_risk_per_trade * 1.05)
                elif win_rate < 0.4: # Losing streak -> Decrease risk
                    current_risk_per_trade = max(0.005, current_risk_per_trade * 0.95)
            
            # Update Portfolio Value
            holdings_value = 0
            
            # Check Exits
            for t in list(portfolio.keys()):
                if current_date in market_data[t].index:
                    price = market_data[t].loc[current_date, 'Close']
                    pred = predictions_map[t].get(current_date, 0)
                    pos = portfolio[t]
                    
                    # Exit Conditions
                    is_stop_hit = price < pos['stop_loss']
                    is_sell_signal = pred < 0
                    
                    if is_sell_signal or is_stop_hit:
                        # Sell
                        revenue = pos['qty'] * price
                        cash += revenue
                        
                        pnl = revenue - (pos['qty'] * pos['entry_price'])
                        
                        # Record Win/Loss for Adaptation
                        win_history.append(1 if pnl > 0 else 0)
                        
                        reason = "Stop Loss" if is_stop_hit else "AI Signal"
                        trade_log.append({
                            "Date": current_date,
                            "Ticker": t,
                            "Action": "SELL",
                            "Price": price,
                            "Qty": pos['qty'],
                            "PnL": pnl,
                            "Reason": reason
                        })
                        del portfolio[t]
                    else:
                        holdings_value += pos['qty'] * price
                        # Trailing Stop Logic (Optional but good for "maximizing return")
                        # Move SL to breakeven if price moves up 5%
                        if price > pos['entry_price'] * 1.05 and pos['stop_loss'] < pos['entry_price']:
                             portfolio[t]['stop_loss'] = pos['entry_price']
                else:
                    # No data today, assume price holds
                    if t in portfolio:
                        # Use last known price roughly for valuation
                        holdings_value += portfolio[t]['qty'] * portfolio[t]['entry_price']

            current_port_value = cash + holdings_value
            
            # Check Entries
            # Dynamic Max Positions based on Capital (e.g. 1 position per 50k, max 20)
            dynamic_max_pos = min(20, max(5, int(current_port_value / 50000)))
            
            if len(portfolio) < dynamic_max_pos:
                candidates = []
                for t in valid_tickers:
                    if t not in portfolio and current_date in market_data[t].index:
                        pred = predictions_map[t].get(current_date, 0)
                        if pred > 0:
                            candidates.append((t, pred))
                
                # Sort by AI Confidence (Predicted Return)
                candidates.sort(key=lambda x: x[1], reverse=True)
                
                slots_available = dynamic_max_pos - len(portfolio)
                
                for t, pred in candidates[:slots_available]:
                    row = market_data[t].loc[current_date]
                    price = row['Close']
                    atr = row['ATR']
                    
                    if atr > 0:
                        # Volatility Sizing: Risk fixed % of equity based on ATR distance
                        stop_distance = atr * 2.0 # Standard 2 ATR stop
                        risk_amount = current_port_value * current_risk_per_trade
                        
                        shares_to_buy = int(risk_amount // stop_distance)
                        
                        # Cap max allocation to 20% of portfolio to prevent "dumping into one"
                        max_cost = current_port_value * 0.20
                        cost = shares_to_buy * price
                        
                        if cost > max_cost:
                            shares_to_buy = int(max_cost // price)
                            
                        if shares_to_buy > 0 and cash >= (shares_to_buy * price):
                            cost = shares_to_buy * price
                            cash -= cost
                            stop_price = price - stop_distance
                            
                            portfolio[t] = {
                                'qty': shares_to_buy,
                                'entry_price': price,
                                'stop_loss': stop_price
                            }
                            
                            trade_log.append({
                                "Date": current_date,
                                "Ticker": t,
                                "Action": "BUY",
                                "Price": price,
                                "Qty": shares_to_buy,
                                "PnL": 0,
                                "Reason": f"Pred: {pred:.2f}%"
                            })
            
            # Record History
            history.append({
                "Date": current_date,
                "Portfolio Value": current_port_value,
                "Cash": cash,
                "Risk Level": current_risk_per_trade * 100 # Log the adaptive risk
            })
            
            progress_bar.progress(0.3 + (0.7 * (i + 1) / len(sim_dates)))
        
        status_area.empty()
        
        # Results
        if not history:
            st.error("No simulation data generated.")
        else:
            df_hist = pd.DataFrame(history).set_index("Date")
            df_trades = pd.DataFrame(trade_log)
            
            final_val = df_hist['Portfolio Value'].iloc[-1]
            total_ret = final_val - bot_capital
            total_ret_pct = (total_ret / bot_capital) * 100
            
            st.subheader("📊 Autonomous Bot Performance")
            m1, m2, m3 = st.columns(3)
            m1.metric("Final Portfolio Value", f"₹{final_val:,.0f}")
            m2.metric("Total Return", f"{total_ret_pct:.2f}%", f"₹{total_ret:,.0f}")
            m3.metric("Total Trades", len(df_trades))
            
            # Plot Equity Curve
            fig_bot = go.Figure()
            fig_bot.add_trace(go.Scatter(x=df_hist.index, y=df_hist['Portfolio Value'], name="Portfolio Value", line=dict(color='#00ff00', width=2)))
            fig_bot.add_trace(go.Scatter(x=df_hist.index, y=df_hist['Cash'], name="Cash Held", line=dict(color='gray', dash='dot')))
            fig_bot.update_layout(title="Bot Portfolio Performance", xaxis_title="Date", yaxis_title="Value (₹)", height=450)
            st.plotly_chart(fig_bot, use_container_width=True)
            
            # Plot Risk Adaptation
            fig_risk = go.Figure()
            fig_risk.add_trace(go.Scatter(x=df_hist.index, y=df_hist['Risk Level'], name="Risk % Per Trade", line=dict(color='orange')))
            fig_risk.update_layout(title="Adaptive Risk Level (Self-Correction)", xaxis_title="Date", yaxis_title="Risk %", height=300)
            st.plotly_chart(fig_risk, use_container_width=True)
            
            # Trade Log
            st.subheader("📜 Trade History")
            if not df_trades.empty:
                st.dataframe(df_trades)
            else:
                st.info("No trades were executed.")
