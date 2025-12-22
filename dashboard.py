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
import json
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
PORTFOLIO_STATE_FILE = "portfolio_state.json"
LOOKBACK_WINDOW = 60
INITIAL_CAPITAL = 100000

# Device Config
device = torch.device("cpu")

# ==========================================
# PORTFOLIO STATE PERSISTENCE
# ==========================================
def load_portfolio_state():
    """Load portfolio state from JSON file."""
    default_state = {
        "initial_cash_balance": 1200000,
        "realized_pnl": 0.0,
        "trade_history": []
    }
    if os.path.exists(PORTFOLIO_STATE_FILE):
        try:
            with open(PORTFOLIO_STATE_FILE, 'r') as f:
                state = json.load(f)
                # Ensure all keys exist (backward compat)
                for key in default_state:
                    if key not in state:
                        state[key] = default_state[key]
                return state
        except:
            return default_state
    return default_state

def save_portfolio_state(state):
    """Save portfolio state to JSON file."""
    with open(PORTFOLIO_STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)

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

@st.cache_data(show_spinner=False)
def get_portfolio_signal(ticker):
    """Return AI forecast/action for a portfolio holding."""
    df = get_data(ticker, start="2020-01-01")
    if df is None or len(df) <= LOOKBACK_WINDOW:
        return None
    df_proc = df.copy()
    df_proc['Log_Ret'] = np.log(df_proc['Close'] / df_proc['Close'].shift(1)) * 100
    df_proc['RSI_Norm'] = df_proc['RSI'] / 100.0
    df_proc['Price_EMA20'] = (df_proc['Close'] / df_proc['EMA_20']) - 1
    df_proc['Price_EMA50'] = (df_proc['Close'] / df_proc['EMA_50']) - 1
    df_proc['ATR_Price'] = df_proc['ATR'] / df_proc['Close']
    df_proc['MACD_Norm'] = df_proc['MACD'] / df_proc['Close']
    df_proc['CCI_Norm'] = df_proc['CCI'] / 100.0
    df_proc.dropna(inplace=True)
    if len(df_proc) <= LOOKBACK_WINDOW:
        return None
    feature_cols = ['Log_Ret', 'RSI_Norm', 'Price_EMA20', 'Price_EMA50', 'ATR_Price', 'MACD_Norm', 'CCI_Norm']
    try:
        df_proc[feature_cols] = scaler.transform(df_proc[feature_cols])
    except Exception:
        return None
    last_window = df_proc.iloc[-LOOKBACK_WINDOW:].copy()
    X_next = last_window[feature_cols].values.reshape(1, LOOKBACK_WINDOW, len(feature_cols)).astype(np.float32)
    X_tensor = torch.from_numpy(X_next).to(device)
    with torch.no_grad():
        pred_next = float(model(X_tensor).item())
    action = "HOLD" if pred_next > 0 else "SELL"
    return {"prediction": pred_next, "action": action}

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

# Tabs - Reordered: Portfolio first, then Signals, then Testing
tab1, tab2, tab3 = st.tabs(["💼 My Portfolio", "📢 Weekly Signals", "🧪 Testing"])

# ==========================================
# TAB 1: MY PORTFOLIO (Now First)
# ==========================================
with tab1:
    st.header("💼 My Portfolio Tracker")
    
    # Load persisted state
    portfolio_state = load_portfolio_state()
    INITIAL_CASH_BALANCE = portfolio_state["initial_cash_balance"]
    realized_pnl = portfolio_state["realized_pnl"]
    trade_history = portfolio_state["trade_history"]
    
    # Load Portfolio
    if os.path.exists(PORTFOLIO_FILE):
        portfolio_df = pd.read_csv(PORTFOLIO_FILE)
        if 'Buy_Date' not in portfolio_df.columns:
            portfolio_df['Buy_Date'] = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    else:
        portfolio_df = pd.DataFrame(columns=["Ticker", "Quantity", "Buy_Price", "Buy_Date"])
    
    # Settings Expander
    with st.expander("⚙️ Portfolio Settings"):
        new_initial = st.number_input("Initial Cash Balance (₹)", value=INITIAL_CASH_BALANCE, step=50000, key="port_init_balance")
        if st.button("Update Initial Balance", key="btn_update_balance"):
            portfolio_state["initial_cash_balance"] = new_initial
            save_portfolio_state(portfolio_state)
            st.success("Initial balance updated!")
            st.rerun()
        
        st.write(f"**Realized P&L (from sold stocks):** ₹{realized_pnl:,.2f}")
        
        if trade_history:
            st.write("**Recent Trade History:**")
            st.dataframe(pd.DataFrame(trade_history[-10:]))
        
    # Add New Position
    with st.expander("➕ Add New Position"):
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
        st.subheader("📊 Current Holdings")
        
        # Calculate Live Values & History
        total_invested = 0.0
        total_current_val = 0.0
        
        portfolio_data = []
        
        # Initialize Time Series for Graphs
        ref_ticker = "RELIANCE.NS"
        ref_dat = yf.Ticker(ref_ticker)
        ref_hist = ref_dat.history(period="6mo", auto_adjust=True)
        
        if not ref_hist.empty:
            ref_hist.index = ref_hist.index.tz_localize(None)
            full_date_index = ref_hist.index
            daily_portfolio_value = pd.Series(INITIAL_CASH_BALANCE, index=full_date_index)
            daily_invested_value = pd.Series(0.0, index=full_date_index)  # Track invested amount over time
        else:
            daily_portfolio_value = None
            daily_invested_value = None
        
        individual_histories = {}
        
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
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
                dat = yf.Ticker(ticker)
                df_live = dat.history(period="6mo", auto_adjust=True)
                
                if df_live.empty:
                    continue
                
                df_live.index = df_live.index.tz_localize(None)
                current_price = float(df_live['Close'].iloc[-1])
                
                invested_val = qty * buy_price
                current_val = qty * current_price
                pl = current_val - invested_val
                pl_pct = (pl / invested_val) * 100
                signal_info = get_portfolio_signal(ticker)
                ai_action = signal_info['action'] if signal_info else "N/A"
                ai_forecast = f"{signal_info['prediction']:.4f}%" if signal_info else "N/A"
                
                total_invested += invested_val
                total_current_val += current_val
                
                # Graph Logic
                if daily_portfolio_value is not None:
                    aligned_close = df_live['Close'].reindex(daily_portfolio_value.index, method='ffill').ffill().bfill().fillna(buy_price)
                    mask = daily_portfolio_value.index >= buy_date_ts
                    
                    # Track invested amount
                    invested_series = pd.Series(0.0, index=daily_portfolio_value.index)
                    invested_series[mask] = invested_val
                    daily_invested_value = daily_invested_value + invested_series
                    
                    # Track stock value
                    stock_value_series = aligned_close * qty
                    stock_value_series[~mask] = 0.0
                    
                    # PnL series for individual graph
                    pnl_series = (aligned_close - buy_price) * qty
                    pnl_series[~mask] = 0.0
                    
                    daily_portfolio_value = daily_portfolio_value + pnl_series
                    individual_histories[ticker] = pnl_series

                portfolio_data.append({
                    "Index": index,
                    "Ticker": ticker,
                    "Quantity": qty,
                    "Buy Price": f"₹{buy_price:.2f}",
                    "Current Price": f"₹{current_price:.2f}",
                    "Invested": f"₹{invested_val:,.0f}",
                    "Current Value": f"₹{current_val:,.0f}",
                    "P/L": f"₹{pl:,.0f}",
                    "P/L %": f"{pl_pct:.2f}%",
                    "AI Forecast": ai_forecast,
                    "AI Action": ai_action,
                    "Buy Date": buy_date_str
                })
                    
            except Exception as e:
                st.error(f"Error fetching {ticker}: {e}")
        
        progress_text.empty()
        progress_bar.empty()
        
        # Overall P/L Graph - IMPROVED with 3 lines
        if daily_portfolio_value is not None and not portfolio_df.empty and 'Buy_Date' in portfolio_df.columns:
            try:
                portfolio_df['Buy_Date'] = pd.to_datetime(portfolio_df['Buy_Date'])
                min_date = portfolio_df['Buy_Date'].min()
                start_date_ts = pd.Timestamp(min_date).tz_localize(None)
                
                filtered_val = daily_portfolio_value[daily_portfolio_value.index >= start_date_ts]
                if not filtered_val.empty:
                    daily_portfolio_value = filtered_val
                    daily_invested_value = daily_invested_value[daily_invested_value.index >= start_date_ts]
            except:
                pass

            st.subheader("📈 Portfolio Growth Over Time")
            
            # Calculate Cash Balance (Initial - Invested + Realized PnL)
            daily_cash = INITIAL_CASH_BALANCE - daily_invested_value + realized_pnl
            # Total = Cash + Holdings Value = Initial + Unrealized PnL + Realized PnL
            # daily_portfolio_value already = Initial + Unrealized PnL (sum of pnl_series)
            daily_total = daily_portfolio_value + realized_pnl
            
            fig_overall = go.Figure()
            
            # Total Portfolio Value (Green)
            fig_overall.add_trace(go.Scatter(
                x=daily_total.index, 
                y=daily_total.values, 
                mode='lines', 
                name='Total Value',
                line=dict(color='#00ff00', width=2.5)
            ))
            
            # Invested Amount (Blue)
            fig_overall.add_trace(go.Scatter(
                x=daily_invested_value.index, 
                y=daily_invested_value.values, 
                mode='lines', 
                name='Invested Amount',
                line=dict(color='#1f77b4', width=2)
            ))
            
            # Cash Balance (Gray dashed)
            fig_overall.add_trace(go.Scatter(
                x=daily_cash.index, 
                y=daily_cash.values, 
                mode='lines', 
                name='Cash Balance',
                line=dict(color='gray', dash='dash', width=1.5)
            ))
            
            # Initial Balance Reference Line
            fig_overall.add_hline(y=INITIAL_CASH_BALANCE, line_dash="dot", line_color="orange", 
                                  annotation_text=f"Initial: ₹{INITIAL_CASH_BALANCE:,.0f}")
            
            final_val = daily_total.iloc[-1] if not daily_total.empty else INITIAL_CASH_BALANCE
            growth = final_val - INITIAL_CASH_BALANCE
            growth_pct = (growth / INITIAL_CASH_BALANCE) * 100
            
            fig_overall.update_layout(
                title=f"Total Value: ₹{final_val:,.0f} | Growth: ₹{growth:,.0f} ({growth_pct:.2f}%)",
                xaxis_title="Date",
                yaxis_title="Value (₹)",
                height=450,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=60, b=20)
            )
            st.plotly_chart(fig_overall, width='stretch')

        # Holdings Table with Sell Buttons
        st.write("### 📋 Holdings")
        
        cols = st.columns([1.4, 0.9, 1.3, 1.3, 1.3, 1.3, 1.2, 1.3, 0.9])
        headers = ["Ticker", "Qty", "Buy Price", "Cur Price", "Invested", "Cur Value", "P/L", "AI Signal", "Action"]
        for col, h in zip(cols, headers):
            col.markdown(f"**{h}**")
            
        for item in portfolio_data:
            cols = st.columns([1.4, 0.9, 1.3, 1.3, 1.3, 1.3, 1.2, 1.3, 0.9])
            cols[0].write(item['Ticker'])
            cols[1].write(item['Quantity'])
            cols[2].write(item['Buy Price'])
            cols[3].write(item['Current Price'])
            cols[4].write(item['Invested'])
            cols[5].write(item['Current Value'])
            
            pl_val = float(item['P/L'].replace('₹', '').replace(',', ''))
            color = "green" if pl_val >= 0 else "red"
            cols[6].markdown(f":{color}[{item['P/L']}]")
            
            ai_color = "green" if item['AI Action'] == "HOLD" else "red"
            if item['AI Action'] != "N/A":
                signal_text = f":{ai_color}[{item['AI Action']}]\n({item['AI Forecast']})"
            else:
                signal_text = "N/A"
            cols[7].markdown(signal_text)
            
            if cols[8].button("Sell", key=f"sell_{item['Index']}"):
                sell_pnl = float(item['P/L'].replace('₹', '').replace(',', ''))
                sell_price = float(item['Current Price'].replace('₹', '').replace(',', ''))
                
                trade_record = {
                    "date": datetime.now().strftime('%Y-%m-%d %H:%M'),
                    "ticker": item['Ticker'],
                    "action": "SELL",
                    "qty": item['Quantity'],
                    "buy_price": item['Buy Price'],
                    "sell_price": f"₹{sell_price:.2f}",
                    "pnl": sell_pnl
                }
                portfolio_state["trade_history"].append(trade_record)
                portfolio_state["realized_pnl"] += sell_pnl
                save_portfolio_state(portfolio_state)
                
                portfolio_df = portfolio_df.drop(item['Index'])
                portfolio_df.to_csv(PORTFOLIO_FILE, index=False)
                st.success(f"Sold {item['Ticker']}! P&L: ₹{sell_pnl:,.2f}")
                st.rerun()
        
        # Summary Metrics
        current_cash = INITIAL_CASH_BALANCE - total_invested + realized_pnl
        total_portfolio_val = current_cash + total_current_val
        total_pl = total_portfolio_val - INITIAL_CASH_BALANCE
        total_pl_pct = (total_pl / INITIAL_CASH_BALANCE) * 100
        
        st.markdown("---")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("💰 Cash Balance", f"₹{current_cash:,.0f}")
        c2.metric("📊 Invested", f"₹{total_invested:,.0f}")
        c3.metric("📈 Portfolio Value", f"₹{total_portfolio_val:,.0f}")
        c4.metric("✅ Realized P&L", f"₹{realized_pnl:,.0f}")
        c5.metric("📊 Net P/L", f"₹{total_pl:,.0f}", f"{total_pl_pct:.2f}%")
        
        if st.button("🗑️ Clear Entire Portfolio", key="btn_clear_portfolio"):
            if os.path.exists(PORTFOLIO_FILE):
                os.remove(PORTFOLIO_FILE)
                st.rerun()
                
        # Individual Stock Graphs
        st.subheader("📉 Individual Stock Performance")
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
                    st.plotly_chart(fig_ind, width='stretch')

    else:
        st.info("Your portfolio is empty. Add stocks using the form above.")

# ==========================================
# TAB 2: WEEKLY SIGNALS (Improved UI)
# ==========================================
with tab2:
    st.header("📢 Weekly Market Signals")
    st.markdown("""
    <style>
    .signal-card {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Load existing signals if available
    existing_signals = None
    if os.path.exists(SIGNALS_FILE):
        try:
            existing_signals = pd.read_csv(SIGNALS_FILE)
        except:
            pass
    
    col_refresh, col_info = st.columns([1, 3])
    with col_refresh:
        refresh_btn = st.button("🔄 Scan Market", key="scan_signals_btn")
    with col_info:
        if existing_signals is not None and not existing_signals.empty:
            signal_date = existing_signals['Date'].iloc[0] if 'Date' in existing_signals.columns else "Unknown"
            buy_count = len(existing_signals[existing_signals['Action'] == 'BUY']) if 'Action' in existing_signals.columns else 0
            sell_count = len(existing_signals[existing_signals['Action'] == 'SELL']) if 'Action' in existing_signals.columns else 0
            st.caption(f"📅 Last Scan: {signal_date} | 🟢 {buy_count} BUY | 🔴 {sell_count} SELL")
    
    # Display signals in improved format
    if existing_signals is not None and not existing_signals.empty and not refresh_btn:
        # Summary metrics
        st.markdown("---")
        
        # Split into BUY and SELL tabs
        buy_signals = existing_signals[existing_signals['Action'] == 'BUY'] if 'Action' in existing_signals.columns else pd.DataFrame()
        sell_signals = existing_signals[existing_signals['Action'] == 'SELL'] if 'Action' in existing_signals.columns else pd.DataFrame()
        
        signal_tab1, signal_tab2 = st.tabs(["🟢 BUY Signals", "🔴 SELL Signals"])
        
        with signal_tab1:
            if not buy_signals.empty:
                st.success(f"**{len(buy_signals)} stocks** with BUY signals")
                
                # Display as cards in columns
                cols_per_row = 3
                for idx, (_, row) in enumerate(buy_signals.iterrows()):
                    if idx % cols_per_row == 0:
                        cols = st.columns(cols_per_row)
                    
                    with cols[idx % cols_per_row]:
                        ticker = row['Ticker']
                        price = row.get('Price', 'N/A')
                        forecast = row.get('Forecast', 'N/A')
                        qty = row.get('Sugg. Qty', '-')
                        trend = row.get('Trend', '-')
                        
                        with st.container():
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #1a472a 0%, #2d5a3d 100%); 
                                        padding: 15px; border-radius: 10px; border-left: 4px solid #00ff00;">
                                <h4 style="margin: 0; color: #00ff00;">🟢 {ticker}</h4>
                                <p style="margin: 5px 0; color: white;">Price: {price}</p>
                                <p style="margin: 5px 0; color: #90EE90;">Forecast: {forecast}</p>
                                <p style="margin: 5px 0; color: #ffffff; font-weight: bold;">Qty: {qty} | Trend: {trend}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Clickable expander for price chart
                            with st.expander(f"📈 View {ticker} Chart"):
                                try:
                                    ticker_data = yf.Ticker(ticker)
                                    hist = ticker_data.history(period="3mo")
                                    
                                    if not hist.empty:
                                        fig_signal = go.Figure()
                                        
                                        # Candlestick chart
                                        fig_signal.add_trace(go.Candlestick(
                                            x=hist.index,
                                            open=hist['Open'],
                                            high=hist['High'],
                                            low=hist['Low'],
                                            close=hist['Close'],
                                            name='Price'
                                        ))
                                        
                                        # Add volume bar
                                        fig_signal.add_trace(go.Bar(
                                            x=hist.index,
                                            y=hist['Volume'],
                                            name='Volume',
                                            yaxis='y2',
                                            marker_color='rgba(100,100,100,0.3)'
                                        ))
                                        
                                        fig_signal.update_layout(
                                            title=f"{ticker} - 3 Month Chart",
                                            height=350,
                                            yaxis=dict(title='Price (₹)', side='left'),
                                            yaxis2=dict(title='Volume', overlaying='y', side='right', showgrid=False),
                                            xaxis_rangeslider_visible=False,
                                            margin=dict(l=10, r=10, t=40, b=10)
                                        )
                                        st.plotly_chart(fig_signal, width='stretch')
                                        
                                        # Quick stats
                                        current = hist['Close'].iloc[-1]
                                        prev_month = hist['Close'].iloc[-22] if len(hist) > 22 else hist['Close'].iloc[0]
                                        change_1m = ((current - prev_month) / prev_month) * 100
                                        
                                        st.markdown(f"**Current:** ₹{current:.2f} | **1M Change:** {change_1m:+.2f}%")
                                except Exception as e:
                                    st.error(f"Could not load chart: {e}")
            else:
                st.info("No BUY signals found in the current scan.")
        
        with signal_tab2:
            if not sell_signals.empty:
                st.error(f"**{len(sell_signals)} stocks** with SELL signals")
                
                cols_per_row = 3
                for idx, (_, row) in enumerate(sell_signals.iterrows()):
                    if idx % cols_per_row == 0:
                        cols = st.columns(cols_per_row)
                    
                    with cols[idx % cols_per_row]:
                        ticker = row['Ticker']
                        price = row.get('Price', 'N/A')
                        forecast = row.get('Forecast', 'N/A')
                        
                        with st.container():
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #4a1a1a 0%, #5a2d2d 100%); 
                                        padding: 15px; border-radius: 10px; border-left: 4px solid #ff4444;">
                                <h4 style="margin: 0; color: #ff4444;">🔴 {ticker}</h4>
                                <p style="margin: 5px 0; color: white;">Price: {price}</p>
                                <p style="margin: 5px 0; color: #ffcccc;">Forecast: {forecast}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            with st.expander(f"📉 View {ticker} Chart"):
                                try:
                                    ticker_data = yf.Ticker(ticker)
                                    hist = ticker_data.history(period="3mo")
                                    
                                    if not hist.empty:
                                        fig_signal = go.Figure()
                                        
                                        fig_signal.add_trace(go.Candlestick(
                                            x=hist.index,
                                            open=hist['Open'],
                                            high=hist['High'],
                                            low=hist['Low'],
                                            close=hist['Close'],
                                            name='Price'
                                        ))
                                        
                                        fig_signal.add_trace(go.Bar(
                                            x=hist.index,
                                            y=hist['Volume'],
                                            name='Volume',
                                            yaxis='y2',
                                            marker_color='rgba(100,100,100,0.3)'
                                        ))
                                        
                                        fig_signal.update_layout(
                                            title=f"{ticker} - 3 Month Chart",
                                            height=350,
                                            yaxis=dict(title='Price (₹)', side='left'),
                                            yaxis2=dict(title='Volume', overlaying='y', side='right', showgrid=False),
                                            xaxis_rangeslider_visible=False,
                                            margin=dict(l=10, r=10, t=40, b=10)
                                        )
                                        st.plotly_chart(fig_signal, width='stretch')
                                        
                                        current = hist['Close'].iloc[-1]
                                        prev_month = hist['Close'].iloc[-22] if len(hist) > 22 else hist['Close'].iloc[0]
                                        change_1m = ((current - prev_month) / prev_month) * 100
                                        
                                        st.markdown(f"**Current:** ₹{current:.2f} | **1M Change:** {change_1m:+.2f}%")
                                except Exception as e:
                                    st.error(f"Could not load chart: {e}")
            else:
                st.info("No SELL signals found in the current scan.")
    
    # Refresh/Scan logic
    if refresh_btn:
        signals = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # --- Load Portfolio State for Sizing ---
        try:
            with open(PORTFOLIO_FILE, 'r') as f:
                pf_state = json.load(f)
            current_cash = pf_state.get('cash_balance', 100000.0)
        except:
            current_cash = 100000.0
            
        CONFIDENCE_THRESHOLD = 0.05  # 0.05% predicted return
        
        for i, ticker in enumerate(NIFTY_250):
            status_text.text(f"🔍 Scanning {ticker}... ({i+1}/{len(NIFTY_250)})")
            df = get_data(ticker, start="2020-01-01") 
            
            if df is not None and len(df) > LOOKBACK_WINDOW:
                X_full, df_sim = prepare_sequences(df, scaler)
                
                if X_full is not None:
                    with torch.no_grad():
                        X_tensor_full = torch.from_numpy(X_full).to(device)
                        predictions_full = model(X_tensor_full).cpu().numpy().flatten()
                    
                    _, trades = run_backtest_logic(df_sim, predictions_full)
                    
                    in_position = False
                    if trades:
                        last_trade = trades[-1]
                        if last_trade['Type'] == 'BUY':
                            in_position = True
                    
                    df_proc = df.copy()
                    df_proc['Log_Ret'] = np.log(df_proc['Close'] / df_proc['Close'].shift(1)) * 100
                    df_proc['RSI_Norm'] = df_proc['RSI'] / 100.0
                    df_proc['Price_EMA20'] = (df_proc['Close'] / df_proc['EMA_20']) - 1
                    df_proc['Price_EMA50'] = (df_proc['Close'] / df_proc['EMA_50']) - 1
                    df_proc['ATR_Price'] = df_proc['ATR'] / df_proc['Close']
                    df_proc['MACD_Norm'] = df_proc['MACD'] / df_proc['Close']
                    df_proc['CCI_Norm'] = df_proc['CCI'] / 100.0
                    
                    df_proc.bfill(inplace=True)
                    df_proc.fillna(0, inplace=True)
                    
                    last_window_raw = df_proc.iloc[-LOOKBACK_WINDOW:].copy()
                    feature_cols = ['Log_Ret', 'RSI_Norm', 'Price_EMA20', 'Price_EMA50', 'ATR_Price', 'MACD_Norm', 'CCI_Norm']
                    last_window_raw[feature_cols] = scaler.transform(last_window_raw[feature_cols])
                    
                    X_next = last_window_raw[feature_cols].values.reshape(1, LOOKBACK_WINDOW, len(feature_cols))
                    X_tensor_next = torch.from_numpy(X_next.astype(np.float32)).to(device)
                    
                    with torch.no_grad():
                        pred_next = model(X_tensor_next).item()
                    
                    # --- Advanced Filtering Logic ---
                    current_price = df['Close'].iloc[-1]
                    ema_20 = df['EMA_20'].iloc[-1]
                    ema_50 = df['EMA_50'].iloc[-1]
                    current_vol = df['Volume'].iloc[-1]
                    # Calculate Volume SMA (20) if not present, or just compute on the fly
                    vol_ma_20 = df['Volume'].rolling(window=20).mean().iloc[-1]
                    
                    # Filters
                    trend_bullish = ema_20 > ema_50
                    vol_confirmed = current_vol > vol_ma_20
                    is_confident = abs(pred_next) > CONFIDENCE_THRESHOLD
                    
                    signal_type = "NEUTRAL"
                    status = "FLAT"
                    suggested_qty = 0
                    
                    if in_position:
                        status = "HOLDING"
                        if pred_next < 0.0:
                            signal_type = "SELL"
                        else:
                            signal_type = "HOLD"
                    else:
                        status = "FLAT"
                        # Only BUY if ALL conditions met: Positive Pred + Confidence + Trend + Volume
                        if pred_next > 0.0 and is_confident and trend_bullish and vol_confirmed:
                            signal_type = "BUY"
                            
                            # Position Sizing: 
                            # Base 10% of cash. Scale up slightly by confidence.
                            # Cap at 20% of cash to be safe.
                            base_alloc = current_cash * 0.10
                            conf_multiplier = 1 + (abs(pred_next) * 5) # e.g. 0.1% pred -> 1.005x
                            alloc_amt = min(base_alloc * conf_multiplier, current_cash * 0.20)
                            suggested_qty = max(1, int(alloc_amt / current_price))
                        else:
                            signal_type = "WAIT"
                    
                    if signal_type in ["BUY", "SELL"]:
                        signals.append({
                            "Date": datetime.now().strftime("%Y-%m-%d"),
                            "Ticker": ticker,
                            "Price": f"₹{current_price:.2f}",
                            "Forecast": f"{pred_next:.4f}%",
                            "Trend": "Bull" if trend_bullish else "Bear",
                            "Vol > Avg": "Yes" if vol_confirmed else "No",
                            "Sugg. Qty": suggested_qty if signal_type == "BUY" else "-",
                            "Action": signal_type
                        })
                        
            progress_bar.progress((i + 1) / len(NIFTY_250))
            
        status_text.text("✅ Scan Complete!")
        progress_bar.empty()
        
        if signals:
            st.success(f"🎯 Found {len(signals)} high-quality signals!")
            df_signals = pd.DataFrame(signals)
            df_signals.to_csv(SIGNALS_FILE, index=False)
            st.rerun()
        else:
            st.info("No signals found matching strict criteria (Trend + Vol + Conf).")

# ==========================================
# TAB 3: TESTING (Backtest / Bot Simulation)
# ==========================================
with tab3:
    st.header("🧪 Testing & Simulation")
    
    # Dropdown selector for test mode
    test_mode = st.selectbox(
        "Select Testing Mode", 
        ["📈 Backtest Single Stock", "🤖 Bot Simulation (NIFTY 250)"],
        key="test_mode_selector"
    )
    
    st.markdown("---")
    
    # =====================
    # BACKTEST MODE
    # =====================
    if test_mode == "📈 Backtest Single Stock":
        st.subheader("📈 Backtest Single Stock")
        
        col_bt1, col_bt2, col_bt3 = st.columns([2, 2, 1])
        with col_bt1:
            stock_option = st.selectbox("Select Stock", ["Custom"] + NIFTY_250, key="backtest_ticker_select")
        with col_bt2:
            if stock_option == "Custom":
                selected_ticker = st.text_input("Enter Ticker (e.g., ITC.NS, AAPL)", value="ITC.NS", key="custom_ticker_input")
            else:
                selected_ticker = stock_option
                st.text_input("Selected Ticker", value=selected_ticker, disabled=True, key="display_ticker")
        with col_bt3:
            initial_capital = st.number_input("Capital (₹)", value=100000, step=10000, key="backtest_capital")
    
        if st.button("Run Backtest", key="btn_run_backtest"):
            with st.spinner(f"Running simulation for {selected_ticker}..."):
                df = get_data(selected_ticker, start="2020-01-01") 
                
                if df is not None and len(df) > LOOKBACK_WINDOW:
                    X, df_sim = prepare_sequences(df, scaler)
                    
                    if X is not None:
                        with torch.no_grad():
                            X_tensor = torch.from_numpy(X).to(device)
                            predictions = model(X_tensor).cpu().numpy().flatten()
                        
                        port_vals, trades = run_backtest_logic(df_sim, predictions, initial_capital)
                        
                        final_val = port_vals[-1]
                        net_profit = final_val - initial_capital
                        ret_pct = (net_profit / initial_capital) * 100
                        
                        bnh_shares = initial_capital / df_sim['Close'].iloc[0]
                        bnh_values = df_sim['Close'] * bnh_shares
                        bnh_ret = ((bnh_values.iloc[-1] - initial_capital) / initial_capital) * 100
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("AI Strategy Return", f"{ret_pct:.2f}%", f"₹{net_profit:,.0f}")
                        col2.metric("Buy & Hold Return", f"{bnh_ret:.2f}%", f"₹{(bnh_values.iloc[-1] - initial_capital):,.0f}")
                        col3.metric("Total Trades", len(trades))
                        
                        # Price History
                        fig_price = go.Figure()
                        fig_price.add_trace(go.Scatter(x=df_sim.index, y=df_sim['Close'], name="Close Price", line=dict(color='#1f77b4', width=2)))
                        fig_price.update_layout(title=f"{selected_ticker} Price History", xaxis_title="Date", yaxis_title="Price (₹)", height=400)
                        st.plotly_chart(fig_price, width='stretch')

                        # Equity Curve
                        fig_equity = go.Figure()
                        fig_equity.add_trace(go.Scatter(x=df_sim.index, y=port_vals, name="AI Strategy", line=dict(color='#00ff00', width=2)))
                        fig_equity.add_trace(go.Scatter(x=df_sim.index, y=bnh_values, name="Buy & Hold", line=dict(color='gray', dash='dash')))
                        fig_equity.update_layout(title="Equity Curve Comparison", xaxis_title="Date", yaxis_title="Portfolio Value (₹)", height=400)
                        st.plotly_chart(fig_equity, width='stretch')
                        
                        # Trade Execution
                        fig_trades = go.Figure()
                        fig_trades.add_trace(go.Scatter(x=df_sim.index, y=df_sim['Close'], name="Price", line=dict(color='gray', width=1)))
                        
                        buy_dates = [t['Date'] for t in trades if t['Type'] == 'BUY']
                        buy_prices = [t['Price'] for t in trades if t['Type'] == 'BUY']
                        fig_trades.add_trace(go.Scatter(x=buy_dates, y=buy_prices, mode='markers', name="Buy", marker=dict(color='green', symbol='triangle-up', size=10)))
                        
                        sell_dates = [t['Date'] for t in trades if t['Type'] == 'SELL']
                        sell_prices = [t['Price'] for t in trades if t['Type'] == 'SELL']
                        sell_pnl = [f"PnL: ₹{t['PnL']:.2f} ({t['PnL_Pct']:.2f}%)" for t in trades if t['Type'] == 'SELL']
                        
                        fig_trades.add_trace(go.Scatter(
                            x=sell_dates, y=sell_prices, mode='markers', name="Sell", 
                            marker=dict(color='red', symbol='triangle-down', size=10),
                            text=sell_pnl, hoverinfo='text+x+y'
                        ))
                        
                        fig_trades.update_layout(title="Trade Execution Points (with P&L)", xaxis_title="Date", yaxis_title="Price (₹)", height=400)
                        st.plotly_chart(fig_trades, width='stretch')
                        
                        # Technical Indicators
                        st.subheader("Technical Indicators")
                        col_tech1, col_tech2 = st.columns(2)
                        
                        with col_tech1:
                            fig_rsi = go.Figure()
                            fig_rsi.add_trace(go.Scatter(x=df_sim.index, y=df_sim['RSI'], name="RSI", line=dict(color='purple')))
                            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                            fig_rsi.update_layout(title="RSI (14)", height=300, yaxis_range=[0, 100])
                            st.plotly_chart(fig_rsi, width='stretch')
                            
                        with col_tech2:
                            if 'MACD' in df_sim.columns:
                                fig_macd = go.Figure()
                                fig_macd.add_trace(go.Scatter(x=df_sim.index, y=df_sim['MACD'], name="MACD", line=dict(color='blue')))
                                fig_macd.add_hline(y=0, line_dash="dash", line_color="gray")
                                fig_macd.update_layout(title="MACD", height=300)
                                st.plotly_chart(fig_macd, width='stretch')
                        
                        # Forecast
                        st.subheader("🔮 Forecast for Tomorrow")
                        
                        df_proc = df.copy()
                        df_proc['Log_Ret'] = np.log(df_proc['Close'] / df_proc['Close'].shift(1)) * 100
                        df_proc['RSI_Norm'] = df_proc['RSI'] / 100.0
                        df_proc['Price_EMA20'] = (df_proc['Close'] / df_proc['EMA_20']) - 1
                        df_proc['Price_EMA50'] = (df_proc['Close'] / df_proc['EMA_50']) - 1
                        df_proc['ATR_Price'] = df_proc['ATR'] / df_proc['Close']
                        df_proc['MACD_Norm'] = df_proc['MACD'] / df_proc['Close']
                        df_proc['CCI_Norm'] = df_proc['CCI'] / 100.0
                        
                        df_proc.bfill(inplace=True)
                        df_proc.fillna(0, inplace=True)
                        
                        last_window_raw = df_proc.iloc[-LOOKBACK_WINDOW:].copy()
                        feature_cols = ['Log_Ret', 'RSI_Norm', 'Price_EMA20', 'Price_EMA50', 'ATR_Price', 'MACD_Norm', 'CCI_Norm']
                        last_window_raw[feature_cols] = scaler.transform(last_window_raw[feature_cols])
                        
                        X_next = last_window_raw[feature_cols].values.reshape(1, LOOKBACK_WINDOW, len(feature_cols))
                        X_tensor_next = torch.from_numpy(X_next.astype(np.float32)).to(device)
                        
                        with torch.no_grad():
                            pred_next = model(X_tensor_next).item()
                            
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
                        
                        with st.expander("View Trade Log"):
                            st.dataframe(pd.DataFrame(trades))
                    else:
                        st.error("Not enough data to generate sequences.")
                else:
                    st.error("Could not fetch data or data too short.")
    
    # =====================
    # BOT SIMULATION MODE
    # =====================
    else:  # Bot Simulation
        st.subheader("🤖 Autonomous AI Bot Strategy")
        st.markdown("""
        This bot autonomously trades the entire **NIFTY 250** universe. 
        It dynamically manages risk, position sizing, and portfolio rebalancing based on AI confidence and market volatility.
        **Self-Correction:** The bot adjusts its risk appetite based on its recent win rate.
        """)
        
        col_b1, col_b2 = st.columns(2)
        
        with col_b1:
            bot_capital = st.number_input("Initial Bot Capital (₹)", value=500000, step=50000, key="bot_capital")
            
        with col_b2:
            bot_start_date = st.date_input("Start Date", datetime(2023, 1, 1), key="bot_start")
            bot_end_date = st.date_input("End Date", datetime.now(), key="bot_end")

        if st.button("🚀 Run Autonomous Bot", key="btn_run_bot"):
            status_area = st.empty()
            progress_bar = st.progress(0)
            
            status_area.text("Scanning NIFTY 250 Universe... This may take a moment.")
            
            market_data = {}
            predictions_map = {}
            valid_tickers = []
            
            universe = NIFTY_250 
            
            for i, t in enumerate(universe):
                df = get_data(t, start=str(bot_start_date - timedelta(days=150))) 
                
                if df is not None and len(df) > LOOKBACK_WINDOW:
                    X, df_sim = prepare_sequences(df, scaler)
                    if X is not None:
                        with torch.no_grad():
                            X_tensor = torch.from_numpy(X).to(device)
                            preds = model(X_tensor).cpu().numpy().flatten()
                        
                        pred_series = pd.Series(preds, index=df_sim.index)
                        
                        market_data[t] = df_sim
                        predictions_map[t] = pred_series
                        valid_tickers.append(t)
                
                progress_bar.progress((i + 1) / len(universe) * 0.3)
            
            if not valid_tickers:
                st.error("No valid data found for NIFTY 250 tickers.")
                st.stop()
                
            status_area.text("Simulating trading days with Adaptive Risk Management...")
            
            all_dates = sorted(list(set().union(*[df.index for df in market_data.values()])))
            sim_dates = [d for d in all_dates if pd.Timestamp(bot_start_date) <= d <= pd.Timestamp(bot_end_date)]
            
            cash = bot_capital
            portfolio_bot = {}
            history = [] 
            trade_log = []
            
            current_risk_per_trade = 0.02
            win_history = []
            
            for i, current_date in enumerate(sim_dates):
                if len(win_history) > 20:
                    win_history = win_history[-20:]
                
                if len(win_history) >= 5:
                    win_rate = sum(win_history) / len(win_history)
                    if win_rate > 0.6:
                        current_risk_per_trade = min(0.05, current_risk_per_trade * 1.05)
                    elif win_rate < 0.4:
                        current_risk_per_trade = max(0.005, current_risk_per_trade * 0.95)
                
                holdings_value = 0
                
                for t in list(portfolio_bot.keys()):
                    if current_date in market_data[t].index:
                        price = market_data[t].loc[current_date, 'Close']
                        pred = predictions_map[t].get(current_date, 0)
                        pos = portfolio_bot[t]
                        
                        is_stop_hit = price < pos['stop_loss']
                        is_sell_signal = pred < 0
                        
                        if is_sell_signal or is_stop_hit:
                            revenue = pos['qty'] * price
                            cash += revenue
                            
                            pnl = revenue - (pos['qty'] * pos['entry_price'])
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
                            del portfolio_bot[t]
                        else:
                            holdings_value += pos['qty'] * price
                            if price > pos['entry_price'] * 1.05 and pos['stop_loss'] < pos['entry_price']:
                                portfolio_bot[t]['stop_loss'] = pos['entry_price']
                    else:
                        if t in portfolio_bot:
                            holdings_value += portfolio_bot[t]['qty'] * portfolio_bot[t]['entry_price']

                current_port_value = cash + holdings_value
                
                dynamic_max_pos = min(20, max(5, int(current_port_value / 50000)))
                
                if len(portfolio_bot) < dynamic_max_pos:
                    candidates = []
                    for t in valid_tickers:
                        if t not in portfolio_bot and current_date in market_data[t].index:
                            pred = predictions_map[t].get(current_date, 0)
                            if pred > 0:
                                candidates.append((t, pred))
                    
                    candidates.sort(key=lambda x: x[1], reverse=True)
                    slots_available = dynamic_max_pos - len(portfolio_bot)
                    
                    for t, pred in candidates[:slots_available]:
                        row = market_data[t].loc[current_date]
                        price = row['Close']
                        atr = row['ATR']
                        
                        if atr > 0:
                            stop_distance = atr * 2.0
                            risk_amount = current_port_value * current_risk_per_trade
                            
                            shares_to_buy = int(risk_amount // stop_distance)
                            max_cost = current_port_value * 0.20
                            cost = shares_to_buy * price
                            
                            if cost > max_cost:
                                shares_to_buy = int(max_cost // price)
                                
                            if shares_to_buy > 0 and cash >= (shares_to_buy * price):
                                cost = shares_to_buy * price
                                cash -= cost
                                stop_price = price - stop_distance
                                
                                portfolio_bot[t] = {
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
                
                history.append({
                    "Date": current_date,
                    "Portfolio Value": current_port_value,
                    "Cash": cash,
                    "Risk Level": current_risk_per_trade * 100
                })
                
                progress_bar.progress(0.3 + (0.7 * (i + 1) / len(sim_dates)))
            
            status_area.empty()
            
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
                
                fig_bot = go.Figure()
                fig_bot.add_trace(go.Scatter(x=df_hist.index, y=df_hist['Portfolio Value'], name="Portfolio Value", line=dict(color='#00ff00', width=2)))
                fig_bot.add_trace(go.Scatter(x=df_hist.index, y=df_hist['Cash'], name="Cash Held", line=dict(color='gray', dash='dot')))
                fig_bot.update_layout(title="Bot Portfolio Performance", xaxis_title="Date", yaxis_title="Value (₹)", height=450)
                st.plotly_chart(fig_bot, width='stretch')
                
                fig_risk = go.Figure()
                fig_risk.add_trace(go.Scatter(x=df_hist.index, y=df_hist['Risk Level'], name="Risk % Per Trade", line=dict(color='orange')))
                fig_risk.update_layout(title="Adaptive Risk Level (Self-Correction)", xaxis_title="Date", yaxis_title="Risk %", height=300)
                st.plotly_chart(fig_risk, width='stretch')
                
                st.subheader("📜 Trade History")
                if not df_trades.empty:
                    st.dataframe(df_trades)
                else:
                    st.info("No trades were executed.")
