import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
try:
    import torch_directml
except ImportError:
    torch_directml = None
import os
import joblib

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = "lstm_trading_model.pth"
SCALER_PATH = "scaler.pkl"
TICKER = "ITC.NS"
START_DATE = "2020-01-01" 
INITIAL_CAPITAL = 100_000
LOOKBACK_WINDOW = 60

# Setup Device
try:
    if torch_directml:
        device = torch_directml.device()
    else:
        device = torch.device("cpu")
except:
    device = torch.device("cpu")

# ==========================================
# MODEL DEFINITION (Must match training)
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
# DATA PREPARATION
# ==========================================
def get_data(ticker):
    print(f"Fetching backtest data for {ticker}...")
    df = yf.download(ticker, start=START_DATE, progress=False, auto_adjust=True)
    
    if isinstance(df.columns, pd.MultiIndex):
        if 'Close' in df.columns.get_level_values(0):
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = df.columns.get_level_values(1)

    df['RSI'] = df.ta.rsi(length=14)
    df['EMA_20'] = df.ta.ema(length=20)
    df['EMA_50'] = df.ta.ema(length=50)
    df['ATR'] = df.ta.atr(length=14)
    
    macd = df.ta.macd(fast=12, slow=26, signal=9)
    if macd is not None:
        df['MACD'] = macd['MACD_12_26_9']
        
    df['CCI'] = df.ta.cci(length=20)
    
    df.dropna(inplace=True)
    return df

def prepare_sequences(df, scaler):
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1)) * 100
    df['RSI_Norm'] = df['RSI'] / 100.0
    df['Price_EMA20'] = (df['Close'] / df['EMA_20']) - 1
    df['Price_EMA50'] = (df['Close'] / df['EMA_50']) - 1
    df['ATR_Price'] = df['ATR'] / df['Close']
    df['MACD_Norm'] = df['MACD'] / df['Close']
    df['CCI_Norm'] = df['CCI'] / 100.0
    
    df.dropna(inplace=True)
    
    feature_cols = ['Log_Ret', 'RSI_Norm', 'Price_EMA20', 'Price_EMA50', 'ATR_Price', 'MACD_Norm', 'CCI_Norm']
    
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.transform(df[feature_cols])
    
    X = []
    indices = []
    
    data_values = df_scaled[feature_cols].values
    
    for i in range(LOOKBACK_WINDOW, len(df)):
        X.append(data_values[i-LOOKBACK_WINDOW:i])
        indices.append(df.index[i])
        
    return np.array(X, dtype=np.float32), df.loc[indices]

# ==========================================
# BACKTEST ENGINE
# ==========================================
def run_backtest(df, predictions):
    print("Running Backtest Simulation...")
    
    capital = INITIAL_CAPITAL
    shares = 0
    in_position = False
    entry_price = 0
    
    portfolio_values = []
    trades = []
    
    df['Predicted_Log_Ret'] = predictions
    
    # Dynamic Thresholds based on Percentiles to FORCE trades
    # We buy when the model is in its "most confident" top 20% of predictions
    # We sell when the model drops below the median or hits stop loss
    
    # Add tiny noise to predictions to break ties if model is flat
    if predictions.std() < 1e-6:
        print("\n[WARNING] Model predictions are flat. Adding dithering noise to force trades.")
        predictions += np.random.normal(0, 1e-5, predictions.shape)
        
    buy_threshold = np.percentile(predictions, 80) 
    sell_threshold = np.percentile(predictions, 50)
    
    print(f"Prediction Stats: Min={predictions.min():.6f}, Max={predictions.max():.6f}, Mean={predictions.mean():.6f}, Std={predictions.std():.6f}")
    print(f"Using Buy Threshold (80th percentile): {buy_threshold:.6f}")
    print(f"Using Sell Threshold (50th percentile): {sell_threshold:.6f}")

    for date, row in df.iterrows():
        price = row['Close']
        # Use the potentially dithered prediction
        pred_ret = predictions[df.index.get_loc(date)]
        
        # Trading Logic
        # 1. Buy Signal (Top 20% bullish predictions)
        if pred_ret >= buy_threshold and not in_position:
            shares = int(capital // price)
            if shares > 0:
                cost = shares * price
                capital -= cost
                in_position = True
                entry_price = price
                trades.append({'Date': date, 'Type': 'BUY', 'Price': price, 'Balance': capital})
        
        # 2. Sell Signal (Below Median Prediction OR Stop Loss)
        elif in_position:
            # Calculate current PnL percentage
            pnl_pct = (price - entry_price) / entry_price * 100
            
            # Sell if prediction weakens OR Stop Loss hit (-2%)
            if pred_ret < sell_threshold or pnl_pct < -2.0:
                revenue = shares * price
                capital += revenue
                shares = 0
                in_position = False
                reason = 'Signal' if pred_ret < sell_threshold else 'Stop Loss'
                trades.append({'Date': date, 'Type': 'SELL', 'Price': price, 'Balance': capital, 'Reason': reason})
            
        current_val = capital + (shares * price)
        portfolio_values.append(current_val)
        
    return portfolio_values, trades

# ==========================================
# VISUALIZATION
# ==========================================
def plot_results(data, portfolio_values, trades):
    final_val = portfolio_values[-1]
    net_profit = final_val - INITIAL_CAPITAL
    ret_pct = (net_profit / INITIAL_CAPITAL) * 100
    
    bnh_shares = INITIAL_CAPITAL / data['Close'].iloc[0]
    bnh_values = data['Close'] * bnh_shares
    bnh_ret = ((bnh_values.iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100

    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    ax1.plot(data.index, portfolio_values, color='#00ff00', linewidth=2, label=f'LSTM Model ({ret_pct:.1f}%)')
    ax1.plot(data.index, bnh_values, color='gray', alpha=0.5, linestyle='--', label=f'Buy & Hold ({bnh_ret:.1f}%)')
    ax1.set_title(f"Final Balance: ₹{final_val:.0f} | Net Profit: ₹{net_profit:.0f}", fontsize=14, color='white')
    ax1.legend()
    ax1.grid(color='gray', linestyle=':', alpha=0.3)
    
    ax2.plot(data.index, data['Close'], color='gray', alpha=0.5)
    
    buy_dates = [t['Date'] for t in trades if t['Type'] == 'BUY']
    buy_prices = [t['Price'] for t in trades if t['Type'] == 'BUY']
    ax2.scatter(buy_dates, buy_prices, color='#00ff00', marker='^', s=80, label='Buy')
    
    sell_dates = [t['Date'] for t in trades if t['Type'] == 'SELL']
    sell_prices = [t['Price'] for t in trades if t['Type'] == 'SELL']
    ax2.scatter(sell_dates, sell_prices, color='red', marker='v', s=80, label='Sell')
    
    ax2.set_title("Trade Execution Points")
    ax2.grid(color='gray', linestyle=':', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Please run train_model.py first.")
        exit()
        
    print(f"Loading model and scaler on {device}...")
    model = LSTMModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    scaler = joblib.load(SCALER_PATH)
    
    # 1. Get Data
    df = get_data(TICKER)
    
    # 2. Prepare Sequences
    X, df_sim = prepare_sequences(df, scaler)
    
    # 3. Predict
    print("Generating predictions...")
    with torch.no_grad():
        X_tensor = torch.from_numpy(X).to(device)
        predictions = model(X_tensor).cpu().numpy().flatten()
    
    # 4. Backtest
    port_vals, trades = run_backtest(df_sim, predictions)
    
    # 5. Plot
    plot_results(df_sim, port_vals, trades)
    print(f"Total Trades: {len(trades)}")
