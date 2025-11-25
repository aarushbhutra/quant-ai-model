import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
try:
    import torch_directml
except ImportError:
    torch_directml = None
import joblib
import os
from datetime import datetime, timedelta

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = "lstm_trading_model.pth"
SCALER_PATH = "scaler.pkl"
LOOKBACK_WINDOW = 60
TICKER = "TCS.NS" 

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
# UTILS
# ==========================================
def get_recent_data(ticker, days=365):
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    df = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
    
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
    return df

def prepare_input(df, scaler):
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1)) * 100
    df['RSI_Norm'] = df['RSI'] / 100.0
    df['Price_EMA20'] = (df['Close'] / df['EMA_20']) - 1
    df['Price_EMA50'] = (df['Close'] / df['EMA_50']) - 1
    df['ATR_Price'] = df['ATR'] / df['Close']
    df['MACD_Norm'] = df['MACD'] / df['Close']
    df['CCI_Norm'] = df['CCI'] / 100.0
    
    if len(df) < LOOKBACK_WINDOW:
        raise ValueError(f"Not enough data. Need {LOOKBACK_WINDOW} rows, got {len(df)}.")

    last_window = df.iloc[-LOOKBACK_WINDOW:].copy()
    feature_cols = ['Log_Ret', 'RSI_Norm', 'Price_EMA20', 'Price_EMA50', 'ATR_Price', 'MACD_Norm', 'CCI_Norm']
    last_window[feature_cols] = scaler.transform(last_window[feature_cols])
    
    X = last_window[feature_cols].values.reshape(1, LOOKBACK_WINDOW, len(feature_cols))
    return torch.from_numpy(X.astype(np.float32)).to(device)

# ==========================================
# SELF CORRECTION
# ==========================================
def self_correct(model, df, scaler):
    print("Performing Self-Correction (Backpropagation)...")
    
    if len(df) < LOOKBACK_WINDOW + 2:
        print("Not enough data to self-correct.")
        return model

    feature_cols = ['Log_Ret', 'RSI_Norm', 'Price_EMA20', 'Price_EMA50', 'ATR_Price']
    
    df_features = df.copy()
    df_features['Log_Ret'] = np.log(df_features['Close'] / df_features['Close'].shift(1)) * 100
    df_features['RSI_Norm'] = df_features['RSI'] / 100.0
    df_features['Price_EMA20'] = (df_features['Close'] / df_features['EMA_20']) - 1
    df_features['Price_EMA50'] = (df_features['Close'] / df_features['EMA_50']) - 1
    df_features['ATR_Price'] = df_features['ATR'] / df_features['Close']
    
    actual_return_today = df_features['Log_Ret'].iloc[-1]
    prev_window = df_features.iloc[-LOOKBACK_WINDOW-1:-1].copy()
    prev_window[feature_cols] = scaler.transform(prev_window[feature_cols])
    
    X_batch = prev_window[feature_cols].values.reshape(1, LOOKBACK_WINDOW, len(feature_cols))
    y_batch = np.array([[actual_return_today]])
    
    X_tensor = torch.from_numpy(X_batch.astype(np.float32)).to(device)
    y_tensor = torch.from_numpy(y_batch.astype(np.float32)).to(device)
    
    # Train Step
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    optimizer.zero_grad()
    output = model(X_tensor)
    loss = criterion(output, y_tensor)
    loss.backward()
    optimizer.step()
    
    return model

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Please run train_model.py first.")
        exit()
        
    print(f"Loading model for {TICKER} on {device}...")
    model = LSTMModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    scaler = joblib.load(SCALER_PATH)
    
    # 1. Get Data
    df = get_recent_data(TICKER)
    current_price = df['Close'].iloc[-1]
    print(f"Current Price: {current_price}")
    
    # 2. Self Correct
    model = self_correct(model, df, scaler)
    
    # 3. Predict
    model.eval()
    with torch.no_grad():
        X_new = prepare_input(df, scaler)
        predicted_pct = model(X_new).item()
        
    predicted_log_ret = predicted_pct / 100.0
    predicted_price = current_price * np.exp(predicted_log_ret)
    
    print(f"\nPrediction for Tomorrow:")
    print(f"Predicted Return: {predicted_pct:.2f}%")
    print(f"Predicted Price: {predicted_price:.2f}")
    
    if predicted_pct > 0:
        print("ACTION: BUY (Predicted UP)")
    else:
        print("ACTION: SELL/HOLD (Predicted DOWN)")
        
    # 4. Save
    torch.save(model.state_dict(), MODEL_PATH)
    print("Model updated and saved.")
