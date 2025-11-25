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
from sklearn.preprocessing import StandardScaler
import joblib
import os

# ==========================================
# CONFIGURATION
# ==========================================
TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "HCLTECH.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "BAJFINANCE.NS", "WIPRO.NS",
    "NESTLEIND.NS", "TATASTEEL.NS", "POWERGRID.NS", "NTPC.NS", "M&M.NS"
]

START_DATE = "2015-01-01"
LOOKBACK_WINDOW = 60
FORECAST_HORIZON = 30  # Predict next 30 days at once
MODEL_PATH = "long_term_forecast_model.pth"
SCALER_PATH = "scaler_forecast.pkl"

# Setup Device
try:
    if torch_directml:
        device = torch_directml.device()
        print(f"Using DirectML Device: {device}")
    else:
        raise ImportError("torch_directml module not found")
except Exception as e:
    print(f"DirectML not available, using CPU. Error: {e}")
    device = torch.device("cpu")

# ==========================================
# DATA PREPARATION
# ==========================================
def get_stock_data(ticker):
    print(f"Fetching data for {ticker}...")
    try:
        df = yf.download(ticker, start=START_DATE, progress=False, auto_adjust=True)
        if df.empty: return None
            
        if isinstance(df.columns, pd.MultiIndex):
            if 'Close' in df.columns.get_level_values(0):
                df.columns = df.columns.get_level_values(0)
            else:
                df.columns = df.columns.get_level_values(1)

        df['RSI'] = df.ta.rsi(length=14)
        df['EMA_20'] = df.ta.ema(length=20)
        df['EMA_50'] = df.ta.ema(length=50)
        df['ATR'] = df.ta.atr(length=14)
        
        # New Indicators for better trend detection
        macd = df.ta.macd(fast=12, slow=26, signal=9)
        if macd is not None:
            df['MACD'] = macd['MACD_12_26_9']
            df['MACD_Signal'] = macd['MACDs_12_26_9']
        
        df['CCI'] = df.ta.cci(length=20)
        
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

def prepare_dataset(tickers):
    combined_data = []
    
    for ticker in tickers:
        df = get_stock_data(ticker)
        if df is not None:
            df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1)) * 100
            df['RSI_Norm'] = df['RSI'] / 100.0
            df['Price_EMA20'] = (df['Close'] / df['EMA_20']) - 1
            df['Price_EMA50'] = (df['Close'] / df['EMA_50']) - 1
            df['ATR_Price'] = df['ATR'] / df['Close']
            df['MACD_Norm'] = df['MACD'] / df['Close']
            df['CCI_Norm'] = df['CCI'] / 100.0
            
            df.dropna(inplace=True)
            
            # We need to create sequences where Y is a vector of next 30 returns
            # We can't just use shift(-1), we need to construct it manually
            combined_data.append(df[['Log_Ret', 'RSI_Norm', 'Price_EMA20', 'Price_EMA50', 'ATR_Price', 'MACD_Norm', 'CCI_Norm']])

    if not combined_data: raise ValueError("No data fetched.")
    
    # Fit Scaler on all data
    full_df = pd.concat(combined_data)
    scaler = StandardScaler()
    feature_cols = ['Log_Ret', 'RSI_Norm', 'Price_EMA20', 'Price_EMA50', 'ATR_Price', 'MACD_Norm', 'CCI_Norm']
    scaler.fit(full_df[feature_cols])
    
    X_list, y_list = [], []
    
    print("Creating sequences...")
    for df in combined_data:
        # Scale per ticker
        df[feature_cols] = scaler.transform(df[feature_cols])
        data_values = df[feature_cols].values
        target_values = df['Log_Ret'].values # We predict Log Returns
        
        # Create sequences
        # X: [t-60 : t]
        # y: [t : t+30]
        for i in range(LOOKBACK_WINDOW, len(df) - FORECAST_HORIZON):
            X_list.append(data_values[i-LOOKBACK_WINDOW:i])
            y_list.append(target_values[i:i+FORECAST_HORIZON])
            
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32), scaler

# ==========================================
# MODEL DEFINITION (Attention LSTM)
# ==========================================
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, rnn_output):
        # rnn_output: [batch, seq_len, hidden_size]
        energies = self.attn(rnn_output) # [batch, seq_len, 1]
        weights = torch.softmax(energies, dim=1)
        context = torch.sum(rnn_output * weights, dim=1) # [batch, hidden_size]
        return context

class MultiStepLSTM(nn.Module):
    def __init__(self, input_size=7, hidden_size=128, num_layers=2, output_size=30):
        super(MultiStepLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.attention = Attention(hidden_size)
        
        # Output layer predicts 'output_size' steps at once
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        # x: [batch, lookback, features]
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        # Use Attention instead of just last hidden state
        context = self.attention(out)
        
        prediction = self.fc(context)
        return prediction

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("Starting Data Collection & Training for Long-Term Model...")
    
    X, y, scaler = prepare_dataset(TICKERS)
    print(f"Training Data Shape: X={X.shape}, y={y.shape}")
    
    # Convert to Tensors
    X_tensor = torch.from_numpy(X).to(device)
    y_tensor = torch.from_numpy(y).to(device)
    
    # Initialize Model
    model = MultiStepLSTM(input_size=7, output_size=FORECAST_HORIZON).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005) # Lower LR for stability
    
    # Training Loop
    epochs = 20 # Increased epochs
    batch_size = 64
    num_samples = X_tensor.size(0)
    
    print(f"Training on {device}...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        permutation = torch.randperm(num_samples)
        for i in range(0, num_samples, batch_size):
            indices = permutation[i:i+batch_size]
            batch_X, batch_y = X_tensor[indices], y_tensor[indices]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/ (num_samples/batch_size):.6f}")
    
    # Save
    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Scaler saved to {SCALER_PATH}")
