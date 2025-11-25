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
from datetime import timedelta

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = "long_term_forecast_model.pth"
SCALER_PATH = "scaler_forecast.pkl"
FORECAST_FILE = "forecast_data.csv"
TICKER = "TCS.NS"
FORECAST_YEARS = 2
FORECAST_DAYS = 252 * FORECAST_YEARS
LOOKBACK_WINDOW = 60
FORECAST_HORIZON = 30 # Must match training

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
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, rnn_output):
        energies = self.attn(rnn_output)
        weights = torch.softmax(energies, dim=1)
        context = torch.sum(rnn_output * weights, dim=1)
        return context

class MultiStepLSTM(nn.Module):
    def __init__(self, input_size=7, hidden_size=128, num_layers=2, output_size=30):
        super(MultiStepLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.attention = Attention(hidden_size)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        context = self.attention(out)
        prediction = self.fc(context)
        return prediction

# ==========================================
# DATA & FORECASTING
# ==========================================
def get_recent_data(ticker):
    print(f"Fetching recent data for {ticker}...")
    # Get enough data to establish indicators and lookback
    df = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
    
    if isinstance(df.columns, pd.MultiIndex):
        if 'Close' in df.columns.get_level_values(0):
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = df.columns.get_level_values(1)
            
    return df

def calculate_indicators(df):
    df = df.copy()
    df['RSI'] = df.ta.rsi(length=14)
    df['EMA_20'] = df.ta.ema(length=20)
    df['EMA_50'] = df.ta.ema(length=50)
    df['ATR'] = df.ta.atr(length=14)
    
    macd = df.ta.macd(fast=12, slow=26, signal=9)
    if macd is not None:
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_Signal'] = macd['MACDs_12_26_9']
    
    df['CCI'] = df.ta.cci(length=20)
    return df

def prepare_last_window(df, scaler):
    # Feature Engineering
    df_feat = df.copy()
    df_feat['Log_Ret'] = np.log(df_feat['Close'] / df_feat['Close'].shift(1)) * 100
    df_feat['RSI_Norm'] = df_feat['RSI'] / 100.0
    df_feat['Price_EMA20'] = (df_feat['Close'] / df_feat['EMA_20']) - 1
    df_feat['Price_EMA50'] = (df_feat['Close'] / df_feat['EMA_50']) - 1
    df_feat['ATR_Price'] = df_feat['ATR'] / df_feat['Close']
    df_feat['MACD_Norm'] = df_feat['MACD'] / df_feat['Close']
    df_feat['CCI_Norm'] = df_feat['CCI'] / 100.0
    
    last_window = df_feat.iloc[-LOOKBACK_WINDOW:].copy()
    feature_cols = ['Log_Ret', 'RSI_Norm', 'Price_EMA20', 'Price_EMA50', 'ATR_Price', 'MACD_Norm', 'CCI_Norm']
    
    # Check for NaNs
    if last_window[feature_cols].isnull().values.any():
        last_window[feature_cols] = last_window[feature_cols].ffill().bfill()

    last_window[feature_cols] = scaler.transform(last_window[feature_cols])
    X = last_window[feature_cols].values.reshape(1, LOOKBACK_WINDOW, len(feature_cols))
    return torch.from_numpy(X.astype(np.float32)).to(device)

def generate_forecast(model, df_initial, scaler, days):
    print(f"Generating forecast for {days} days (approx {FORECAST_YEARS} years)...")
    
    # We work with a growing dataframe
    df_curr = calculate_indicators(df_initial).dropna()
    
    forecast_data = []
    last_date = df_curr.index[-1]
    
    # Generate future business dates
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=days)
    
    model.eval()
    
    # We predict in chunks of FORECAST_HORIZON (30 days)
    current_date_idx = 0
    
    while current_date_idx < len(future_dates):
        if current_date_idx % 30 == 0:
            print(f"Forecasting step {current_date_idx}/{days}...")
            
        # 1. Prepare Input
        X_tensor = prepare_last_window(df_curr, scaler)
        
        # 2. Predict Next 30 Days
        with torch.no_grad():
            # Output is [1, 30]
            pred_pcts = model(X_tensor).cpu().numpy().flatten()
            
        # 3. Process each predicted day in the chunk
        for i in range(len(pred_pcts)):
            if current_date_idx >= len(future_dates): break
            
            date = future_dates[current_date_idx]
            pred_pct = pred_pcts[i]
            
            last_close = df_curr['Close'].iloc[-1]
            pred_log_ret = pred_pct / 100.0
            new_price = last_close * np.exp(pred_log_ret)
            
            # 4. Append to DataFrame
            new_row = pd.DataFrame({
                'Close': [new_price],
                'High': [new_price],
                'Low': [new_price],
                'Open': [new_price],
                'Volume': [0]
            }, index=[date])
            
            df_curr = pd.concat([df_curr, new_row])
            
            # Store result
            action = "BUY" if pred_pct > 0 else "SELL"
            forecast_data.append({
                'Date': date,
                'Predicted_Price': new_price,
                'Predicted_Return_Pct': pred_pct,
                'Action': action
            })
            
            current_date_idx += 1
            
        # 5. Re-calculate indicators after the chunk (or every day? Chunk is faster)
        # Recalculating every chunk (30 days) is efficient and accurate enough
        calc_window = df_curr.iloc[-300:].copy()
        
        if 'ATR' in df_curr.columns:
            prev_atr = df_curr['ATR'].iloc[-2]
        else:
            prev_atr = 0
            
        calc_window = calculate_indicators(calc_window)
        
        if 'ATR' in calc_window.columns:
            calc_window.iloc[-1, calc_window.columns.get_loc('ATR')] = prev_atr
        
        # Update indicators for the newly added rows
        cols_to_update = ['RSI', 'EMA_20', 'EMA_50', 'ATR', 'MACD', 'CCI']
        for col in cols_to_update:
            if col not in df_curr.columns:
                df_curr[col] = np.nan
        
        # We need to update the last 'chunk_size' rows
        # But simpler to just update the tail that matches calc_window
        # Actually, we need accurate indicators for the NEXT prediction.
        # So we must update the indicators for the rows we just added.
        
        # Align indices
        common_indices = df_curr.index.intersection(calc_window.index)
        df_curr.loc[common_indices, cols_to_update] = calc_window.loc[common_indices, cols_to_update]
        
    return pd.DataFrame(forecast_data)

# ==========================================
# FINE-TUNING
# ==========================================
def fine_tune_model(model, df, scaler, epochs=10):
    print(f"Fine-tuning model on {TICKER} specific data...")
    
    # Prepare data for this specific ticker
    df_feat = df.copy()
    df_feat['Log_Ret'] = np.log(df_feat['Close'] / df_feat['Close'].shift(1)) * 100
    df_feat['RSI_Norm'] = df_feat['RSI'] / 100.0
    df_feat['Price_EMA20'] = (df_feat['Close'] / df_feat['EMA_20']) - 1
    df_feat['Price_EMA50'] = (df_feat['Close'] / df_feat['EMA_50']) - 1
    df_feat['ATR_Price'] = df_feat['ATR'] / df_feat['Close']
    df_feat['MACD_Norm'] = df_feat['MACD'] / df_feat['Close']
    df_feat['CCI_Norm'] = df_feat['CCI'] / 100.0
    
    df_feat.dropna(inplace=True)
    
    feature_cols = ['Log_Ret', 'RSI_Norm', 'Price_EMA20', 'Price_EMA50', 'ATR_Price', 'MACD_Norm', 'CCI_Norm']
    df_feat[feature_cols] = scaler.transform(df_feat[feature_cols])
    
    data_values = df_feat[feature_cols].values
    target_values = df_feat['Log_Ret'].values
    
    X_list, y_list = [], []
    for i in range(LOOKBACK_WINDOW, len(df_feat) - FORECAST_HORIZON):
        X_list.append(data_values[i-LOOKBACK_WINDOW:i])
        y_list.append(target_values[i:i+FORECAST_HORIZON])
        
    if not X_list:
        print("Not enough data to fine-tune.")
        return model
        
    X_tensor = torch.from_numpy(np.array(X_list, dtype=np.float32)).to(device)
    y_tensor = torch.from_numpy(np.array(y_list, dtype=np.float32)).to(device)
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) # Very low LR for fine-tuning
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 2 == 0:
            print(f"Fine-tune Epoch {epoch}/{epochs}, Loss: {loss.item():.6f}")
            
    return model

# ==========================================
# VISUALIZATION & SAVING
# ==========================================
def plot_forecast(history_df, forecast_df):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 1. Historical Data
    # Show last 1 year of history
    history_subset = history_df.iloc[-252:]
    ax.plot(history_subset.index, history_subset['Close'], color='white', linewidth=1.5, label='Historical Data')
    
    # 2. Forecast Data
    ax.plot(forecast_df['Date'], forecast_df['Predicted_Price'], color='#00ff00', linestyle='--', linewidth=2, label='Forecast (2 Years)')
    
    # 3. Buy/Sell Points on Forecast
    # Filter for significant moves to avoid clutter
    # Only show buy if return > 0.5% or sell if return < -0.5% (example)
    # Or just show all transitions
    
    # Let's show points where action changes or significant trend
    # For clarity, let's just mark the "Buy" zones with a green background and "Sell" with red?
    # Or just scatter points.
    
    buy_points = forecast_df[forecast_df['Predicted_Return_Pct'] > 0]
    sell_points = forecast_df[forecast_df['Predicted_Return_Pct'] < 0]
    
    # Downsample markers for visibility (every 10th day maybe?)
    # ax.scatter(buy_points['Date'][::10], buy_points['Predicted_Price'][::10], color='green', marker='^', alpha=0.5, s=30)
    # ax.scatter(sell_points['Date'][::10], sell_points['Predicted_Price'][::10], color='red', marker='v', alpha=0.5, s=30)

    # 4. Buy & Hold Reference (Projected from start of forecast)
    start_price = history_subset['Close'].iloc[-1]
    # Simple CAGR of 12% line for comparison
    days = len(forecast_df)
    bnh_end_price = start_price * (1.12 ** FORECAST_YEARS) # 12% annual growth
    bnh_line = np.linspace(start_price, bnh_end_price, days)
    ax.plot(forecast_df['Date'], bnh_line, color='gray', linestyle=':', label='Ref: 12% Annual Growth')

    ax.set_title(f"{TICKER} - 2 Year Price Forecast", fontsize=16, color='white')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(color='gray', linestyle=':', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Please run train_forecast_model.py first.")
        exit()
        
    print(f"Loading model on {device}...")
    model = MultiStepLSTM(output_size=FORECAST_HORIZON).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    scaler = joblib.load(SCALER_PATH)
    
    # 1. Get Data
    df_history = get_recent_data(TICKER)
    
    # 1.5 Fine-tune on recent data
    model = fine_tune_model(model, df_history, scaler, epochs=10)
    
    # 2. Generate Forecast
    forecast_df = generate_forecast(model, df_history, scaler, FORECAST_DAYS)
    
    # 3. Save
    forecast_df.to_csv(FORECAST_FILE, index=False)
    print(f"Forecast saved to {FORECAST_FILE}")
    
    # 4. Plot
    plot_forecast(df_history, forecast_df)
