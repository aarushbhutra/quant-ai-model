import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# ==========================================
# 1. ADVANCED CONFIGURATION
# ==========================================
TICKER = "TCS.NS"
START_DATE = "2010-01-01"  # Maximum history for better training
INITIAL_CAPITAL = 100_000

# OPTIMIZED ML PARAMS
# We lower the threshold but add a Trend Filter to keep it safe
PROB_THRESHOLD = 0.52      # 52% confidence is enough if trend is up
TRAINING_WINDOW_YEARS = 3  # Years of data to train on before starting

# ==========================================
# 2. FEATURE ENGINEERING (The "Alpha")
# ==========================================
def prepare_rich_data(ticker):
    print(f"Fetching Deep Data for {ticker}...")
    df = yf.download(ticker, start=START_DATE, progress=False, auto_adjust=True)
    
    if isinstance(df.columns, pd.MultiIndex):
        # Handle different MultiIndex structures (Price, Ticker) vs (Ticker, Price)
        if 'Close' in df.columns.get_level_values(0):
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = df.columns.get_level_values(1)

    # --- 1. Normalized Momentum (RSI & ROC) ---
    df['RSI'] = df.ta.rsi(length=14) / 100.0  # Normalize to 0-1
    df['ROC'] = df.ta.roc(length=10) # Rate of Change

    # --- 2. Trend Interaction (Price vs EMAs) ---
    df['EMA_20'] = df.ta.ema(length=20)
    df['EMA_50'] = df.ta.ema(length=50)
    df['EMA_200'] = df.ta.ema(length=200)
    
    # Ratios are better for ML than raw prices
    df['Dist_EMA20'] = (df['Close'] / df['EMA_20']) - 1
    df['Dist_EMA200'] = (df['Close'] / df['EMA_200']) - 1

    # --- 3. Volatility Squeeze (Bollinger Bands) ---
    bb = df.ta.bbands(length=20, std=2)
    # Bandwidth: Narrow bandwidth = squeeze coming
    # Dynamically identify columns to avoid version mismatch errors (e.g. BBU_20_2.0 vs BBU_20_2)
    bbu = [c for c in bb.columns if c.startswith('BBU')][0]
    bbl = [c for c in bb.columns if c.startswith('BBL')][0]
    df['BB_Width'] = (bb[bbu] - bb[bbl]) / df['EMA_20'] 

    # --- 4. The TARGET (Prediction Goal) ---
    # We predict: Will Close be higher tomorrow? (Binary 1 or 0)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    return df.dropna()

# ==========================================
# 3. ROLLING WINDOW TRAINING (The "Brain")
# ==========================================
def get_ml_signals(df):
    print("Training Rolling Model (Simulating live trading)...")
    
    predictors = ['RSI', 'ROC', 'Dist_EMA20', 'Dist_EMA200', 'BB_Width']
    
    # Random Forest: More estimators, limited depth to prevent overfitting
    model = RandomForestClassifier(n_estimators=200, min_samples_split=20, max_depth=10, random_state=42)
    
    predictions = []
    
    # Walk-Forward Validation
    # We start trading after 'TRAINING_WINDOW_YEARS' of data
    start_index = int(252 * TRAINING_WINDOW_YEARS) 
    
    # Retrain every Quarter (60 trading days)
    step = 60 
    
    for i in range(start_index, len(df), step):
        # Sliding Window: Train on ALL past data available up to 'i'
        train = df.iloc[:i].copy()
        test = df.iloc[i:i+step].copy()
        
        if len(test) == 0: break
        
        model.fit(train[predictors], train['Target'])
        
        # Get Probability of "Up" move (Class 1)
        probs = model.predict_proba(test[predictors])[:, 1]
        
        combined = pd.DataFrame(dict(actual=test['Target'], probability=probs), index=test.index)
        predictions.append(combined)

    return pd.concat(predictions)

# ==========================================
# 4. HYBRID BACKTEST ENGINE
# ==========================================
def run_hybrid_strategy(df, pred_df):
    print("\nRunning Ensemble Backtest...")
    
    # Align Dataframes
    data = df.loc[pred_df.index].copy()
    data['ML_Prob'] = pred_df['probability']
    
    capital = INITIAL_CAPITAL
    shares = 0
    in_position = False
    
    portfolio_values = []
    trades = []
    
    # TRAILING STOP LOGIC
    highest_price = 0
    trailing_stop_pct = 0.03 # 3% Trailing Stop
    
    for date, row in data.iterrows():
        price = row['Close']
        ml_score = row['ML_Prob']
        trend_ok = price > row['EMA_200'] # FILTER: Only buy if above 200 EMA
        
        # --- SELL LOGIC ---
        if in_position:
            # Update High Water Mark
            if price > highest_price:
                highest_price = price
            
            # Dynamic Stop Loss
            stop_price = highest_price * (1 - trailing_stop_pct)
            
            # Exit if Stop hit OR ML says "Market looks bad" (Prob < 0.40)
            if price < stop_price or ml_score < 0.40:
                revenue = shares * price
                capital += revenue
                trades.append({'Date': date, 'Type': 'SELL', 'Price': price, 'Balance': capital})
                shares = 0
                in_position = False
        
        # --- BUY LOGIC ---
        # 1. ML says > 52% chance
        # 2. Trend Filter (Price > 200 EMA) is True
        if not in_position and ml_score > PROB_THRESHOLD and trend_ok:
            shares = int(capital // price)
            cost = shares * price
            capital -= cost
            in_position = True
            highest_price = price # Reset Stop logic
            trades.append({'Date': date, 'Type': 'BUY', 'Price': price, 'Balance': capital})
            
        # Track Equity
        curr_val = capital + (shares * price)
        portfolio_values.append(curr_val)

    return data, portfolio_values, trades

# ==========================================
# 5. VISUALIZATION
# ==========================================
def plot_results(data, portfolio_values, trades):
    final_val = portfolio_values[-1]
    net_profit = final_val - INITIAL_CAPITAL
    ret_pct = (net_profit / INITIAL_CAPITAL) * 100
    
    # Benchmark (Buy & Hold)
    bnh_shares = INITIAL_CAPITAL / data['Close'].iloc[0]
    bnh_values = data['Close'] * bnh_shares
    bnh_ret = ((bnh_values.iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100

    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    # 1. Equity Curve
    ax1.plot(data.index, portfolio_values, color='#00ff00', linewidth=2, label=f'Ensemble Algo ({ret_pct:.1f}%)')
    ax1.plot(data.index, bnh_values, color='gray', alpha=0.5, linestyle='--', label=f'Buy & Hold ({bnh_ret:.1f}%)')
    ax1.set_title(f"Final Balance: ₹{final_val:.0f} | Net Profit: ₹{net_profit:.0f}", fontsize=14, color='white')
    ax1.legend()
    
    # 2. Entries & Exits
    ax2.plot(data.index, data['Close'], color='gray', alpha=0.5)
    
    buy_dates = [t['Date'] for t in trades if t['Type'] == 'BUY']
    buy_prices = [t['Price'] for t in trades if t['Type'] == 'BUY']
    ax2.scatter(buy_dates, buy_prices, color='#00ff00', marker='^', s=80, label='Buy')
    
    sell_dates = [t['Date'] for t in trades if t['Type'] == 'SELL']
    sell_prices = [t['Price'] for t in trades if t['Type'] == 'SELL']
    ax2.scatter(sell_dates, sell_prices, color='red', marker='v', s=80, label='Sell')
    
    ax2.set_title("Trade Execution Points")
    plt.tight_layout()
    plt.show()

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    try:
        # 1. Get Data
        df = prepare_rich_data(TICKER)
        
        # 2. ML Predictions
        pred_df = get_ml_signals(df)
        
        # 3. Hybrid Backtest
        data, port_vals, trades = run_hybrid_strategy(df, pred_df)
        
        # 4. Show
        plot_results(data, port_vals, trades)
        print(f"Total Trades: {len(trades)}")
        
    except Exception as e:
        print(f"Error: {e}")