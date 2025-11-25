import pandas as pd
import yfinance as yf
import os

PORTFOLIO_FILE = "portfolio.csv"

def test_portfolio_fetching():
    if os.path.exists(PORTFOLIO_FILE):
        portfolio_df = pd.read_csv(PORTFOLIO_FILE)
        print("Portfolio Loaded:")
        print(portfolio_df)
    else:
        print("Portfolio file not found.")
        return

    print("\nFetching Prices with yf.Ticker()...")
    for index, row in portfolio_df.iterrows():
        ticker = row['Ticker']
        qty = float(row['Quantity'])
        buy_price = float(row['Buy_Price'])
        
        try:
            # New Logic
            dat = yf.Ticker(ticker)
            df_live = dat.history(period="5d", auto_adjust=True)
            
            if df_live.empty:
                print(f"Warning: Empty data for {ticker}")
                continue
                
            current_price = float(df_live['Close'].iloc[-1])
            
            print(f"Ticker: {ticker}, Buy: {buy_price}, Current: {current_price}")
            
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")

if __name__ == "__main__":
    test_portfolio_fetching()
