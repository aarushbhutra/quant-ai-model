import pandas as pd
from datetime import datetime, timedelta

PORTFOLIO_FILE = "portfolio.csv"

def migrate_portfolio():
    try:
        df = pd.read_csv(PORTFOLIO_FILE)
        
        # Check if Buy_Date already exists
        if 'Buy_Date' not in df.columns:
            print("Adding Buy_Date column...")
            # Default to yesterday (2025-11-22)
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            df['Buy_Date'] = yesterday
            
            df.to_csv(PORTFOLIO_FILE, index=False)
            print(f"Migration complete. Added Buy_Date: {yesterday}")
            print(df.head())
        else:
            print("Buy_Date column already exists.")
            
    except Exception as e:
        print(f"Error migrating portfolio: {e}")

if __name__ == "__main__":
    migrate_portfolio()
