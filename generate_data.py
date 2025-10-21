import pandas as pd
import numpy as np
import os

def create_sample_data(filename: str = "data/historical_data.csv"):
    """
    Generates a sample CSV file with OHLCV and Target columns.
    
    This script creates a dummy dataset that is suitable for financial machine learning
    tasks, including technical indicators and candlestick pattern features.
    """
    print("⏳ Generating sample data...")
    
    # Create the 'data' directory if it doesn't exist
    if not os.path.exists("data"):
        os.makedirs("data")
        
    # Generate a sample DataFrame
    np.random.seed(42)
    rows = 2000
    
    # Generate price data
    prices = 100 + np.random.randn(rows).cumsum()
    
    df = pd.DataFrame({
        "open": prices,
        "high": prices + np.abs(np.random.randn(rows)),
        "low": prices - np.abs(np.random.randn(rows)),
        "close": prices + np.random.randn(rows),
        "volume": np.random.randint(1000, 10000, size=rows)
    })
    
    # Ensure close is within high/low
    df["close"] = df.apply(lambda row: np.clip(row["close"], row["low"], row["high"]), axis=1)
    
    # Create a simple binary target: 1 if the next candle's close is higher than current, 0 otherwise
    df["Target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df.dropna(inplace=True)

    # Save to CSV
    df.to_csv(filename, index=False)
    
    print(f"✅ Sample data saved successfully to '{filename}'")

if __name__ == "__main__":
    create_sample_data()
