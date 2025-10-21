import pandas as pd
import numpy as np
import joblib
import os
from bot.feature_engineering import add_features
import xgboost as xgb

def run_prediction():
    """
    Loads the trained model and makes a prediction on sample live data.
    """
    try:
        # Check if the model file exists
        model_path = "model/xgb_model.joblib"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model file not found at '{model_path}'. Please run train_model.py first.")

        # Load the trained model
        model = joblib.load(model_path)
        print("✅ Model loaded successfully!")

        # --- Simulate a new live data point ---
        # NOTE: In a real-world scenario, you would fetch real-time data here
        # For this example, we'll create a single new data row
        # To make a prediction, the new data must have the same features as the training data
        
        # Create a sample DataFrame with OHLCV data for the new data point
        sample_live_data = pd.DataFrame([{
            "open": 105.5,
            "high": 106.0,
            "low": 104.5,
            "close": 105.8,
            "volume": 5500
        }])
        
        # Add a placeholder for 'Target' so feature engineering runs smoothly
        sample_live_data['Target'] = 0
        
        # We need more data for the rolling features, so we'll load the last
        # 100 rows of the historical data to generate the features correctly.
        historical_path = "data/historical_data.csv"
        if not os.path.exists(historical_path):
            raise FileNotFoundError(f"❌ Historical data file not found at '{historical_path}'. Please run generate_data.py first.")
        
        # Read the last 100 rows and append the new live data point
        historical_data = pd.read_csv(historical_path).tail(100)
        combined_df = pd.concat([historical_data, sample_live_data], ignore_index=True)
        
        # Add features to the combined DataFrame
        processed_df = add_features(combined_df)
        
        # --- FIX START ---
        # Get the feature columns from the processed DataFrame, excluding the Target column
        feature_cols = [col for col in processed_df.columns if col != "Target"]

        # Get the last row, which is our new data point, and select only the feature columns
        live_data_row = processed_df.tail(1)[feature_cols]
        # --- FIX END ---

        # Make prediction
        dmatrix_live = xgb.DMatrix(live_data_row)
        prediction_prob = model.predict(dmatrix_live)[0]
        prediction = (prediction_prob > 0.5).astype(int)

        print("-" * 30)
        print("Live Data Prediction:")
        print(f"Prediction Probability: {prediction_prob:.4f}")
        
        if prediction == 1:
            print("📈 Prediction: Up (1)")
        else:
            print("📉 Prediction: Down (0)")
        
        print("-" * 30)

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    run_prediction()
