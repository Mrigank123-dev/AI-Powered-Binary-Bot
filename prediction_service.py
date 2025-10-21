import pandas as pd
import numpy as np
import joblib
import os
import xgboost as xgb

# ------------------ Handle flexible imports ------------------
try:
    from bot.feature_engineering import add_features
except ModuleNotFoundError:
    try:
        from bot.feature_engineering import add_features  # if inside 'bot/' package
    except ModuleNotFoundError as e:
        raise ImportError("❌ Could not import add_features. "
                          "Make sure feature_engineering.py exists in the same folder "
                          "or inside 'bot/' with __init__.py") from e

# ------------------ Load model at startup ------------------
MODEL_PATH = os.path.join("model", "xgb_model.joblib")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}. Run train_model.py first.")

model = joblib.load(MODEL_PATH)

# ------------------ Prediction function ------------------
def get_prediction_result(live_data_json):
    """
    Makes a prediction on a single live data point.

    Args:
        live_data_json (dict): Example: {"open": 100, "high": 105, "low": 98, "close": 102, "volume": 5000}

    Returns:
        dict: {"prediction_prob": float, "prediction": int}
    """
    try:
        # 1️⃣ Create a DataFrame from new live data
        sample_live_data = pd.DataFrame([live_data_json])

        # 2️⃣ Load last 100 rows of historical data
        historical_path = os.path.join("data", "historical_data.csv")
        if not os.path.exists(historical_path):
            return {"error": f"❌ Historical data file not found at {historical_path}. Run generate_data.py first."}

        historical_data = pd.read_csv(historical_path).tail(100)
        combined_df = pd.concat([historical_data, sample_live_data], ignore_index=True)

        # 3️⃣ Add features
        processed_df = add_features(combined_df)

        # 4️⃣ Keep only feature columns
        feature_cols = [col for col in processed_df.columns if col != "Target"]
        live_data_row = processed_df.tail(1)[feature_cols]

        # 5️⃣ Make prediction
        dmatrix_live = xgb.DMatrix(live_data_row)
        prediction_prob = float(model.predict(dmatrix_live)[0])
        prediction = int(prediction_prob > 0.5)

        return {"prediction_prob": prediction_prob, "prediction": prediction}

    except Exception as e:
        return {"error": f"❌ Prediction failed: {str(e)}"}
