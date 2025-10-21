from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import random
import os

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return send_file(os.path.join(os.path.dirname(__file__), 'index.html'))

@app.route('/get_candle', methods=['GET'])
def get_candle():
    """Simulate candlestick data (Open, High, Low, Close)."""
    open_price = round(random.uniform(100, 110), 2)
    close_price = round(open_price + random.uniform(-1, 1), 2)
    high_price = max(open_price, close_price) + round(random.uniform(0, 0.5), 2)
    low_price = min(open_price, close_price) - round(random.uniform(0, 0.5), 2)

    return jsonify({
        "open": open_price,
        "high": high_price,
        "low": low_price,
        "close": close_price
    })

@app.route('/place_trade', methods=['POST'])
def place_trade():
    """Simulate placing a trade automatically when trend detected."""
    data = request.json
    trend = data.get("trend", "neutral")
    decision = "HOLD"

    if trend == "uptrend":
        decision = "BUY"
    elif trend == "downtrend":
        decision = "SELL"

    return jsonify({
        "trend": trend,
        "trade_decision": decision,
        "message": f"Trade executed: {decision}"
    })

if __name__ == '__main__':
    app.run(debug=True)
