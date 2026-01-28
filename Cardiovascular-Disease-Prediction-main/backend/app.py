import os
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")

# Load model and scaler
model = joblib.load(os.path.join(MODEL_DIR, "logistic_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

@app.route("/")
def index():
    return jsonify({
        "message": "Cardiovascular Disease Prediction API is live!",
        "endpoints": {
            "POST /predict": "Send JSON data to get prediction",
            "GET /health": "Check if the service is running"
        }
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    features = np.array([[
        float(data["age"]) * 365.25,
        float(data["gender"]),
        float(data["height"]),
        float(data["weight"]),
        float(data["ap_hi"]),
        float(data["ap_lo"]),
        float(data["cholesterol"]),
        float(data["gluc"]),
        float(data["smoke"]),
        float(data["alco"]),
        float(data["active"])
    ]])

    scaled = scaler.transform(features)
    prob = model.predict_proba(scaled)[0][1]

    return jsonify({
        "percentage": round(prob * 100, 2),
        "risk": "High" if prob >= 0.5 else "Low"
    })

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
