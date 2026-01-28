import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Points to your frontend folder without 'template' or 'static' definitions
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'frontend'))

def load_models():
    try:
        model_path = os.path.join(BASE_DIR, '..', 'model', 'logistic_model.pkl')
        scaler_path = os.path.join(BASE_DIR, '..', 'model', 'scaler.pkl')
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        print("Double-check that your .pkl files are real files and not Git LFS pointers.\n")
        return None, None

model, scaler = load_models()

@app.route('/')
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(FRONTEND_DIR, path)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded on server'}), 500
    try:
        data = request.get_json()
        # Ensure input matches your training features
        features = np.array([[
            float(data['age']) * 365.25, float(data['gender']), float(data['height']), 
            float(data['weight']), float(data['ap_hi']), float(data['ap_lo']),
            float(data['cholesterol']), float(data['gluc']), float(data['smoke']),
            float(data['alco']), float(data['active'])
        ]])
        scaled = scaler.transform(features)
        prob = model.predict_proba(scaled)[0][1]
        
        return jsonify({
            'percentage': round(prob * 100, 2),
            'risk': "High" if prob >= 0.5 else "Low"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)