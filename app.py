# ============================================================
#   Supply Chain Disruption Alerts — Flask Backend
#   Phase 6 : Connects Two-Stage ML Pipeline to the UI
# ============================================================

from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

# GLOBAL ARCHITECTURAL TUNING
CUSTOM_THRESHOLD = 45.0  # Matches your optimized decision boundary

# ============================================================
# Load Models, Scaler, and Metadata at Startup
# ============================================================
print("\n⚙️ Loading Machine Learning Pipeline Components...")

with open('model.pkl', 'rb') as f:
    classifier = pickle.load(f)

with open('regressor.pkl', 'rb') as f:
    regressor = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('model_info.pkl', 'rb') as f:
    model_info = pickle.load(f)

print(f"   ✅ Classifier Loaded : {model_info['name']}")
print(f"   ✅ Regressor Loaded  : XGBoost Regressor (Severity Engine)")
print(f"   ✅ Input Scaler Loaded")

# FIX: Removed 'downtime_hours' from inputs. This list contains only what the user submits.
FEATURE_COLS = [
    'kg_score', 'Em_Score', 'twitter_alert', 'weather_risk',
    'economic_risk', 'geo_risk', 'tech_risk'
]

FEATURE_LABELS = {
    'kg_score'      : 'KG Score',
    'Em_Score'      : 'Emergency Score',
    'twitter_alert' : 'Twitter Alert',
    'weather_risk'  : 'Weather Risk',
    'economic_risk' : 'Economic Risk',
    'geo_risk'      : 'Geo Risk',
    'tech_risk'     : 'Tech Risk'
}

@app.route('/')
def home():
    return render_template('index.html')

# ============================================================
# ROUTE 1 — Predict Risk & Severity for a Single Supplier
# ============================================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Validate incoming fields
        missing = [col for col in FEATURE_COLS if col not in data]
        if missing:
            return jsonify({'error': f'Missing fields: {missing}'}), 400

        # Build DataFrame and scale
        input_df = pd.DataFrame([{col: float(data[col]) for col in FEATURE_COLS}])
        input_scaled = scaler.transform(input_df)

        # 1. STAGE 1: Predict Risk Score from Calibrated Classifier
        prediction_prob = classifier.predict_proba(input_scaled)[0][1]
        risk_score = round(float(prediction_prob) * 100, 1)

        # Map scores to labels based on custom thresholds
        if risk_score >= CUSTOM_THRESHOLD:
            risk_level = 'High Risk'
            is_disruption = 1
        elif risk_score >= max(0.0, CUSTOM_THRESHOLD - 20.0):
            risk_level = 'Medium Risk'
            is_disruption = 0
        else:
            risk_level = 'Low Risk'
            is_disruption = 0

        # 2. STAGE 2: Predict Downtime Severity using the Regressor
        if is_disruption == 1:
            raw_downtime = regressor.predict(input_scaled)[0]
            # Ensure no random negative background values are passed to the frontend
            predicted_downtime = round(max(0.0, float(raw_downtime)), 1)
        else:
            # Gated logic: Cleared suppliers automatically receive zero downtime
            predicted_downtime = 0.0

        # FIX: Extract importances safely across Calibrated CV ensemble estimators
        try:
            importances = np.mean([est.feature_importances_ for est in classifier.calibrated_classifiers_], axis=0)
        except AttributeError:
            # Fallback if the pipeline format differs
            importances = np.zeros(len(FEATURE_COLS))

        factors = []
        for i, col in enumerate(FEATURE_COLS):
            factors.append({
                'feature': FEATURE_LABELS[col],
                'value': float(input_df[col].iloc[0]),
                'importance': round(float(importances[i]) * 100, 2)
            })
        factors = sorted(factors, key=lambda x: x['importance'], reverse=True)

        return jsonify({
            'risk_score': risk_score,
            'prediction': risk_level,
            'is_disruption': is_disruption,
            'predicted_downtime_hours': predicted_downtime,
            'factors': factors
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================
# ROUTE 2 — Batch Prediction (Multiple Suppliers)
# ============================================================
@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    try:
        data = request.get_json()
        suppliers = data.get('suppliers', [])

        if not suppliers:
            return jsonify({'error': 'No suppliers provided'}), 400

        results = []
        for sup in suppliers:
            missing = [col for col in FEATURE_COLS if col not in sup]
            if missing:
                results.append({
                    'name': sup.get('name', 'Unknown'),
                    'error': f'Missing fields: {missing}'
                })
                continue

            input_df = pd.DataFrame([{col: float(sup[col]) for col in FEATURE_COLS}])
            input_scaled = scaler.transform(input_df)

            # Predict Risk
            prediction_prob = classifier.predict_proba(input_scaled)[0][1]
            risk_score = round(float(prediction_prob) * 100, 1)

            if risk_score >= CUSTOM_THRESHOLD:
                risk_level = 'High Risk'
                # Predict Downtime hours for high risk batch rows
                raw_dt = regressor.predict(input_scaled)[0]
                predicted_downtime = round(max(0.0, float(raw_dt)), 1)
            elif risk_score >= max(0.0, CUSTOM_THRESHOLD - 20.0):
                risk_level = 'Medium Risk'
                predicted_downtime = 0.0
            else:
                risk_level = 'Low Risk'
                predicted_downtime = 0.0

            results.append({
                'name': sup.get('name', 'Unknown'),
                'risk_score': risk_score,
                'prediction': risk_level,
                'predicted_downtime_hours': predicted_downtime
            })

        return jsonify({'results': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_info', methods=['GET'])
def get_model_info():
    return jsonify({
        'model_name': model_info['name'],
        'metrics': model_info['results']
    })

if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("   SUPPLY CHAIN DISRUPTION ALERTS — WEB SERVER")
    print("=" * 50)
    print(f"   Classifier : {model_info['name']}")
    print(f"   Regressor  : XGBoost Severity Engine")
    print(f"   Threshold  : {CUSTOM_THRESHOLD}%")
    print(f"   Running on : http://localhost:5000")
    print("=" * 50 + "\n")

    app.run(debug=True, port=5000)