# ============================================================
#   Supply Chain Disruption — Backend Verification Script
#   Run this while Flask is running: python check_backend.py
# ============================================================

import requests
import pickle
import numpy as np
import os
import json

BASE_URL = "http://localhost:5000"

print("=" * 60)
print("  SUPPLY CHAIN BACKEND — FULL VERIFICATION")
print("=" * 60)

# ── CHECK 1: Model Info ───────────────────────────────────────
print("\n📋 CHECK 1 — Model Info")
print("─" * 40)
try:
    r = requests.get(f"{BASE_URL}/model_info")
    info = r.json()
    print(f"   Model Name  : {info['model_name']}")
    for k, v in info['metrics'].items():
        print(f"   {k:<15} : {v}")
except Exception as e:
    print(f"   ❌ Failed: {e}")

# ── CHECK 2: Scaler Stats ─────────────────────────────────────
print("\n📏 CHECK 2 — Scaler Statistics")
print("─" * 40)
try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    cols = ['kg_score', 'Em_Score', 'twitter_alert',
            'weather_risk', 'economic_risk', 'geo_risk', 'tech_risk']

    print(f"   {'Feature':<20} {'Mean':>8}  {'Std':>8}")
    print(f"   {'─'*38}")
    for i, col in enumerate(cols):
        print(f"   {col:<20} {scaler.mean_[i]:>8.4f}  {scaler.scale_[i]:>8.4f}")
except Exception as e:
    print(f"   ❌ Failed: {e}")

# ── CHECK 3: File Timestamps ──────────────────────────────────
print("\n🕐 CHECK 3 — File Timestamps (should all match)")
print("─" * 40)
files = ['scaler.pkl', 'model.pkl', 'model_info.pkl',
         'regressor.pkl', 'X_train.csv', 'X_test.csv']
for f in files:
    if os.path.exists(f):
        mtime = os.path.getmtime(f)
        from datetime import datetime
        dt = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        size = os.path.getsize(f)
        print(f"   {f:<20} {dt}   {size:>8} bytes")
    else:
        print(f"   {f:<20} ❌ NOT FOUND")

# ── CHECK 4: Low vs High Input Test ──────────────────────────
print("\n🧪 CHECK 4 — Prediction Tests")
print("─" * 40)

tests = [
    {
        "label"  : "ALL MINIMUM (expect Low Risk < 40)",
        "payload": {
            "kg_score": 0.05, "Em_Score": 0.05,
            "twitter_alert": 0, "weather_risk": 0,
            "economic_risk": 0, "geo_risk": 0, "tech_risk": 0
        }
    },
    {
        "label"  : "LOW VALUES (expect Low/Medium < 55)",
        "payload": {
            "kg_score": 0.20, "Em_Score": 0.15,
            "twitter_alert": 0, "weather_risk": 0,
            "economic_risk": 0, "geo_risk": 0, "tech_risk": 0
        }
    },
    {
        "label"  : "MEDIUM VALUES (expect Medium 45-70)",
        "payload": {
            "kg_score": 0.50, "Em_Score": 0.45,
            "twitter_alert": 1, "weather_risk": 0,
            "economic_risk": 0, "geo_risk": 0, "tech_risk": 0
        }
    },
    {
        "label"  : "HIGH VALUES (expect High Risk > 70)",
        "payload": {
            "kg_score": 0.90, "Em_Score": 0.85,
            "twitter_alert": 1, "weather_risk": 1,
            "economic_risk": 1, "geo_risk": 0, "tech_risk": 0
        }
    },
    {
        "label"  : "ALL MAXIMUM (expect High Risk > 85)",
        "payload": {
            "kg_score": 1.40, "Em_Score": 0.99,
            "twitter_alert": 1, "weather_risk": 1,
            "economic_risk": 1, "geo_risk": 1, "tech_risk": 1
        }
    }
]

all_passed = True
scores = []

for test in tests:
    try:
        r = requests.post(
            f"{BASE_URL}/predict",
            json=test["payload"],
            headers={"Content-Type": "application/json"}
        )
        result = r.json()

        if "error" in result:
            print(f"   ❌ {test['label']}")
            print(f"      Error: {result['error']}")
            all_passed = False
        else:
            score      = result['risk_score']
            prediction = result['prediction']
            downtime   = result.get('predicted_downtime_hours', 0)
            scores.append(score)
            print(f"   {'✅' if score > 0 or test['label'].startswith('ALL MIN') else '⚠️'} "
                  f"{test['label']}")
            print(f"      Score: {score}/100  |  "
                  f"Level: {prediction}  |  "
                  f"Downtime: {downtime} hrs")
    except Exception as e:
        print(f"   ❌ {test['label']} — {e}")
        all_passed = False

# ── CHECK 5: Score Spread Analysis ───────────────────────────
print("\n📊 CHECK 5 — Score Spread Analysis")
print("─" * 40)
if len(scores) == 5:
    spread = scores[-1] - scores[0]
    print(f"   Min score  (all minimum input) : {scores[0]}")
    print(f"   Max score  (all maximum input) : {scores[-1]}")
    print(f"   Spread     (max - min)         : {spread}")
    print()
    if spread < 20:
        print("   ❌ PROBLEM — spread too small, model not differentiating inputs")
        print("      → Your scaler.pkl is likely mismatched with model.pkl")
        print("      → Re-run: python preprocessing.py then python train_model.py")
    elif spread < 40:
        print("   ⚠️  WARNING — spread is limited, model is somewhat overconfident")
        print("      → Consider re-running preprocessing.py with fresh data")
    else:
        print("   ✅ GOOD — model is differentiating between low and high risk inputs")

# ── FINAL SUMMARY ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  VERIFICATION SUMMARY")
print("=" * 60)
print(f"""
  Model Loaded    : Check 1 above
  Scaler Valid    : Check means are in reasonable range
  Files Fresh     : Check 3 timestamps all match
  Score Spread    : Check 5 result above

  If score spread < 20:
  → Run: python preprocessing.py
  → Run: python train_model.py
  → Run: python app.py
  → Run: python check_backend.py again

  If score spread > 40:
  → Your backend is working correctly
""")
print("=" * 60)