# ============================================================
#   Supply Chain Disruption Alerts — Preprocessing
#   Dataset : Supplier_Disruption_LogTable_Coded.csv
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

print("=" * 55)
print("  SUPPLY CHAIN DISRUPTION — PREPROCESSING")
print("=" * 55)

# ============================================================
# STEP 1 — Load Dataset
# ============================================================
print("\n📂 STEP 1 — Loading Dataset...")

df = pd.read_csv('Supplier_Disruption_LogTable_Coded.csv')

print(f"    ✅ Loaded successfully")
print(f"    Rows    : {df.shape[0]}")
print(f"    Columns : {df.shape[1]}")
 
# ============================================================
# STEP 1.5 — Add Realistic Noise to Numeric Features
# ============================================================
print("\n🎲 STEP 1.5 — Adding Realistic Noise...")

import numpy as np
np.random.seed(42)

# FIX: Only add noise to feature inputs, NEVER to downtime_hours which is your target!
noise_cols = ['kg_score', 'Em_Score']
for col in noise_cols:
    noise = np.random.normal(0, df[col].std() * 0.15, size=len(df))
    df[col] = df[col] + noise

print(f"    ✅ Added 15% noise to: {noise_cols}")

# ============================================================
# STEP 2 — Drop Unnecessary Columns
# ============================================================
print("\n🗑️  STEP 2 — Dropping Unnecessary Columns...")

# FIX: Isolate clean targets from the dataframe BEFORE dropping columns
y_downtime_target = df['downtime_hours'].copy()
y_high_disruption_p = df['high_disruption_p'].copy()

# 'date'     — just a date record, not useful for prediction
# 'supplier' — text name, not a numeric feature
# FIX: Drop target columns here so they do not leak into your scaled features (X)
df = df.drop(columns=['date', 'supplier', 'high_disruption_p', 'downtime_hours'])

print(f"    Dropped  : 'date', 'supplier', 'high_disruption_p', 'downtime_hours'")
print(f"    Remaining columns : {list(df.columns)}")

# ============================================================
# STEP 3 — Create Binary Label Column
# ============================================================
print("\n🏷️  STEP 3 — Creating Label Column...")

# high_disruption_p is a probability value (0.0 to 1.0)
# We convert it to binary:
#   1 = High Risk  (probability >= 0.5)
#   0 = Low Risk   (probability <  0.5)

# FIX: We use our saved target variable here since high_disruption_p was dropped from df
df['disruption'] = (y_high_disruption_p >= 0.65).astype(int)

low  = (df['disruption'] == 0).sum()
high = (df['disruption'] == 1).sum()

print(f"    ✅ Label 'disruption' created from 'high_disruption_p'")
print(f"    Threshold : >= 0.65 = High Risk (1)")
print(f"    Low Risk  (0) : {low}  records  ({low/len(df)*100:.1f}%)")
print(f"    High Risk (1) : {high} records  ({high/len(df)*100:.1f}%)")

# ============================================================
# STEP 4 — Define Features (X) and Label (y)
# ============================================================
print("\n⚙️  STEP 4 — Defining Features and Label...")

# Features — all columns the model will learn from
# FIX: Removed 'downtime_hours' from this list since it's dropped from df and is our target
feature_cols = [
    'kg_score',       # Knowledge graph disruption score
    'Em_Score',       # Emergency/event score
    'twitter_alert',  # Social media alert flag (0 or 1)
    'weather_risk',   # Weather risk flag (0 or 1)
    'economic_risk',  # Economic risk flag (0 or 1)
    'geo_risk',       # Geopolitical risk flag (0 or 1)
    'tech_risk'       # Technical risk flag (0 or 1)
]

X = df[feature_cols]   # Input features
y = df['disruption']   # Target label

print(f"    ✅ Features (X) : {list(X.columns)}")
print(f"    ✅ Label    (y) : 'disruption'")
print(f"    X shape : {X.shape}")
print(f"    y shape : {y.shape}")

# ============================================================
# STEP 5 — Scale Numeric Features
# ============================================================
print("\n📏 STEP 5 — Scaling Numeric Features...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

print(f"    ✅ StandardScaler applied to all features")
print(f"    Before scaling — kg_score range : {df['kg_score'].min():.3f} to {df['kg_score'].max():.3f}")
print(f"    After  scaling — kg_score range : {X_scaled['kg_score'].min():.3f} to {X_scaled['kg_score'].max():.3f}")

# ============================================================
# STEP 6 — Train / Test Split
# ============================================================
print("\n✂️  STEP 6 — Splitting into Train and Test Sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y        # keeps same ratio of 0/1 in both train and test
)

print(f"    ✅ Split complete")
print(f"    Training set : {X_train.shape[0]} rows  (80%)")
print(f"    Test set     : {X_test.shape[0]} rows  (20%)")
print(f"    Train — Low Risk: {(y_train==0).sum()}  High Risk: {(y_train==1).sum()}")
print(f"    Test  — Low Risk: {(y_test==0).sum()}   High Risk: {(y_test==1).sum()}")

# ============================================================
# STEP 7 — Save Everything for Model Training
# ============================================================
print("\n💾 STEP 7 — Saving Processed Data...")

# Save train and test sets as CSV
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv',  index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv',  index=False)

# Save the scaler so Flask can use same scaling on new inputs
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print(f"    ✅ X_train.csv saved  — {X_train.shape[0]} rows")
print(f"    ✅ X_test.csv  saved  — {X_test.shape[0]} rows")
print(f"    ✅ y_train.csv saved")
print(f"    ✅ y_test.csv  saved")
print(f"    ✅ scaler.pkl  saved  — use this in Flask for live predictions")

# ── Save regression target (downtime_hours) ──────────────
print("\n   Saving regression targets...")

from sklearn.model_selection import train_test_split as tts

# FIX: Added stratify=y here so this split mirrors the exact shuffling of Step 6.
# This prevents row desynchronization between your features and regression targets.
_, _, y_train_dt, y_test_dt = tts(
    X_scaled, y_downtime_target,
    test_size=0.2,
    random_state=42,
    stratify=y
)

y_train_dt.to_csv('y_train_downtime.csv', index=False)
y_test_dt.to_csv('y_test_downtime.csv',   index=False)

print(f"    ✅ y_train_downtime.csv saved — {len(y_train_dt)} rows")
print(f"    ✅ y_test_downtime.csv  saved — {len(y_test_dt)} rows")
print(f"    Downtime range: {y_downtime_target.min():.1f} to {y_downtime_target.max():.1f} hours")

# ============================================================
# STEP 8 — Final Summary
# ============================================================
print("\n" + "=" * 55)
print("  PREPROCESSING COMPLETE — SUMMARY")
print("=" * 55)
print(f"""
  Dataset        : Supplier_Disruption_LogTable_Coded.csv
  Total Records  : {len(df)}
  Features Used  : {len(feature_cols)}
  Label Column   : disruption (0 = Low Risk, 1 = High Risk)

  Columns Dropped : date, supplier, high_disruption_p, downtime_hours
  Scaling Applied : StandardScaler
  Split Ratio     : 80% Train / 20% Test

  Training Rows  : {X_train.shape[0]}
  Test Rows      : {X_test.shape[0]}

  Files Created:
  → X_train.csv
  → X_test.csv
  → y_train.csv
  → y_test.csv
  → y_train_downtime.csv
  → y_test_downtime.csv
  → scaler.pkl
""")
print("=" * 55)