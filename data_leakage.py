import pandas as pd
import numpy as np

# 1. Load the split files
try:
    X_train = pd.read_csv('X_train.csv')
    X_test  = pd.read_csv('X_test.csv')
    y_train = pd.read_csv('y_train.csv').squeeze()
    y_test  = pd.read_csv('y_test.csv').squeeze()
    print("📂 Files loaded successfully for audit.\n")
except FileNotFoundError as e:
    print(f"❌ Error: {e}. Make sure X_train.csv and X_test.csv exist in this directory.")
    exit()

print("=" * 60)
print("             DATA LEAKAGE AUDIT REPORT")
print("=" * 60)

# ------------------------------------------------------------
# TEST 1: Row Overlap (Train-Test Contamination)
# ------------------------------------------------------------
print("\n🔍 TEST 1: Checking for Row Overlap...")

# Merge train and test to find duplicate rows based on all features
combined = pd.concat([X_train, X_test])
total_duplicates = combined.duplicated().sum()

print(f"   ↳ Total rows in Training set: {X_train.shape[0]}")
print(f"   ↳ Total rows in Testing set:  {X_test.shape[0]}")
print(f"   ↳ Duplicate rows across sets: {total_duplicates}")

if total_duplicates > 0:
    print("   ⚠️  WARNING: Identical rows exist in both train and test sets! This will fake high performance.")
else:
    print("   ✅ CLEAN: No exact row duplicates found between Train and Test splits.")

# ------------------------------------------------------------
# TEST 2: Target Leakage (Target Proxy Check)
# ------------------------------------------------------------
print("\n🔍 TEST 2: Checking for Target Leakage in Features...")

# Check if the target itself or high-correlation proxies are in features
suspicious_keywords = ['target', 'disruption', 'label', 'class', 'high_disruption_p']
found_suspicious = [col for col in X_train.columns if any(kw in col.lower() for kw in suspicious_keywords)]

print(f"   ↳ Features present: {list(X_train.columns)}")

if found_suspicious:
    print(f"   ❌ CRITICAL LEAKAGE: Suspicious columns found in features: {found_suspicious}")
    print("      You must drop these columns from X_train and X_test before training!")
else:
    print("   ✅ CLEAN: No obvious target/outcome columns found inside the feature matrix.")

# ------------------------------------------------------------
# TEST 3: High Feature Correlation with Target
# ------------------------------------------------------------
print("\n🔍 TEST 3: Checking Feature Correlation with Target...")

# Combine features and target temporarily to check correlation matrix
train_temp = X_train.copy()
train_temp['TARGET_VARIABLE'] = y_train

correlations = train_temp.corr()['TARGET_VARIABLE'].drop('TARGET_VARIABLE').abs()
print("   ↳ Absolute Pearson Correlation with Target:")
for col, val in correlations.items():
    status = "⚠️  EXTREMELY HIGH" if val > 0.95 else "Normal"
    print(f"     - {col:<15} : {val:.4f} ({status})")

extreme_leakers = correlations[correlations > 0.95].index.tolist()
if extreme_leakers:
    print(f"\n   ⚠️  WARNING: Features {extreme_leakers} have >95% correlation with the target.")
    print("      If these features are calculated *after* a disruption happens, they are causing leakage!")
else:
    print("\n   ✅ CLEAN: No individual feature has an unnaturally high (>95%) correlation score.")

print("\n" + "=" * 60)
print("             AUDIT COMPLETE")
print("=" * 60)