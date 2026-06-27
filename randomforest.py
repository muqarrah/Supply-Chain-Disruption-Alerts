import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE

# 1. Load Data
X_train = pd.read_csv('X_train.csv')
X_test  = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').squeeze()
y_test  = pd.read_csv('y_test.csv').squeeze()

# 2. Balance using SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# 3. Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
rf_model.fit(X_train_balanced, y_train_balanced)

# 4. Get probabilities and apply custom threshold (0.45)
rf_probs = rf_model.predict_proba(X_test)[:, 1]
custom_threshold = 0.45
rf_preds = (rf_probs >= custom_threshold).astype(int)

# 5. Calculate exact summary metrics
accuracy  = accuracy_score(y_test, rf_preds) * 100
precision = precision_score(y_test, rf_preds) * 100
recall    = recall_score(y_test, rf_preds) * 100
f1        = f1_score(y_test, rf_preds)
roc_auc   = roc_auc_score(y_test, rf_probs) # uses raw probabilities for true curve evaluation

# 6. Get Confusion Matrix breakdown
tn, fp, fn, tp = confusion_matrix(y_test, rf_preds).ravel()

# 7. Print the exact terminal block report
print("────────────────────────────────────────────────────────────")
print("  STEP 5 — SELECTING BEST MODEL")
print("────────────────────────────────────────────────────────────")
print(f"   🏆 Target Model   : Random Forest")
print(f"   Accuracy        : {accuracy:.2f}%")
print(f"   Precision       : {precision:.2f}%")
print(f"   Recall          : {recall:.2f}%")
print(f"   F1 Score        : {f1:.4f}")
print(f"   ROC-AUC Score   : {roc_auc:.4f}")
print(f"\n   Applied Threshold: {custom_threshold}")
print("────────────────────────────────────────────────────────────")
print("  STEP 6 — DETAILED REPORT: RANDOM FOREST")
print("────────────────────────────────────────────────────────────")
print("\n   Classification Report:")
print(classification_report(y_test, rf_preds, target_names=['Low Risk (0)', 'High Risk (1)']))

print("   Confusion Matrix Breakdown:")
print(f"   ✅ Correct Disruption Alerts  (TP) : {tp}")
print(f"   ✅ Correct Clear Signals      (TN) : {tn}")
print(f"   ⚠️  False Alarms               (FP) : {fp}")
print(f"   ❌ Missed Disruptions          (FN) : {fn}")
print("────────────────────────────────────────────────────────────")