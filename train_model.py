# ============================================================
#   Supply Chain Disruption Alerts — Model Training
#   Dataset : Supplier_Disruption_LogTable_Coded.csv
#   Phase 4 : Train 4 Models + Save Best Model
# ============================================================

import pandas as pd
import numpy as np
import pickle
import time

from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from xgboost                 import XGBClassifier

# Crucial import for balancing the dataset
from imblearn.over_sampling  import SMOTE 

from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    confusion_matrix, classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Plot style ───────────────────────────────────────────────
plt.rcParams['figure.facecolor'] = '#0f1117'
plt.rcParams['axes.facecolor']   = '#1a1d27'
plt.rcParams['axes.edgecolor']   = '#2e3347'
plt.rcParams['axes.labelcolor']  = '#c8ccd8'
plt.rcParams['xtick.color']      = '#8891a8'
plt.rcParams['ytick.color']      = '#8891a8'
plt.rcParams['text.color']       = '#c8ccd8'
plt.rcParams['grid.color']       = '#2e3347'
plt.rcParams['grid.linestyle']   = '--'
plt.rcParams['grid.alpha']       = 0.5
plt.rcParams['font.family']      = 'monospace'

ACCENT = '#00e5ff'
SAFE   = '#2ed573'
WARN   = '#ffd166'
DANGER = '#ff4757'
PURPLE = '#a855f7'

print("=" * 60)
print("  SUPPLY CHAIN DISRUPTION ALERTS — MODEL TRAINING")
print("=" * 60)

# ============================================================
# STEP 1 — Load Preprocessed Data
# ============================================================
print("\n📂 STEP 1 — Loading Preprocessed Data...")

X_train = pd.read_csv('X_train.csv')
X_test  = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').squeeze()
y_test  = pd.read_csv('y_test.csv').squeeze()

print(f"   ✅ Data loaded")
print(f"   Training set : {X_train.shape[0]} rows, {X_train.shape[1]} features")
print(f"   Test set     : {X_test.shape[0]} rows,  {X_test.shape[1]} features")

# ============================================================
# STEP 1.5 — Balance Training Data with SMOTE
# ============================================================
print("\n⚖️  STEP 1.5 — Balancing Training Set via SMOTE...")
print(f"   Before SMOTE: Class 0: {X_train[y_train==0].shape[0]} rows | Class 1: {X_train[y_train==1].shape[0]} rows")

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"   ✅ SMOTE applied successfully")
print(f"   After SMOTE : Class 0: {X_train_balanced[y_train_balanced==0].shape[0]} rows | Class 1: {X_train_balanced[y_train_balanced==1].shape[0]} rows")
print(f"   Features     : {list(X_train_balanced.columns)}")

## ============================================================
# STEP 2 — Define All 4 Models
# ============================================================
print("\n🤖 STEP 2 — Defining 4 Models...")


# 2. Define the unified models dictionary cleanly
models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        C=0.5,              # regularization — prevents overfit
        random_state=42
    ),
    'Decision Tree': DecisionTreeClassifier(
        max_depth=4,        # shallow tree — prevents overfit
        random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=5,        # controlled depth
        min_samples_leaf=5, # needs 5 samples minimum per leaf
        random_state=42
    ),
    'XGBoost': XGBClassifier(
        n_estimators=80,
        max_depth=3,        # shallow — most important setting
        learning_rate=0.05, # slow learning = better generalization
        subsample=0.8,      # uses 80% of data per tree
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )
}

print(f"    ✅ 4 models defined and ready to train")
# ============================================================
# STEP 3 — Train All Models + Collect Results
# ============================================================
print("\n🏋️  STEP 3 — Training All Models...\n")

results    = {}
trained    = {}

for name, model in models.items():
    print(f"   Training: {name}...", end=' ')
    start = time.time()

    # CRITICAL FIX: Train on the BALANCED data
    model.fit(X_train_balanced, y_train_balanced)

    # CRITICAL FIX: Evaluate on the ORIGINAL, UNTOUCHED test data split
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)
    dur  = time.time() - start

    results[name] = {
        'Accuracy'  : round(acc  * 100, 2),
        'Precision' : round(prec * 100, 2),
        'Recall'    : round(rec  * 100, 2),
        'F1 Score'  : round(f1, 4),
        'Time (s)'  : round(dur, 3),
        'y_pred'    : y_pred,
        'y_prob'    : y_pred_prob,
        'cm'        : cm
    }
    trained[name] = model

    print(f"done ✅  Accuracy: {acc*100:.1f}%  Recall: {rec*100:.1f}%  ({dur:.2f}s)")

# ============================================================
# STEP 4 — Results Comparison Table
# ============================================================
print("\n" + "─" * 60)
print("  STEP 4 — MODEL COMPARISON")
print("─" * 60)

summary = pd.DataFrame({
    name: {
        'Accuracy %' : results[name]['Accuracy'],
        'Precision %': results[name]['Precision'],
        'Recall %'   : results[name]['Recall'],
        'F1 Score'   : results[name]['F1 Score'],
        'Train Time' : results[name]['Time (s)']
    }
    for name in results
}).T

print(f"\n{'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1 Score':>10}")
print("─" * 64)
for name, row in summary.iterrows():
    print(f"  {name:<20} {row['Accuracy %']:>9.1f}%  {row['Precision %']:>8.1f}%  {row['Recall %']:>8.1f}%  {row['F1 Score']:>9.4f}")

# ============================================================
# STEP 5 — Pick Best Model (by Recall then F1)
# ============================================================
print("\n" + "─" * 60)
print("  STEP 5 — SELECTING BEST MODEL")
print("─" * 60)

best_name = summary['Recall %'].idxmax()

top_recall = summary[summary['Recall %'] == summary['Recall %'].max()]
if len(top_recall) > 1:
    best_name = top_recall['F1 Score'].idxmax()

best_model   = trained[best_name]
best_results = results[best_name]

print(f"\n   🏆 Best Model   : {best_name}")
print(f"   Accuracy        : {best_results['Accuracy']}%")
print(f"   Precision       : {best_results['Precision']}%")
print(f"   Recall          : {best_results['Recall']}%")
print(f"   F1 Score        : {best_results['F1 Score']}")
print(f"\n   Reason: Highest Recall — catches the most real disruptions")

# ============================================================
# STEP 6 — Detailed Report for Best Model
# ============================================================
print("\n" + "─" * 60)
print(f"  STEP 6 — DETAILED REPORT: {best_name.upper()}")
print("─" * 60)

print(f"\n   Classification Report:")
print(classification_report(
    y_test, best_results['y_pred'],
    target_names=['Low Risk (0)', 'High Risk (1)']
))

cm = best_results['cm']
tn, fp, fn, tp = cm.ravel()
print(f"   Confusion Matrix Breakdown:")
print(f"   ✅ Correct Disruption Alerts  (TP) : {tp}")
print(f"   ✅ Correct Clear Signals      (TN) : {tn}")
print(f"   ⚠️  False Alarms               (FP) : {fp}")
print(f"   ❌ Missed Disruptions          (FN) : {fn}")

# ============================================================
# STEP 7 — Save Best Model
# ============================================================
print("\n" + "─" * 60)
print("  STEP 7 — SAVING BEST MODEL")
print("─" * 60)

with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('model_info.pkl', 'wb') as f:
    pickle.dump({'name': best_name, 'results': {
        k: v for k, v in best_results.items()
        if k not in ['y_pred', 'y_prob', 'cm']
    }}, f)

print(f"\n   ✅ model.pkl       saved — {best_name}")
print(f"   ✅ model_info.pkl saved — metadata")

# ============================================================
# STEP 8 — Charts
# ============================================================
print("\n📊 STEP 8 — Generating Charts...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Supply Chain Disruption — Model Training Results (SMOTE Balanced)',
             fontsize=13, fontweight='bold', color=ACCENT)
fig.patch.set_facecolor('#0f1117')

# ── Chart 1: Model Comparison Bar Chart ─────────────────────
ax1 = axes[0]
metrics   = ['Accuracy %', 'Precision %', 'Recall %']
x         = np.arange(len(metrics))
bar_width  = 0.18
colors     = [ACCENT, SAFE, WARN, DANGER]
model_names= list(results.keys())

for i, name in enumerate(model_names):
    vals = [results[name]['Accuracy'],
            results[name]['Precision'],
            results[name]['Recall']]
    bars = ax1.bar(x + i * bar_width, vals, bar_width,
                   label=name, color=colors[i], alpha=0.85)

ax1.set_title('Model Comparison', color=ACCENT, fontsize=11)
ax1.set_xticks(x + bar_width * 1.5)
ax1.set_xticklabels(metrics, fontsize=9)
ax1.set_ylabel('Score (%)')
ax1.set_ylim(0, 110)
ax1.legend(fontsize=7, loc='lower right')
ax1.grid(True, axis='y')
for spine in ax1.spines.values():
    spine.set_edgecolor('#2e3347')

# ── Chart 2: Confusion Matrix of Best Model ─────────────────
ax2 = axes[1]
sns.heatmap(
    cm, annot=True, fmt='d', ax=ax2,
    cmap='Blues',
    xticklabels=['Low Risk', 'High Risk'],
    yticklabels=['Low Risk', 'High Risk'],
    annot_kws={'size': 14, 'weight': 'bold'},
    linewidths=2, linecolor='#0f1117',
    cbar=False
)
ax2.set_title(f'Confusion Matrix\n{best_name}', color=ACCENT, fontsize=11)
ax2.set_xlabel('Predicted', fontsize=9)
ax2.set_ylabel('Actual', fontsize=9)

# ── Chart 3: F1 Score Comparison ────────────────────────────
ax3 = axes[2]
f1_scores  = [results[n]['F1 Score'] for n in model_names]
bar_colors = [ACCENT if n == best_name else '#2e3347' for n in model_names]
bars       = ax3.barh(model_names, f1_scores, color=bar_colors, edgecolor='#0f1117')

for bar, val in zip(bars, f1_scores):
    ax3.text(val + 0.005, bar.get_y() + bar.get_height()/2,
             f'{val:.4f}', va='center', fontsize=9,
             color=ACCENT if val == max(f1_scores) else '#8891a8')

ax3.set_title('F1 Score Comparison', color=ACCENT, fontsize=11)
ax3.set_xlabel('F1 Score')
ax3.set_xlim(0, 1.1)
ax3.grid(True, axis='x')
ax3.axvline(0.78, color=WARN, linestyle='--', linewidth=1,
            alpha=0.6, label='Target (0.78)')
ax3.legend(fontsize=8)

plt.tight_layout()
plt.savefig('model_results.png', dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
plt.show()
print("   ✅ Chart saved: model_results.png")

# ============================================================
# STEP 9 — Risk Score Demo
# ============================================================
print("\n" + "─" * 60)
print("  STEP 9 — RISK SCORE DEMO")
print("─" * 60)
print("\n   Showing risk scores for first 10 test records:\n")

probs       = best_results['y_prob']
risk_scores= (probs * 100).round(1)

print(f"   {'Record':<10} {'Risk Score':>12} {'Level':<15} {'Actual Label'}")
print("   " + "─" * 52)
for i in range(min(10, len(risk_scores))):
    score  = risk_scores[i]
    actual = 'High Risk' if y_test.iloc[i] == 1 else 'Low Risk'
    if score >= 70:
        level = '🔴 HIGH RISK'
    elif score >= 45:
        level = '🟡 MEDIUM RISK'
    else:
        level = '🟢 LOW RISK'
    print(f"   Record {i+1:<4}  {score:>9.1f}/100   {level:<15}  {actual}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("  MODEL TRAINING COMPLETE — SUMMARY")
print("=" * 60)
print(f"""
  Dataset      : Supplier_Disruption_LogTable_Coded.csv
  Models Tested: Logistic Regression, Decision Tree,
                 Random Forest, XGBoost

  Best Model   : {best_name}
  Accuracy     : {best_results['Accuracy']}%
  Precision    : {best_results['Precision']}%
  Recall       : {best_results['Recall']}%
  F1 Score     : {best_results['F1 Score']}

  Files Saved:
  → model.pkl       (use this in Flask)
  → model_info.pkl  (model metadata)
  → model_results.png
""")
print("=" * 60)

# ============================================================
# XGBOOST REGRESSOR — Predict Disruption Severity
# ============================================================
print("\n" + "=" * 60)
print("  XGBOOST REGRESSOR — ANOMALY SEVERITY PREDICTION")
print("=" * 60)

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("\n📂 Loading regression targets...")

y_train_dt = pd.read_csv('y_train_downtime.csv').squeeze()
y_test_dt  = pd.read_csv('y_test_downtime.csv').squeeze()

print(f"   Train target range : {y_train_dt.min():.1f} – {y_train_dt.max():.1f} hours")
print(f"   Test  target range : {y_test_dt.min():.1f}  – {y_test_dt.max():.1f} hours")

print("\n🏋️  Training XGBoost Regressor...")

regressor = XGBRegressor(
    n_estimators  = 100,
    max_depth     = 4,
    learning_rate = 0.08,
    reg_alpha     = 0.1,
    random_state  = 42,
    verbosity     = 0
)

regressor.fit(X_train, y_train_dt)
y_pred_reg = regressor.predict(X_test)

# Metrics
mae  = mean_absolute_error(y_test_dt, y_pred_reg)
rmse = np.sqrt(mean_squared_error(y_test_dt, y_pred_reg))
r2   = r2_score(y_test_dt, y_pred_reg)

print(f"\n   ✅ Regressor trained")
print(f"\n   {'Metric':<30} {'Value':>10}")
print("   " + "─" * 42)
print(f"   {'MAE  (avg error in hours)':<30} {mae:>10.2f}")
print(f"   {'RMSE (penalizes large errors)':<30} {rmse:>10.2f}")
print(f"   {'R² Score (1.0 = perfect)':<30} {r2:>10.4f}")

# Sample predictions
print(f"\n   Sample Predictions vs Actual:")
print(f"   {'Record':<10} {'Predicted (hrs)':>16} {'Actual (hrs)':>14}")
print("   " + "─" * 42)
for i in range(min(8, len(y_pred_reg))):
    print(f"   Record {i+1:<4}  {y_pred_reg[i]:>12.1f} hrs   "
          f"{y_test_dt.iloc[i]:>10.1f} hrs")

# Save regressor
with open('regressor.pkl', 'wb') as f:
    pickle.dump(regressor, f)

print(f"\n   ✅ regressor.pkl saved")
print(f"\n   Interpretation:")
print(f"   → MAE of {mae:.1f} means predictions are off by ~{mae:.1f} hours on average")
print(f"   → R² of {r2:.4f} means model explains "
      f"{r2*100:.1f}% of downtime variation")

print("\n" + "=" * 60)
print("  BOTH MODELS SAVED AND READY FOR FLASK")
print("=" * 60)
print(f"""
  Classifier  → model.pkl      (High Risk / Low Risk)
  Regressor   → regressor.pkl  (Expected downtime hours)

  
""")
print("=" * 60)