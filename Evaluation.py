# ============================================================
#   Supply Chain Disruption Alerts — Evaluation & SHAP
#   Dataset : Supplier_Disruption_LogTable_Coded.csv
#   Phase 5 : Evaluate Best Model + Explain Predictions
# ============================================================

import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier
from sklearn.tree          import DecisionTreeClassifier
from xgboost               import XGBClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
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
print("  SUPPLY CHAIN DISRUPTION — EVALUATION & SHAP")
print("=" * 60)

# ============================================================
# STEP 1 — Load Model and Test Data
# ============================================================
print("\n📂 STEP 1 — Loading Model and Test Data...")

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model_info.pkl', 'rb') as f:
    model_info = pickle.load(f)

model_name = model_info['name']

X_test  = pd.read_csv('X_test.csv')
y_test  = pd.read_csv('y_test.csv').squeeze()
X_train = pd.read_csv('X_train.csv')

print(f"   ✅ Model loaded     : {model_name}")
print(f"   ✅ Test set loaded  : {X_test.shape[0]} rows")
print(f"   ✅ Train set loaded : {X_train.shape[0]} rows")

feature_names = [
    'kg_score', 'Em_Score', 'twitter_alert',
    'weather_risk', 'economic_risk', 'geo_risk',
    'tech_risk'
]

feature_labels = {
    'kg_score'      : 'KG Score',
    'Em_Score'      : 'Emergency Score',
    'twitter_alert' : 'Twitter Alert',
    'weather_risk'  : 'Weather Risk',
    'economic_risk' : 'Economic Risk',
    'geo_risk'      : 'Geo Risk',
    'tech_risk'     : 'Tech Risk',
    'downtime_hours': 'Downtime Hours'
}

# ============================================================
# STEP 2 — Generate Predictions
# ============================================================
print("\n🔮 STEP 2 — Generating Predictions...")

y_pred      = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]
risk_scores = (y_pred_prob * 100).round(1)

print(f"   ✅ Predictions generated for {len(y_pred)} test records")

# ============================================================
# STEP 3 — Full Metric Evaluation
# ============================================================
print("\n" + "─" * 60)
print("  STEP 3 — PERFORMANCE METRICS")
print("─" * 60)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)
cm   = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc     = auc(fpr, tpr)

precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_prob)
pr_auc = auc(recall_vals, precision_vals)

print(f"""
  Model           : {model_name}
  ──────────────────────────────────────
  Accuracy        : {acc*100:.2f}%
  Precision       : {prec*100:.2f}%
  Recall          : {rec*100:.2f}%
  F1 Score        : {f1:.4f}
  ROC-AUC Score   : {roc_auc:.4f}
  PR-AUC Score    : {pr_auc:.4f}
  ──────────────────────────────────────
  True Positives  (TP) : {tp}   ✅ Correctly flagged disruptions
  True Negatives  (TN) : {tn}   ✅ Correctly cleared suppliers
  False Positives (FP) : {fp}   ⚠️  False alarms raised
  False Negatives (FN) : {fn}   ❌ Disruptions missed
""")

# ============================================================
# STEP 4 — Classification Report
# ============================================================
print("─" * 60)
print("  STEP 4 — CLASSIFICATION REPORT")
print("─" * 60)

print(f"\n{classification_report(y_test, y_pred, target_names=['Low Risk (0)', 'High Risk (1)'])}")

# ============================================================
# STEP 5 — Risk Score Distribution
# ============================================================
print("─" * 60)
print("  STEP 5 — RISK SCORE DISTRIBUTION")
print("─" * 60)

bands = {
    '0–20   Very Low' : 0,
    '21–40  Low'      : 0,
    '41–60  Medium'   : 0,
    '61–80  High'     : 0,
    '81–100 Critical' : 0
}

for s in risk_scores:
    if   s <= 20: bands['0–20   Very Low']  += 1
    elif s <= 40: bands['21–40  Low']        += 1
    elif s <= 60: bands['41–60  Medium']     += 1
    elif s <= 80: bands['61–80  High']       += 1
    else:         bands['81–100 Critical']   += 1

print(f"\n  {'Risk Band':<22} {'Count':>6}  {'Share':>7}")
print("  " + "─" * 42)
for band, count in bands.items():
    bar   = '█' * int(count / len(risk_scores) * 28)
    share = count / len(risk_scores) * 100
    print(f"  {band:<22} {count:>6}   {share:>5.1f}%  {bar}")

# ============================================================
# STEP 6 — SHAP Explainability (Auto-detects model type)
# ============================================================

print("\n🔍 STEP 6 — SHAP EXPLAINABILITY")

import shap
import numpy as np
import pandas as pd

# 1. Initialize Explainer on the calibrated model's predict method
# Using the .predict function ensures SHAP evaluates the final probability output
explainer = shap.KernelExplainer(model.predict, X_test)
shap_values = explainer.shap_values(X_test)

# 2. FIX: Handle calibrated classifier dimensionality output
# If shap_values is a list or 3D array, extract or average it
if isinstance(shap_values, list):
    # For binary metrics, take the SHAP values for Class 1
    mean_shap = np.abs(shap_values[1]).mean(axis=0)
elif len(shap_values.shape) == 3:
    # If shape is (samples, features, classes), extract Class 1
    mean_shap = np.abs(shap_values[:, :, 1]).mean(axis=0)
else:
    # Standard 2D fallback
    mean_shap = np.abs(shap_values).mean(axis=0)

# Ensure feature names match X_test columns
feature_names = X_test.columns.tolist()

# 3. Create the Summary DataFrame safely
shap_df = pd.DataFrame({
    'Feature': feature_names,
    'Absolute_Importance': mean_shap
})

# Sort by highest structural impact
shap_df = shap_df.sort_values(by='Absolute_Importance', ascending=False)

print("\n📊 Feature Global Importance (SHAP):")
print(shap_df.to_string(index=False))

# ============================================================
# STEP 7 — Evaluation Charts
# ============================================================
print("\n📊 STEP 7 — Generating Evaluation Charts...")

fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor('#0f1117')
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

fig.suptitle(
    f'Supply Chain Disruption Alerts — Evaluation Report\nModel: {model_name}',
    fontsize=13, fontweight='bold', color=ACCENT, y=1.01
)

# Chart 1 — Confusion Matrix
ax1 = fig.add_subplot(gs[0, 0])
cm_annot  = np.array([[f'TN\n{tn}', f'FP\n{fp}'],
                       [f'FN\n{fn}', f'TP\n{tp}']])
cm_values = np.array([[tn, fp], [fn, tp]])
sns.heatmap(
    cm_values, annot=cm_annot, fmt='', ax=ax1,
    cmap='Blues', linewidths=2, linecolor='#0f1117',
    xticklabels=['Predicted\nLow Risk', 'Predicted\nHigh Risk'],
    yticklabels=['Actual\nLow Risk',    'Actual\nHigh Risk'],
    annot_kws={'size': 12, 'weight': 'bold'}, cbar=False
)
ax1.set_title('Confusion Matrix', color=ACCENT, fontsize=11, pad=12)

# Chart 2 — Metrics Bar
ax2 = fig.add_subplot(gs[0, 1])
m_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
m_vals  = [acc, prec, rec, f1]
m_colors= [ACCENT, SAFE, DANGER, WARN]
bars2 = ax2.bar(m_names, [v * 100 for v in m_vals],
                color=m_colors, alpha=0.85, edgecolor='#0f1117')
ax2.set_ylim(0, 118)
ax2.set_title('Performance Metrics', color=ACCENT, fontsize=11, pad=12)
ax2.set_ylabel('Score (%)')
ax2.axhline(80, color='white', linestyle='--',
            linewidth=1, alpha=0.4, label='Target 80%')
ax2.legend(fontsize=8)
ax2.grid(True, axis='y')
for bar, val in zip(bars2, m_vals):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1.5,
        f'{val*100:.1f}%',
        ha='center', va='bottom',
        fontsize=9, color='white', fontweight='bold'
    )

# Chart 3 — ROC Curve
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(fpr, tpr, color=ACCENT, linewidth=2,
         label=f'AUC = {roc_auc:.4f}')
ax3.plot([0, 1], [0, 1], color=WARN, linestyle='--',
         linewidth=1, alpha=0.5, label='Random')
ax3.fill_between(fpr, tpr, alpha=0.08, color=ACCENT)
ax3.set_title('ROC Curve', color=ACCENT, fontsize=11, pad=12)
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate')
ax3.legend(fontsize=9)
ax3.grid(True)

# Chart 4 — SHAP Feature Importance
ax4 = fig.add_subplot(gs[1, 0:2])
shap_colors = [ACCENT if i == 0 else SAFE if i == 1 else '#3a4060'
               for i in range(len(shap_df))]
bars4 = ax4.barh(
    shap_df['Feature'][::-1],
    shap_df['Absolute_Importance'][::-1],
    color=shap_colors[::-1],
    edgecolor='#0f1117', alpha=0.9
)
ax4.set_title(
    'SHAP Feature Importance\n(Which factors drive disruption predictions most)',
    color=ACCENT, fontsize=11, pad=12
)
ax4.set_xlabel('Mean |SHAP Value|')
ax4.grid(True, axis='x')
for bar, val in zip(bars4, shap_df['Absolute_Importance'][::-1]):
    ax4.text(
        val + shap_df['Absolute_Importance'].max() * 0.01,
        bar.get_y() + bar.get_height() / 2,
        f'{val:.4f}', va='center', fontsize=8, color='white'
    )

# Chart 5 — Risk Score Histogram
ax5 = fig.add_subplot(gs[1, 2])
ax5.hist(risk_scores, bins=20, color=ACCENT,
         edgecolor='#0f1117', alpha=0.85)
# Look for ax5.axvline lines around your distribution chart and update them:
ax5.axvline(25, color=WARN,   linestyle='--', linewidth=1.5, alpha=0.8, label='Medium (25)')
ax5.axvline(45, color=DANGER, linestyle='--', linewidth=1.5, alpha=0.8, label='High (45)')
ax5.set_title('Risk Score Distribution', color=ACCENT, fontsize=11, pad=12)
ax5.set_xlabel('Risk Score (0–100)')
ax5.set_ylabel('Number of Records')
ax5.legend(fontsize=8)
ax5.grid(True, axis='y')

plt.savefig('evaluation_report.png', dpi=150,
            bbox_inches='tight', facecolor='#0f1117')
plt.show()
print("   ✅ Saved: evaluation_report.png")

# ============================================================
# STEP 8 — SHAP Summary Bar Plot
# ============================================================
print("\n📊 STEP 8 — Generating SHAP Summary Plot...")

X_test_labeled         = X_test.copy()
X_test_labeled.columns = [feature_labels[f] for f in feature_names]

fig2, ax_shap = plt.subplots(figsize=(10, 6))
fig2.patch.set_facecolor('#0f1117')

shap.summary_plot(
    shap_values,
    X_test_labeled,
    plot_type='bar',
    show=False,
    max_display=8
)

plt.title(
    'SHAP — Feature Impact on Disruption Prediction',
    color=ACCENT, fontsize=12, fontweight='bold', pad=15
)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=150,
            bbox_inches='tight', facecolor='#0f1117')
plt.show()
print("   ✅ Saved: shap_summary.png")

# ============================================================
# STEP 9 — Sample Predictions Table
# ============================================================
print("\n" + "─" * 60)
print("  STEP 9 — SAMPLE PREDICTIONS WITH RISK SCORES")
print("─" * 60)

print(f"\n  {'#':<5} {'Score':>9} {'Level':<16} {'Predicted':<12} {'Actual':<12} {'Result'}")
print("  " + "─" * 65)

for i in range(min(15, len(risk_scores))):
    score     = risk_scores[i]
    predicted = 'High Risk' if y_pred[i]      == 1 else 'Low Risk'
    actual    = 'High Risk' if y_test.iloc[i] == 1 else 'Low Risk'
    match     = '✅ Correct' if y_pred[i] == y_test.iloc[i] else '❌ Wrong'

    if   score >= 70: level = '🔴 CRITICAL'
    elif score >= 45: level = '🟡 MEDIUM'
    else:             level = '🟢 LOW'

    print(f"  {i+1:<5} {score:>7.1f}/100  {level:<16} {predicted:<12} {actual:<12} {match}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("  EVALUATION COMPLETE — FINAL SUMMARY")
print("=" * 60)
print(f"""
  Model           : {model_name}
  ──────────────────────────────────────
  Accuracy        : {acc*100:.2f}%
  Precision       : {prec*100:.2f}%
  Recall          : {rec*100:.2f}%
  F1 Score        : {f1:.4f}
  ROC-AUC         : {roc_auc:.4f}

  Top Risk Factors (from SHAP):
  #1  {shap_df.iloc[0]['Feature']}
  #2  {shap_df.iloc[1]['Feature']}
  #3  {shap_df.iloc[2]['Feature']}
  ──────────────────────────────────────
  Charts Saved:
  → evaluation_report.png
  → shap_summary.png

  
""")
print("=" * 60)