# ============================================================
#   Supply Chain Disruption Alerts — EDA (Exploratory Data Analysis)
#   Dataset : Supplier_Disruption_LogTable_Coded.csv
# ============================================================

import pandas as pd
import numpy as np
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

ACCENT  = '#00e5ff'
WARN    = '#ffd166'
DANGER  = '#ff4757'
SAFE    = '#2ed573'
PURPLE  = '#a855f7'

# ============================================================
# STEP 1 — Load the Dataset
# ============================================================
print("=" * 55)
print("  SUPPLY CHAIN DISRUPTION ALERTS — EDA REPORT")
print("=" * 55)

df = pd.read_csv('Supplier_Disruption_LogTable_Coded.csv')

print(f"\n✅ Dataset loaded successfully")
print(f"   Rows    : {df.shape[0]}")
print(f"   Columns : {df.shape[1]}")

# ============================================================
# STEP 2 — Basic Dataset Info
# ============================================================
print("\n" + "─" * 55)
print("  STEP 2 — DATASET STRUCTURE")
print("─" * 55)

print("\n📋 Column Names & Data Types:")
print(df.dtypes.to_string())

print("\n📊 First 5 Rows:")
print(df.head().to_string())

# ============================================================
# STEP 3 — Missing Values Check
# ============================================================
print("\n" + "─" * 55)
print("  STEP 3 — MISSING VALUES CHECK")
print("─" * 55)

missing = df.isnull().sum()
print("\n🔍 Missing Values Per Column:")
print(missing.to_string())

if missing.sum() == 0:
    print("\n✅ No missing values found — dataset is clean!")
else:
    print(f"\n⚠️  Total missing values: {missing.sum()}")

# ============================================================
# STEP 4 — Create Binary Label Column
# ============================================================
print("\n" + "─" * 55)
print("  STEP 4 — CREATE DISRUPTION LABEL")
print("─" * 55)

# Convert high_disruption_p (probability) to binary label
# 0 = Low Risk (probability < 0.5)
# 1 = High Risk (probability >= 0.5)
df['disruption'] = (df['high_disruption_p'] >= 0.5).astype(int)

count_0 = (df['disruption'] == 0).sum()
count_1 = (df['disruption'] == 1).sum()

print(f"\n📌 Label Column Created: 'disruption'")
print(f"   Threshold used      : high_disruption_p >= 0.5")
print(f"   Low Risk  (0)       : {count_0} records ({count_0/len(df)*100:.1f}%)")
print(f"   High Risk (1)       : {count_1} records ({count_1/len(df)*100:.1f}%)")

# ============================================================
# STEP 5 — Statistical Summary
# ============================================================
print("\n" + "─" * 55)
print("  STEP 5 — STATISTICAL SUMMARY")
print("─" * 55)

numeric_cols = ['kg_score', 'Em_Score', 'downtime_hours', 'high_disruption_p']
print("\n📈 Key Statistics:")
print(df[numeric_cols].describe().round(3).to_string())

# ============================================================
# STEP 6 — Supplier Summary
# ============================================================
print("\n" + "─" * 55)
print("  STEP 6 — SUPPLIER BREAKDOWN")
print("─" * 55)

supplier_summary = df.groupby('supplier').agg(
    Total_Records   = ('disruption', 'count'),
    High_Risk_Count = ('disruption', 'sum'),
    Avg_Downtime    = ('downtime_hours', 'mean'),
    Avg_KG_Score    = ('kg_score', 'mean'),
    Avg_Disruption_P= ('high_disruption_p', 'mean')
).round(3)

supplier_summary['Risk_%'] = (
    supplier_summary['High_Risk_Count'] / supplier_summary['Total_Records'] * 100
).round(1)

print("\n🏭 Per-Supplier Risk Summary:")
print(supplier_summary.to_string())

# ============================================================
# STEP 7 — Risk Factor Correlation
# ============================================================
print("\n" + "─" * 55)
print("  STEP 7 — RISK FACTOR CORRELATION")
print("─" * 55)

feature_cols = [
    'kg_score', 'Em_Score', 'twitter_alert', 'weather_risk',
    'economic_risk', 'geo_risk', 'tech_risk', 'downtime_hours'
]

corr_with_label = df[feature_cols + ['high_disruption_p']].corr()['high_disruption_p'].drop('high_disruption_p').sort_values(ascending=False)

print("\n🔗 Correlation of Each Feature With Disruption Probability:")
for col, val in corr_with_label.items():
    bar = '█' * int(abs(val) * 20)
    direction = '+' if val > 0 else '-'
    print(f"   {col:<18} {direction}{bar:<20} {val:.4f}")

# ============================================================
# STEP 8 — VISUALIZATIONS
# ============================================================
print("\n" + "─" * 55)
print("  STEP 8 — GENERATING CHARTS")
print("─" * 55)

# ── Chart 1: Disruption Label Distribution ──────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Supply Chain Disruption Alerts — EDA Overview',
             fontsize=14, fontweight='bold', color=ACCENT, y=1.01)
fig.patch.set_facecolor('#0f1117')

# Plot 1 — Label Distribution (pie)
ax1 = axes[0, 0]
sizes  = [count_0, count_1]
colors = [SAFE, DANGER]
labels = [f'Low Risk\n{count_0} records', f'High Risk\n{count_1} records']
wedges, texts, autotexts = ax1.pie(
    sizes, labels=labels, colors=colors, autopct='%1.1f%%',
    startangle=90, textprops={'fontsize': 9}
)
for at in autotexts:
    at.set_color('white')
    at.set_fontweight('bold')
ax1.set_title('Disruption Label Distribution', color=ACCENT, fontsize=10, pad=12)

# Plot 2 — Downtime Hours Distribution
ax2 = axes[0, 1]
ax2.hist(df[df['disruption']==0]['downtime_hours'], bins=20,
         color=SAFE, alpha=0.7, label='Low Risk', edgecolor='#0f1117')
ax2.hist(df[df['disruption']==1]['downtime_hours'], bins=20,
         color=DANGER, alpha=0.7, label='High Risk', edgecolor='#0f1117')
ax2.set_title('Downtime Hours by Risk Level', color=ACCENT, fontsize=10)
ax2.set_xlabel('Downtime Hours')
ax2.set_ylabel('Count')
ax2.legend(fontsize=8)
ax2.grid(True)

# Plot 3 — KG Score Distribution
ax3 = axes[0, 2]
ax3.hist(df[df['disruption']==0]['kg_score'], bins=20,
         color=SAFE, alpha=0.7, label='Low Risk', edgecolor='#0f1117')
ax3.hist(df[df['disruption']==1]['kg_score'], bins=20,
         color=DANGER, alpha=0.7, label='High Risk', edgecolor='#0f1117')
ax3.set_title('KG Score by Risk Level', color=ACCENT, fontsize=10)
ax3.set_xlabel('KG Score')
ax3.set_ylabel('Count')
ax3.legend(fontsize=8)
ax3.grid(True)

# Plot 4 — Correlation Heatmap
ax4 = axes[1, 0]
corr_matrix = df[feature_cols + ['high_disruption_p']].corr()
mask = np.zeros_like(corr_matrix, dtype=bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(
    corr_matrix, ax=ax4, mask=mask,
    cmap='coolwarm', annot=True, fmt='.2f',
    annot_kws={'size': 7}, linewidths=0.5,
    cbar_kws={'shrink': 0.8}
)
ax4.set_title('Feature Correlation Heatmap', color=ACCENT, fontsize=10)
ax4.tick_params(labelsize=7)

# Plot 5 — Risk Flags Count (binary features)
ax5 = axes[1, 1]
binary_cols = ['twitter_alert', 'weather_risk', 'economic_risk', 'geo_risk', 'tech_risk']
risk_counts  = df[binary_cols].sum()
bars = ax5.barh(binary_cols, risk_counts, color=[ACCENT, WARN, DANGER, PURPLE, SAFE])
ax5.set_title('Risk Flag Occurrences', color=ACCENT, fontsize=10)
ax5.set_xlabel('Count')
ax5.grid(True, axis='x')
for bar, val in zip(bars, risk_counts):
    ax5.text(val + 1, bar.get_y() + bar.get_height()/2,
             str(int(val)), va='center', fontsize=8, color='white')

# Plot 6 — Average Disruption Probability by Supplier
ax6 = axes[1, 2]
sup_avg = df.groupby('supplier')['high_disruption_p'].mean().sort_values(ascending=True)
colors_bar = [DANGER if v >= 0.7 else WARN if v >= 0.5 else SAFE for v in sup_avg]
ax6.barh(sup_avg.index, sup_avg.values, color=colors_bar)
ax6.axvline(0.5, color='white', linestyle='--', linewidth=1, alpha=0.5, label='Threshold 0.5')
ax6.set_title('Avg Disruption Prob by Supplier', color=ACCENT, fontsize=10)
ax6.set_xlabel('Avg Disruption Probability')
ax6.grid(True, axis='x')
ax6.legend(fontsize=7)

plt.tight_layout()
plt.savefig('eda_overview.png', dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
plt.show()
print("✅ Chart saved: eda_overview.png")

# ── Chart 2: Disruption Probability Over Time ───────────────
fig2, ax = plt.subplots(figsize=(14, 5))
fig2.patch.set_facecolor('#0f1117')

df['date'] = pd.to_datetime(df['date'])
df_sorted  = df.sort_values('date')

for supplier, grp in df_sorted.groupby('supplier'):
    ax.plot(grp['date'], grp['high_disruption_p'],
            alpha=0.6, linewidth=1.2, label=supplier)

ax.axhline(0.5, color='white', linestyle='--', linewidth=1,
           alpha=0.6, label='Risk Threshold (0.5)')
ax.fill_between(df_sorted['date'].unique(),
                0.5, 1.0, alpha=0.05, color=DANGER)
ax.set_title('Disruption Probability Over Time — All Suppliers',
             color=ACCENT, fontsize=12, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Disruption Probability')
ax.legend(fontsize=8, loc='upper left')
ax.grid(True)

plt.tight_layout()
plt.savefig('eda_timeseries.png', dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
plt.show()
print("✅ Chart saved: eda_timeseries.png")

# ============================================================
# STEP 9 — Final Summary
# ============================================================
print("\n" + "=" * 55)
print("  EDA COMPLETE — SUMMARY")
print("=" * 55)
print(f"\n  Dataset    : Supplier_Disruption_LogTable_Coded.csv")
print(f"  Total Rows : {df.shape[0]}")
print(f"  Features   : {len(feature_cols)}")
print(f"  Label      : disruption (0 = Low Risk, 1 = High Risk)")
print(f"  High Risk  : {count_1} records ({count_1/len(df)*100:.1f}%)")
print(f"  Low Risk   : {count_0} records ({count_0/len(df)*100:.1f}%)")
print(f"  Missing    : None")
print(f"\n  Charts saved:")
print(f"  → eda_overview.png")
print(f"  → eda_timeseries.png")
print(f"\n  ✅ Ready for model training (Phase 3)")
print("=" * 55)