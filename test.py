import pandas as pd
df = pd.read_csv('Supplier_Disruption_LogTable_Coded.csv')

a= df['high_disruption_p'].value_counts()

print(df.head(10))# Create a temporary series where True (1) means >= 0.45 and False (0) means < 0.45
threshold_counts = (df['high_disruption_p'] >= 0.45).value_counts()

print("--- Counts with 0.45 Threshold ---")
print(threshold_counts)