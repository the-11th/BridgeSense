import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('NBI_data_filtered_1.csv', low_memory=False)

df_sub = df

numeric_cols = df_sub.select_dtypes(include='number').columns.tolist()

numeric_low = [
    '27 - Year Built', '29 - Average Daily Traffic',
    '45 - Number of Spans in Main Unit', 'Bridge Age (yr)',
    '34 - Skew Angle (degrees)', '30 - Year of Average Daily Traffic',
    '115 - Year of Future Average Daily Traffic',
    'Average Relative Humidity', 'Average Temperature',
    'Maximum Temperature', 'Minimum Temperature', 'Mean Wind Speed'
]
numeric_med = [
    '64 - Operating Rating (US tons)', '66 - Inventory Rating (US tons)',
    '109 - Average Daily Truck Traffic (Percent ADT)',
    '96 - Total Project Cost', 'Computed - Average Daily Truck Traffic (Volume)'
]
recon_col = '106 - Year Reconstructed'
categorical_cols = [
    '43A - Main Span Material', '43B - Main Span Design',
    '91 - Designated Inspection Frequency'
]

df_sub = df_sub.copy()  # avoid chained-assignment warnings
df_sub['Was_Reconstructed'] = df_sub[recon_col].notna().astype(int)
df_sub[recon_col] = df_sub[recon_col].fillna(0)


# impute with mean/median/mode
for col in numeric_low:
    df_sub[col] = df_sub[col].fillna(df_sub[col].median())

for col in numeric_med:
    flag_col = f"{col}_missing_flag"
    df_sub[flag_col] = df_sub[col].isna().astype(int)
    df_sub[col] = df_sub[col].fillna(df_sub[col].median())

for col in categorical_cols:
    df_sub[col] = df_sub[col].fillna(df_sub[col].mode()[0])

print("Missing counts after full imputation:")
print(df_sub.isna().sum())

df_sub.to_csv('NBI_data_imputed.csv', index=False)

'''
flier_props = dict(marker='o', markerfacecolor='gray', markersize=4, linestyle='none', alpha=0.5)

for col in numeric_cols:
    plt.figure(figsize=(5, 6))
    plt.boxplot(df_sub[col].dropna(), whis=3.0, flierprops=flier_props, boxprops={'linewidth':1.2}, medianprops={'linewidth':1.5})
    plt.title(f'Box Plot of {col}\n(Whiskers = 3×IQR)')
    plt.ylabel(col)
    plt.xticks([1], [''])
    plt.tight_layout()
    #plt.show()
'''
missing_pct = df_sub.isna().sum()
missing_pct_df = missing_pct.reset_index()
missing_pct_df.columns = ['Variable', 'Percent Missing']
print(missing_pct_df)

# missing
plt.figure(figsize=(10, 6))
missing_pct.sort_values(ascending=False).plot(kind='bar')
plt.title('Percentage of Missing Values')
plt.ylabel('Percent Missing')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

categorical_cols = [
    '43A - Main Span Material',
    '43B - Main Span Design',
    '91 - Designated Inspection Frequency',
    'CAT10 - Bridge Condition'
]

for col in categorical_cols:
    counts = df_sub[col].value_counts(dropna=False).sort_index()
    print(f"Value counts for {col}:\n{counts}\n")
    plt.figure(figsize=(8, 4))
    counts.plot(kind='bar')
    plt.title(f'Category Counts: {col}')
    plt.ylabel('Number of Entries')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

numerical_cols = df_sub.select_dtypes(include='number').columns.tolist()

mean_values = df_sub[numerical_cols].mean()
median_values = df_sub[numerical_cols].median()

plt.figure(figsize=(10, 6))
mean_values.sort_values(ascending=False).plot(kind='bar')
plt.title('Mean of Numerical Variables')
plt.ylabel('Mean')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
median_values.sort_values(ascending=False).plot(kind='bar')
plt.title('Median of Numerical Variables')
plt.ylabel('Median')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

before_count = len(df_sub)
missing_mask = df_sub['CAT10 - Bridge Condition'].isna()
removed = df_sub[missing_mask]

print(f"Removed {removed.shape[0]} that were originally missing.")
print("Indices of removed entries:")
print(removed.index.tolist())

df_sub = df_sub[~missing_mask].reset_index(drop=True)

#after_count = len(df_sub)
#print(f"Data reduced from {before_count} to {after_count} rows.")
#df_sub.to_csv('NBI_data_filtered_1.csv', index=False)

#files.download('NBI_data_imputed.csv')
print("Columns in df_sub:")
print(df_sub.columns.tolist())
