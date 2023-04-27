import pandas as pd
import seaborn as sns

df = pd.read_excel('RawDataset.xlsx', header=None, names=['Country', 'GDP', 'LEABY'])

# Reorder the columns
df = df[['Country', 'GDP', 'LEABY']]
print(df.head())

# Drop the 'Country' column
df_numeric = df.drop(columns=['Country'])

# Calculate the correlation matrix
corr = df_numeric.corr()

# Visualize the correlation matrix as a heatmap
sns.heatmap(corr, annot=True)

# Calculate the Pearson correlation coefficient
pearson_coef = df_numeric['GDP'].corr(df_numeric['LEABY'], method='pearson')

# Print the Pearson correlation coefficient
print('Pearson correlation coefficient:', pearson_coef)

