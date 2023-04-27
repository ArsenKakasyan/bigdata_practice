import pandas as pd
import seaborn as sns

df = pd.read_excel('RawDataset.xlsx', header=None, names=['Country', 'GDP', 'LEABY'])

# Изменить порядок столбцов
df = df[['Country', 'GDP', 'LEABY']]
print(df.head())

# Удалите столбец "Страна"
df_numeric = df.drop(columns=['Country'])

# Рассчитать корреляционную матрицу
corr = df_numeric.corr()

# Визуализируйте матрицу корреляции как тепловую карту
sns.heatmap(corr, annot=True)

# Рассчитать коэффициент корреляции Пирсона
pearson_coef = df_numeric['GDP'].corr(df_numeric['LEABY'], method='pearson')

# Вывести коэфициент корреляции Пирсона
print('Pearson correlation coefficient:', pearson_coef)

