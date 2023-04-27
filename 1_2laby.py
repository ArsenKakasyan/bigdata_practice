import numpy as np 
import pandas as pd 
import matplotlib.pyplot as pyplot 
from matplotlib.pyplot import figure 
from sklearn import datasets 
from sklearn import linear_model 
from sklearn.cluster import KMeans 
from sklearn import model_selection 
from sklearn import metrics 
from sklearn.model_selection import train_test_split 
from sklearn import linear_model 
from pandas import DataFrame 
import seaborn as sns 
from scipy import polyval, stats 
 
import warnings 
warnings.filterwarnings("ignore") 
 
# Загружаем набор данных Ирисы: 
iris = datasets.load_iris() 
 
iris_frame = DataFrame(iris.data) 
# Делаем имена колонок такие же, как имена переменных: 
iris_frame.columns = iris.feature_names 
# Добавляем столбец с целевой переменной: 
iris_frame['target'] = iris.target 
# Для наглядности добавляем столбец с сортами: 
iris_frame['name'] = iris_frame.target.apply(lambda x : iris.target_names[x]) 
# Смотрим, что получилось: 
#print() 
#print(iris_frame) 
 
#Алгоритм k-средних находит центры кластеров путем итерационного пересчета матрицы расстояний между объектами и центрами кластеров. 
#Сначала случайным образом выбираются k центров. Затем каждый объект присваивается к ближайшему центру. 
#Далее пересчитываются координаты центров, как среднее значение координат всех объектов, присвоенных к ним. 
#Процесс повторяется до тех пор, пока координаты центров не перестанут изменяться. На этом этапе считается, что алгоритм сошелся и найдены центры кластеров.
train_data, test_data, train_labels, test_labels = train_test_split(iris_frame[['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']], iris_frame[['target']], test_size = 0.3) 
model = KMeans(n_clusters=3) 
model.fit(train_data) 
model_predictions = model.predict(test_data) 
print(metrics.accuracy_score(test_labels, model_predictions)) 
print(metrics.classification_report(test_labels, model_predictions)) 
 
#labels = model.labels_ 
#print(labels) # выводим метки кластеров для каждого объекта в dataset 
 
predictions = pd.DataFrame(model_predictions, columns=['cluster']) 
results = pd.concat([test_data, test_labels, predictions], axis=1) 
print(results) 
 
# Создадим новый столбец с предсказанными кластерами: 
iris_frame['cluster'] = model.predict(iris_frame[['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']]) 
 
# Визуализируем кластеры: 
sns.scatterplot(data=iris_frame, x="petal length (cm)", y="petal width (cm)", hue='name', style='cluster', palette='dark', s=100) 
# Визуализируем центры кластеров: 
centers = model.cluster_centers_ 
sns.scatterplot(x=centers[:, 2], y=centers[:, 3], color='black', s=300, marker='*') 
# Выведем график: 
pyplot.show() 


sns.scatterplot(data=iris_frame, x="sepal length (cm)", y="sepal width (cm)", hue='name', style='cluster', palette='dark', s=100) 
centers = model.cluster_centers_ 
sns.scatterplot(x=centers[:, 0], y=centers[:, 1], color='black', s=300, marker='*') 
pyplot.show() 

# Выполним SVD-разложение набора данных:
U, s, V = np.linalg.svd(iris_frame[['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']])

# Выведем график SVD-разложения:
sns.scatterplot(x=U[:, 0], y=U[:, 1], hue=iris_frame['name'], palette='dark', s=100)
pyplot.xlabel('U1')
pyplot.ylabel('U2')
pyplot.show()

# Данный код выполняет SVD-разложение, которое позволяет представить исходный набор данных в виде произведения трех матриц: U, s и V. 
# Затем, на графике выводится проекция данных на две первые главные компоненты U1 и U2 (столбцы матрицы U), полученные в результате SVD-разложения. 
# Каждый объект из набора данных на графике представлен точкой, цвет которой определяется категорией "name" (разных видов ириса). 
# Таким образом, данный график позволяет визуально оценить, как отличаются объекты в исходном наборе данных и выделить возможные закономерности разделения данных на категории.
