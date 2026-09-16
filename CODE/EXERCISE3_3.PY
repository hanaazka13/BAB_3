# Exercise 3.3 - Scatter Plot Sepal Length vs Sepal Width
import pandas as pd
from sklearn import datasets, svm
from matplotlib import pyplot

# 1. Load dataset Iris
iris_raw = datasets.load_iris()
df = pd.DataFrame(iris_raw.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
df['species'] = [iris_raw.target_names[i] for i in iris_raw.target]

# 2. Bersihkan data jika ada NAN
df = df.dropna()

# 3. MODIFIKASI EXERCISE 3.3: Plot Scatter 2 Fitur Pertama (Sepal Length & Sepal Width)
pyplot.scatter(df['sepal_length'], df['sepal_width'], c='blue')
pyplot.xlabel('Sepal Length (cm)')
pyplot.ylabel('Sepal Width (cm)')
pyplot.title('Scatter Plot: Sepal Length vs Sepal Width')
pyplot.show()

# 4. Melatih Model SVM & Prediksi
X = df.values[:, :2]
s = df['species']
d = dict([(y, x) for x, y in enumerate(sorted(set(s)))])
y = [d[x] for x in s]

clf = svm.SVC()
clf.fit(X, y)

p = clf.predict([[5.4, 3.2]])
print(p)