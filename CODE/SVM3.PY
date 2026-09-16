#Example 3.3 Python SVM Iris CSV Classifications
import pandas as pd
from sklearn import datasets, svm

# 1. Download dataset iris otomatis via scikit-learn
iris_raw = datasets.load_iris()

# 2. Buat DataFrame Pandas (menirukan isi file iris.csv)
df = pd.DataFrame(iris_raw.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
# Mengubah angka target menjadi nama spesies teks ('setosa', 'versicolor', 'virginica')
df['species'] = [iris_raw.target_names[i] for i in iris_raw.target]


# Ambil 2 fitur pertama (Sepal length & Sepal width)
X = df.values[:, :2]

# Proses Encoding Teks Spesies ke Angka
s = df['species']
d = dict([(y, x) for x, y in enumerate(sorted(set(s)))])
y = [d[x] for x in s]

clf = svm.SVC()
clf.fit(X, y)

# Prediksi
p = clf.predict([[5.4, 3.2]])
print(p)