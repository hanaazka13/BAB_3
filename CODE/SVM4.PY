#Example 3.4 Python SVM Iris URL Classifications
import pandas as pd
from sklearn import datasets, svm
from matplotlib import pyplot
from pandas.plotting import scatter_matrix

# 1. Mengambil dataset Iris bawaan Python (Tanpa butuh link URL / file CSV)
iris_raw = datasets.load_iris()

# 2. Mengubah ke DataFrame Pandas (Menirukan struktur file iris.csv)
df = pd.DataFrame(iris_raw.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
df['species'] = [iris_raw.target_names[i] for i in iris_raw.target]

# 3. Menampilkan informasi data (Sesuai modul Example 3.4)
print(df.shape)
print(df.head(10))
print(df.tail(10))
print(df.describe())

# 4. Hitung dan bersihkan missing values (NAN)
print(df.isna().sum().sum())
df = df.dropna()
print(df.groupby('species').size())

# 5. Visualisasi Plot (Tutup jendela gambar setelah muncul agar program lanjut)
df.hist()
pyplot.show()

scatter_matrix(df)
pyplot.show()

# 6. Menyiapkan Fitur (X) dan Target (y)
X = df.values[:, :2]  # Sepal length & Sepal width
s = df['species']
d = dict([(y, x) for x, y in enumerate(sorted(set(s)))])
y = [d[x] for x in s]

# 7. Pelatihan Model & Prediksi SVM
clf = svm.SVC()
clf.fit(X, y)

# Predict 
p = clf.predict([[5.4, 3.2]])
print(p)