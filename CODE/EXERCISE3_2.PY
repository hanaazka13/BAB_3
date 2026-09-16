# Exercise 3.2 Python SVM Iris Classifications (Petal Length & Width)
from sklearn import svm, datasets

iris = datasets.load_iris()

# Mengambil fitur ke-3 dan ke-4: Petal length dan Petal width
X = iris.data[:, 2:4]
y = iris.target  # 0: Setosa, 1: Versicolour, 2: Virginica

clf = svm.SVC()
clf.fit(X, y)

# Prediksi 
p = clf.predict([[5.4, 3.2]])
print(p)