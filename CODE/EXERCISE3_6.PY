# Exercise 3.6 LinearDiscriminantAnalysis Classification
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Modifikasi: n_samples=2000 dan n_features=6
X, y = make_classification(n_samples=2000, n_features=6,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)

print(X)

clf = LinearDiscriminantAnalysis()
clf.fit(X, y)

# Prediksi menggunakan 6 nilai fitur (menyesuaikan n_features=6)
p = clf.predict([[0, 0, 0, 0, 0, 0]])
