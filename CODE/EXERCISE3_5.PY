# Exercise 3.5 Naive Bayes Model Save and Load
import pickle
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB

# 1. Load Data Iris
X, y = load_iris(return_X_y=True)

# 2. Melatih (Train) Model Naive Bayes
clf = GaussianNB()
clf.fit(X, y)

# 3. MENYIMPAN (SAVE) MODEL KE FILE
filename = 'naive_bayes_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(clf, file)
print("Model berhasil disimpan ke file:", filename)

# 4. MEMUAT (LOAD) MODEL DARI FILE
with open(filename, 'rb') as file:
    loaded_model = pickle.load(file)
print("Model berhasil dimuat dari file.")

# 5. MELAKUKAN PREDIKSI MENGGUNAKAN MODEL YANG DILOAD
p = loaded_model.predict([[5.0, 3.4, 1.5, 0.4]])
print("Hasil Prediksi:", p)