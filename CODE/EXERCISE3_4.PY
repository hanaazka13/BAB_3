# Exercise 3.4 Python SVM Breast Cancer Classifications with Histograms
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib import pyplot

# Load dataset breast cancer
cancer = load_breast_cancer()

# Ubah data ke dalam DataFrame Pandas
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)

features_to_plot = ['mean radius', 'mean texture', 'mean area', 'mean smoothness']
df[features_to_plot].hist(figsize=(10, 8))
pyplot.tight_layout()
pyplot.show() # Tutup jendela grafik agar program berlanjut ke klasifikasi

# Pembagian data training & testing
X = cancer.data  # Semua fitur
y = cancer.target  # Semua label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)


clf = SVC()
clf.fit(X_train, y_train)

# 6. Prediksi
y_predict = clf.predict(X_test)

# Print Confusion Matrix & Classification Report
cm = np.array(confusion_matrix(y_test, y_predict, labels=[1, 0]))
confusion = pd.DataFrame(cm, index=['is_cancer', 'is_healthy'],
                         columns=['predicted_cancer', 'predicted_healthy'])

print(confusion)
print("\nClassification Report:\n")
print(classification_report(y_test, y_predict))