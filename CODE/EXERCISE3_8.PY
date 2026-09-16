# Exercise 3.8 Decision Tree Classification on Wine Dataset
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load Wine Dataset
X, y = load_wine(return_X_y=True)

# Split Data (Training 50%, Testing 50%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Prediksi Data Testing
y_pred = clf.predict(X_test)

# Hitung Akurasi Prediksi
N = y_test.shape[0]
C = (y_test == y_pred).sum()

print("Total points: %d Correctly labeled points : %d" % (N, C))