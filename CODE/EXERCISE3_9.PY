# Exercise 3.9 Random Forest on Diabetes Dataset
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load Diabetes Dataset
X, y = load_diabetes(return_X_y=True)

# 2. Split Data (Training 50%, Testing 50%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# 3. Inisialisasi & Fit Model Random Forest Regressor
clf = RandomForestRegressor(random_state=0)
clf.fit(X_train, y_train)

# 4. Prediksi
y_pred = clf.predict(X_test)

# 5. Evaluasi Hasil Regresi
print("Total test points:", y_test.shape[0])
print("Mean Squared Error (MSE): %.2f" % mean_squared_error(y_test, y_pred))
print("R2 Score: %.2f" % r2_score(y_test, y_pred))