# Exercise 3.10 
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Ridge

# List Nama Model
names = [
    "SVR (SVM)", 
    "Linear Regression", 
    "Ridge Regression",
    "Decision Tree", 
    "Random Forest", 
    "Nearest Neighbors", 
    "Neural Networks"
]

# List Algoritma Regresi
regressors = [
    SVR(),
    LinearRegression(),
    Ridge(),
    DecisionTreeRegressor(),
    RandomForestRegressor(random_state=0),
    KNeighborsRegressor(),
    MLPRegressor(max_iter=3000, random_state=0)
]

# Load Diabetes Dataset
X, y = load_diabetes(return_X_y=True)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# Fit dan Evaluasi Skor 
for name, reg in zip(names, regressors):
    reg.fit(X_train, y_train)
    score = reg.score(X_test, y_test)
    print(name + ": " + str(score))