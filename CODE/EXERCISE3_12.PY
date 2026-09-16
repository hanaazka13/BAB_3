# Exercise 3.12 
from sklearn import linear_model
from sklearn.datasets import load_linnerud

# Load dataset Linnerud
linnerud = load_linnerud()
X = linnerud.data    # Fitur: Chins, Situps, Jumps
y = linnerud.target  # Target: Weight, Waist, Pulse

reg = linear_model.LinearRegression()
reg.fit(X, y)

# Menampilkan Koefisien dan Intercept
print('Coefficients: \n', reg.coef_)
print('\nIntercept: \n', reg.intercept_)

# Prediksi untuk data baru 
sample_data = [[5, 80, 40]]
pred = reg.predict(sample_data)
print('\nPrediction (Weight, Waist, Pulse): \n', pred)