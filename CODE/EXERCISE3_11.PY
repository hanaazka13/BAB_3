# Exercise 3.11 
import matplotlib.pyplot as plt
from scipy import stats

x = [0, 1, 2, 3, 4, 5, 6, 7]
y = [3, 5, 5, 6, 7, 8, 9, 11]


slope, intercept, r, p, std_err = stats.linregress(x, y)
print("slope: ", slope)
print("intercept: ", intercept)

def myfunc(x):
    return slope * x + intercept

mymodel = list(map(myfunc, x))

# Membuat Plot
plt.scatter(x, y, color='blue', label='Data Points')        # Menambahkan label untuk legend
plt.plot(x, mymodel, color='red', label='Regression Line') # Menambahkan label untuk legend

# elemen visual grafik
plt.xlabel('X Axis')                  # x label
plt.ylabel('Y Axis')                  # y label
plt.title('Linear Regression Plot')   # title
plt.legend()                          # legend
plt.grid(True)                        # grid

plt.show()