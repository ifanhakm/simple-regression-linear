import numpy as np
import statsmodels.api as sm

# Data
X = np.array([38, 54, 27, 32, 50, 51, 31, 34, 32, 32, 41, 59, 50, 31, 29, 51, 31, 26, 54, 30, 30, 33, 21, 53, 29])  # Variabel independen
Y = np.array([0.855, 0.845, 0.704, 0.672, 0.627, 0.587, 0.551, 0.537, 0.529, 0.484, 0.451, 0.398, 0.388, 0.351, 0.263, 0.254, 0.254, 0.248, 0.232, 0.201, 0.191, 0.183, 0.167, 0.158, 0.134])  # Variabel dependen

# Menambahkan kolom konstanta untuk intercept
X = sm.add_constant(X)

# Membuat model regresi linear sederhana
model = sm.OLS(Y, X).fit()

# Menampilkan hasil regresi
print(model.summary())