import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

lin_reg = LinearRegression()
lin_reg.fit(X, y)

plt.scatter(X, y, color="red")
plt.plot(X, lin_reg.predict(X), color="red")
plt.title("Salary vs Position")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()

poly_reg = PolynomialFeatures(degree=9)
X_poly = poly_reg.fit_transform(X)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

plt.scatter(X, y, color="red")
plt.plot(X, lin_reg_2.predict(poly_reg.transform(X)), color="red")
plt.title("Salary vs Position (Polynomial Features)")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()

print(lin_reg.predict([[6.5]]))
print(lin_reg_2.predict(poly_reg.transform([[6.5]])))
