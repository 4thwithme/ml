import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.common import random_state
from sklearn.ensemble import RandomForestRegressor

dataset = pd.read_csv("Position_Salaries.csv")

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X, y)

y_pred = regressor.predict([[6.5]])

print(y_pred)

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color="red")
plt.plot(X_grid, regressor.predict(X_grid), color="red")
plt.title("Salary vs Position (DTR)")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()
