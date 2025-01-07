import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('slry.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.scatter(X_test, y_test, color = 'blue')
plt.title('Salary/Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

print(regressor.score(X_test, y_test))
print(regressor.score(X_train, y_train))
print(regressor.predict([[12]]))
print(regressor.coef_)
print(regressor.intercept_)

