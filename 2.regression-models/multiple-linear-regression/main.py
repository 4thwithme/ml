import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression


dataset = pd.read_csv('startups.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

CT = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [-1])], remainder='passthrough',)
X = np.array(CT.fit_transform(X))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1) ), axis=1))

print(regressor.predict([[1, 0, 0, 260000, 230000, 500000]]))