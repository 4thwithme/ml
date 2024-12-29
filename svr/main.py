import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values
y = y.reshape(len(y), 1)

ss_X = StandardScaler()
ss_y = StandardScaler()

X = ss_X.fit_transform(X)
y = ss_y.fit_transform(y)

regressor = SVR(kernel='rbf')

regressor.fit(X, y)

pred = ss_y.inverse_transform(regressor.predict(ss_X.transform([[6.5]])).reshape(-1,1))

plt.scatter(ss_X.inverse_transform(X), ss_y.inverse_transform(y), color='red')
plt.plot(ss_X.inverse_transform(X),  ss_y.inverse_transform(regressor.predict(X).reshape(-1,1))
, color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()



