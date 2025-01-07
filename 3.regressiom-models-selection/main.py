import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Importing the dataset
dataset = pd.read_csv("Data.csv")

# Independent variables and dependent variable
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# splitting the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Multiple Linear Regression
regressor_ml = LinearRegression()
regressor_ml.fit(X_train, y_train)
y_pred_ml = regressor_ml.predict(X_test)


# Polynomial Regression
poly_reg = PolynomialFeatures(degree=9)
X_poly = poly_reg.fit_transform(X_train)
regressor_pl = LinearRegression()
regressor_pl.fit(X_poly, y_train)
y_pred_pl = regressor_pl.predict(poly_reg.transform(X_test))

# SVR

y_svr = y.reshape(len(y), 1)
X_train_svr, X_test_svr, y_train_svr, y_test_svr = train_test_split(
    X, y_svr, test_size=0.2, random_state=0
)

sc_X = StandardScaler()
sc_y = StandardScaler()
X_train_svr = sc_X.fit_transform(X_train_svr)
y_train_svr = sc_y.fit_transform(y_train_svr)

regressor_svr = SVR(kernel="rbf")
regressor_svr.fit(X_train_svr, y_train_svr)
y_pred_svr = sc_y.inverse_transform(
    regressor_svr.predict(sc_X.transform(X_test)).reshape(-1, 1)
)

# Decision Tree Model
regressor_dt = DecisionTreeRegressor(random_state=0)
regressor_dt.fit(X_train, y_train)
y_pred_dt = regressor_dt.predict(X_test)

# Random forrest model
regressor_rf = RandomForestRegressor(n_estimators=100, random_state=0)
regressor_rf.fit(X_train, y_train)
y_pred_rf = regressor_rf.predict(X_test)


# r2 score
r2_ml = r2_score(y_test, y_pred_ml)
r2_pl = r2_score(y_test, y_pred_pl)
r2_svr = r2_score(y_test, y_pred_svr)
r2_dt = r2_score(y_test, y_pred_dt)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"R2 score for Multiple Linear Regression: {r2_ml}")
print(f"R2 score for Polynomial Regression: {r2_pl}")
print(f"R2 score for SVR: {r2_svr}")
print(f"R2 score for Decision Tree: {r2_dt}")
print(f"R2 score for Random Forest: {r2_rf}")