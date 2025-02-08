import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print(tf.__version__)

# dataset import
dataset = pd.read_excel("Folds5x2_pp.xlsx")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

# ANN

ann = tf.keras.models.Sequential()
# add input and 1st hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))
# add 2nd hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))
# add output layer
ann.add(tf.keras.layers.Dense(units=1))

# Compile ANN
# optimizer - is tool that perform stochastic gradient descent
# loss - is a loss function that we want to minimize during training. Loss function is a measure of how well the model is doing
# metrics - we can specify a list of metrics that we want to use to evaluate our model
# for regression we use mean_squared_error loss function, for binary classification we use binary_crossentropy loss function
# for multiclass classification we use categorical_crossentropy loss function
# optimizer - adam is the most popular optimizer. It is a stochastic gradient descent algorithm
ann.compile(optimizer="adam", loss="mean_squared_error")

# Train ANN
ann.fit(X_train, y_train, batch_size=32, epochs=100)

y_pred = ann.predict(X_test)

single_observation = X_test[0]
single_observation = np.array([single_observation])
single_observation_y = y_test[0]

res = ann.predict(single_observation)

print(res, single_observation_y)

y_pred = ann.predict(X_test)

for i in range(len(y_pred)):
    pred = y_pred[i][0]
    test = y_test[i]
    print(
        f"Predicted: {pred}, Real: {test}, Diff: {abs(pred - test)}, Diff %: {abs(pred - test) / test * 100}"
    )

# save model
ann.save("model.h5")
