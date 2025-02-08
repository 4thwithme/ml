import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# dataset
dataset = pd.read_csv("./Churn_Modelling.csv")
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Encode categorical data
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
# Encode Geo data
ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(), [1])], remainder="passthrough"
)
X = np.array(ct.fit_transform(X))

# split data on train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling
sc = StandardScaler()

sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

# ann

ann = tf.keras.models.Sequential()
# add input and 1st hidden layer
ann.add(tf.keras.layers.Dense(units=10, activation="relu"))
# add 2nd hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))
# add output layer
ann.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

# Compile ann
# for binary classificaton we always need to use binary_crossentropy loss function
# optimizer adam is the most popular optimizer. It is a stochastic gradient descent algorithm
ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train ann
ann.fit(X_train, y_train, batch_size=32, epochs=100)

single_observation = X_test[0]
single_observation = np.array([single_observation])
single_observation_y = y_test[0]

res = ann.predict(single_observation)

print(res, single_observation_y)

y_pred = ann.predict(X_test)
y_pred = y_pred > 0.5
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# save model
ann.save("model.h5")
