import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

dataset = pd.read_csv("./Social_Network_Ads.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

sc = StandardScaler()
sc.fit(X)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

classifier = SVC(kernel="linear", random_state=0)
classifier.fit(X_train, y_train)

y_0_predict = classifier.predict(sc.transform([X_test[0]]))

print(y_0_predict)

y_pred = classifier.predict(X_test)

c = 0
for i in range(len(y_pred)):
    print(y_pred[i], y_test[i])
    if y_pred[i] != y_test[i]:
        c += 1

print("Count of errors:", c)

cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
