import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

dataset = pd.read_csv("./Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
y = np.where(y == 2, 0, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

# KNN -----------------
classifier = KNeighborsClassifier(n_neighbors=6, metric="minkowski", p=2)
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)

c = 0
for i in range(len(y_pred)):
    if y_pred[i] != y_test[i]:
        c += 1

print("Count of errors KNN:", c)

cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
print("------------------------------------------")
# KNN -----------------

# Linear Regression -----------------
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)

c = 0
for i in range(len(y_pred)):
    if y_pred[i] != y_test[i]:
        c += 1

print("Count of errors LogisticRegression:", c)

cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
print("------------------------------------------")
# Linear Regression -----------------

# SVM -----------------

classifier = SVC(kernel="linear", random_state=0)
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)

c = 0
for i in range(len(y_pred)):
    if y_pred[i] != y_test[i]:
        c += 1

print("Count of errors SVM:", c)

cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

print("------------------------------------------")
# SVM -----------------


# Kernel SVM -----------------

classifier = SVC(kernel="rbf", random_state=0)
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)

c = 0
for i in range(len(y_pred)):
    if y_pred[i] != y_test[i]:
        c += 1

print("Count of errors Kernel:", c)

cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
print("------------------------------------------")
# Kernel SVM -----------------

# Naive Bayes -----------------

classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


c = 0
for i in range(len(y_pred)):
    if y_pred[i] != y_test[i]:
        c += 1

print("Count of errors Kernel:", c)

cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
print("------------------------------------------")
# Naive Bayes -----------------

# Decision Tree -----------------
classifier = DecisionTreeClassifier(criterion="entropy", random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

c = 0
for i in range(len(y_pred)):
    if y_pred[i] != y_test[i]:
        c += 1

print("Count of errors Decision Tree:", c)

cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

print("------------------------------------------")
# Decision Tree -----------------

# Random Forest -----------------
classifier = RandomForestClassifier(
    n_estimators=110, criterion="entropy", random_state=0
)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

c = 0
for i in range(len(y_pred)):
    if y_pred[i] != y_test[i]:
        c += 1

print("Count of errors Rand Forrest:", c)

cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

print("------------------------------------------")
# Random Forest -----------------


# XGBoost -----------------
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print(
    "Accuracy: {:.2f} %".format(accuracies.mean() * 100),
)


print("------------------------------------------")
