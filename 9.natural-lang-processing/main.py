import re

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

dataset = pd.read_csv("./Restaurant_Reviews.tsv", delimiter="\t", quoting=3)

# cleaning the text
corpus = []
all_stopwords = stopwords.words("english")
all_stopwords.remove("not")
all_stopwords.remove("no")
all_stopwords.remove("don't")
all_stopwords.remove("doesn't")
all_stopwords.remove("didn't")
all_stopwords.remove("aren't")
all_stopwords.remove("weren't")
all_stopwords.remove("won't")
all_stopwords.remove("isn't")
all_stopwords.remove("is")
all_stopwords.remove("are")

for i in range(0, len(dataset)):
    review = dataset["Review"][i]
    review = re.sub("[^a-zA-Z]", " ", review).lower().split()
    ps = PorterStemmer()
    review_cleaned = []
    for word in review:
        if word not in set(all_stopwords):
            review_cleaned.append(ps.stem(word))
    review_cleaned = " ".join(review_cleaned)
    corpus.append(review_cleaned)

cv = CountVectorizer(max_features=750)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

print(len(X[0]))  # 1566 -> means we use only 1500 most frequent words

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# classifier = GaussianNB()
# classifier.fit(X_train, y_train)
# 0.73

# classifier = RandomForestClassifier(
#     n_estimators=400, criterion="entropy", random_state=0
# )
# classifier.fit(X_train, y_train)
# 0.78

classifier = SVC(kernel="poly", degree=2, random_state=0)
classifier.fit(X_train, y_train)
# 0.81 linear
# 0.82 poly degree=2, 1500 features
# 0.825 poly degree=2, 800 features
# 0.83 poly degree=2, 750 features


# classifier = LogisticRegression(random_state=0)
# classifier.fit(X_train, y_train)
# 0.795

y_pred = classifier.predict(X_test)


cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))


# check is review is positive or negative

# neg_review = "All dishes were perfect."
# neg_review = "I will not come back again."
neg_review = "Food was not good."
neg_review = re.sub("[^a-zA-Z]", " ", neg_review).lower().split()
review_cleaned_2 = []
for word in neg_review:
    if word not in set(all_stopwords):
        review_cleaned_2.append(ps.stem(word))
review_cleaned_2 = " ".join(review_cleaned_2)
print(review_cleaned_2)
review_cleaned_2 = cv.transform([review_cleaned_2]).toarray()

print(classifier.predict(review_cleaned_2))
