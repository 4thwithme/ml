import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from apyori import apriori

dataset = pd.read_csv("./Market_Basket_Optimisation.csv", header=None)
transactions = []

for i in range(0, len(dataset)):
    row = dataset.iloc[i]
    products = []
    for j in range(0, len(row)):
        products.append(str(dataset.values[i, j]))
    transactions.append(products)

rules = apriori(
    transactions=transactions,
    min_support=0.0025,
    min_confidence=0.2,
    min_lift=3,
    min_length=2,
    max_length=2,
)

results = list(rules)

print(results)


def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))


resultsinDataFrame = pd.DataFrame(
    inspect(results),
    columns=["Left Hand Side", "Right Hand Side", "Support", "Confidence", "Lift"],
)
resultsinDataFrame = resultsinDataFrame.nlargest(n=20, columns="Lift")

print(resultsinDataFrame)
