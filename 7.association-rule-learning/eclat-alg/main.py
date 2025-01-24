import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from apyori import apriori

dataset = pd.read_csv("Market_Basket_Optimisation.csv", header=None)
transactions = []

for i in range(0, len(dataset)):
    products = []
    for j in range(0, len(dataset.iloc[i])):
        products.append(str(dataset.values[i, j]))
    transactions.append(products)


rules = apriori(
    transactions=transactions,
    min_support=0.003,
    min_confidence=0.2,
    min_lift=3,
    min_length=2,
    max_length=2,
)

results = list(rules)


def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    return list(zip(lhs, rhs, supports))


resultsInDataFrame = pd.DataFrame(
    inspect(results),
    columns=["Product 1", "Product 2", "Support"],
)
resultsInDataFrame = resultsInDataFrame.nlargest(n=40, columns="Support")

print(resultsInDataFrame)
