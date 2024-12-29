### Split data for trainig set and test set

Usually we split data 80% for training and 20% for testing.

### Feature scaling

Feature scaling always applies to columns.

There are 2 types of feature scaling:

1. normalization
2. standartization

Normalization - is process of scaling and centering the data. It scales the data between -1 and 1. It is also called Min-Max scaling.
The formula for normalization is:

```py
    let x_normalized = (x - min(x)) / (max(x) - min(x))

    # where x is the column
    # min(x) is the minimum value of the column
    # max(x) is the maximum value of the column
```

```js
const min = Math.min(...xList);
const max = Math.max(...xList);
const x_normalized = xList.map((x) => (x - min) / (max - min));
```

Standardization - is process of scaling and centering the most data between -3 and 3. It scales the data with mean 0 and standard deviation 1.
The formula for standartization is:

```py
    let x_standardized = (x - mean(x)) / std(x)

    # where x is the column
    # mean(x) is the mean value of the column
    # std(x) is the standard deviation of the column
```

```js
const mean = xList.reduce((a, b) => a + b, 0) / x.length;
const std = Math.sqrt(
  x.map((x) => Math.pow(x - mean, 2)).reduce((a, b) => a + b) / x.length,
);
const x_standardized = x.map((x) => (x - mean) / std);
```
