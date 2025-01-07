# Logisic Regression

> It helps to predict categorical data. It is a classification algorithm that is used to predict the probability of a categorical dependent variable. In logistic regression, the dependent variable is a binary variable that contains data coded as 1 (yes, success, etc.) or 0 (no, failure, etc.). In other words, the logistic regression model predicts P(Y=1) as a function of X.

Sigmoid function is used in logistic regression to predict the probability of the dependent variable.
**formula:**

```python
ln(p/(1-p)) = b0 + b1*x
```

where:

- p = probability of the dependent variable
- b0 = intercept
- b1 = coefficient of the independent variable

if dependat variable has more than 2 categories, then formula will be:

```python
ln(p/(1-p)) = b0 + b1*x1 + b2*x2 + b3*x3 + ... + bn*xn
```

**Predictions:**

**formula:**

```python
p = 1/(1 + e^-(b0 + b1*x))
```

Best curve is the one that has the maximum likelihood. The curve that has the maximum likelihood is the one that has the maximum probability of the dependent variable.

**formula:**

```python
L = p1^y1 * (1-p1)^(1-y1) * p2^y2 * (1-p2)^(1-y2) * ... * pn^yn * (1-pn)^(1-yn)
```

where:

- L = likelihood
- p = probability of the dependent variable
- y = actual value of the dependent variable
