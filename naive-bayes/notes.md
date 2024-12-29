# Naive Bayes Intuition

## Bayes Theorem

**formula**:

```py
P(A|B) = P(B|A) * P(A) / P(B)
```

- P(A|B) is the probability of A given B
- P(B|A) is the probability of B given A
- P(A) is the probability of A
- P(B) is the probability of B

Example:

We have daraset of people who drive car and walk. We have to find the probability of a person who drives a car given that he walk.

```py
 P(Walks|X) = P(X|Walks) * P(Walks) / P(X)
 P(Drives|X) = P(X|Drives) * P(Drives) / P(X)
```

1. P(Walks) = num of walkers / total num of people
2. P(X) = num of similar people to X (inide the circle) / total num of people
3. P(X|Walks) = num of similar walkers to X (inside the circle) / total num of walkers
4. P(X|Drives) = num of similar drivers to X (inside the circle) / total num of drivers

## In case of 3 features

**formula**:

```py
P(Y|X) = P(X1|Y) * P(X2|Y) * P(X3|Y) * P(Y) / P(X1) * P(X2) * P(X3)
```
