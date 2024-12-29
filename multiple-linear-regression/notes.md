# Multiple Linear Regression

**formula**
```
y^ = b0 + b1*X1 + b2*X2... + bnXn
```
y<sup>^</sup> = b<sub>0</sub> + b<sub>1</sub>*X<sub>1</sub> + b<sub>2</sub>*X<sub>2</sub>... + b<sub>n</sub>X<sub>n</sub>

where:
- y^ - dependent variable
- Xi - independent varables
- bi - koef's
- const


Assumption for linear regression
1. linearity (linear relation btw X and y)
2. homoscedasticity (equal variance)
3. Multivariate Normality (Normal distribution)
4. Independence (of observation Includes "no auto-correlation")
5. Lack of multicollinearity (predictors are not correlated with each other)
6. The outlier check ( no "extra" value, no "error")

Building a model
1. "all in" model
   u use all variables' coz:
      - prior knowledge
      - or you have to
      - or you ar doing preparing for 'Backward elimination'
2. Backward Elimination
   1. Decide significance lvl to stay in the model (SL = 0.05)
   2. Fit all predictors ("all in" model)
   3. Consider the predictor with the highest P-value. If P > SL, go to STEP 4, otherwise go to FIN
   4. Remove the predictor 
   5. Fit model without this variable*
   6. FIN
3. Forward Selection 
   1. Select a significance level to enter the model (e.g. SL = 0.05)
   2. Fit all simple regression models y ~ Xn Select the one with the lowest P-value 
   3. Keep this variable and fit all possible models with one extra predictor added to the one(s) you
   already have 
   4. Consider the predictor with the lowest P-value. If P < SL, go to STEP 3, otherwise go to FIN
   5. FIN
4. Bidirectional Elimination
   1. Select a significance level to enter and to stay in the model
   e.g. SLENTER = 0.05, SLSTAY = 0.05 
   2. Perform the next step of Forward Selection (new variables must have: P < SLENTER to enter)
   3. Perform ALL steps of Backward Elimination (old variables must have P < SLSTAY to stay)
   4. No new variables can enter and no old variables can exit