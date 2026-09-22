# Simple Linear Regression in Python

Simple linear regression models a numerical response using one predictor. Ordinary least squares (OLS) estimates the intercept and slope by minimizing the residual sum of squares (RSS).

## Model and notation

For observations $(x_i, y_i)$, where $i = 1, \ldots, n$, the statistical model is

$$
y_i = \beta_0 + \beta_1 x_i + \varepsilon_i.
$$

- $n$: number of observations.
- $x_i$: predictor value for observation $i$.
- $y_i$: observed response.
- $\beta_0$: intercept, the expected response at $x = 0$ when the model holds and $E[\varepsilon_i \mid x_i] = 0$.
- $\beta_1$: slope, the change in expected response associated with a one-unit increase in $x$.
- $\varepsilon_i$: unobserved error.

After fitting, the prediction function is

$$
\hat f(x) = \hat\beta_0 + \hat\beta_1 x,
\qquad
\hat y_i = \hat f(x_i).
$$

A hat indicates an estimated coefficient or a predicted response. This formulation follows ISLP, Section 3.1 [1].

## Linear regression in summation form

With $p$ predictors, the general linear regression function is

$$
f(\mathbf{x}_i) = \beta_0 + \sum_{j=1}^{p} \beta_j x_{ij}.
$$

Here, $j$ indexes predictors and $i$ indexes observations. Simple linear regression is the special case $p = 1$:

$$
f(x_i) = \beta_0 + \sum_{j=1}^{1} \beta_j x_{ij}
= \beta_0 + \beta_1 x_i.
$$

The model is linear in its coefficients. The general form is discussed in ESL, Section 3.2, Equation (3.1) [2].

## Residual sum of squares (RSS)

A residual is the difference between an observed response and its fitted value:

$$
e_i = y_i - \hat y_i.
$$

The fitted model's RSS is

$$
\mathrm{RSS} = \sum_{i=1}^{n} e_i^2
= \sum_{i=1}^{n}(y_i - \hat y_i)^2.
$$

To fit the model, treat RSS as a function of candidate coefficients:

$$
\mathrm{RSS}(\beta_0, \beta_1)
= \sum_{i=1}^{n}(y_i - \beta_0 - \beta_1 x_i)^2.
$$

OLS chooses the coefficients that minimize this function:

$$
(\hat\beta_0, \hat\beta_1)
= \underset{\beta_0,\beta_1}{\operatorname{arg\,min}}
\sum_{i=1}^{n}(y_i - \beta_0 - \beta_1 x_i)^2.
$$

Squaring prevents positive and negative residuals from canceling and gives larger residuals more weight. RSS measures the total squared vertical distance from observations to the fitted line. A smaller training RSS indicates a closer fit to those observations, but does not guarantee better predictions on new data. See ISLP, Section 3.1.1, and ESL, Equation (3.2) [1, 2].

## Least-squares coefficient formulas

First calculate the sample means:

$$
\bar x = \frac{1}{n}\sum_{i=1}^{n}x_i,
\qquad
\bar y = \frac{1}{n}\sum_{i=1}^{n}y_i.
$$

The estimated slope and intercept are

$$
\hat\beta_1 =
\frac{\sum_{i=1}^{n}(x_i - \bar x)(y_i - \bar y)}
{\sum_{i=1}^{n}(x_i - \bar x)^2},
\qquad
\hat\beta_0 = \bar y - \hat\beta_1\bar x.
$$

These formulas are the solution obtained by setting the two partial derivatives of RSS to zero. They require variation in $x$: if all predictor values are identical, the denominator is zero and the slope and intercept cannot be uniquely estimated. See ISLP, Equation (3.4) [1].

## Python example

This example uses NumPy to implement the formulas directly.

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 5, 4, 5], dtype=float)

x_mean = x.mean()
y_mean = y.mean()
sxx = np.sum((x - x_mean) ** 2)
if sxx == 0:
    raise ValueError("The predictor must contain at least two distinct values.")

beta_1 = np.sum((x - x_mean) * (y - y_mean)) / sxx
beta_0 = y_mean - beta_1 * x_mean

y_pred = beta_0 + beta_1 * x
residuals = y - y_pred
rss = np.sum(residuals ** 2)
mse = rss / len(y)

print(f"Intercept: {beta_0:.3f}")
print(f"Slope: {beta_1:.3f}")
print(f"RSS: {rss:.3f}")
print(f"MSE: {mse:.3f}")
```

Output:

```text
Intercept: 2.200
Slope: 0.600
RSS: 2.400
MSE: 0.480
```

The fitted line is $\hat y = 2.2 + 0.6x$. Training mean squared error is $\mathrm{MSE} = \mathrm{RSS}/n$. For estimating the error variance under the usual simple linear regression assumptions, use $\hat\sigma^2 = \mathrm{RSS}/(n-2)$ instead, because two coefficients were estimated; this requires $n > 2$.

## Interpretation

OLS can be computed without assuming normally distributed errors. Statistical interpretation and inference require additional assumptions: a correctly specified linear conditional mean, errors with conditional mean zero, and appropriate assumptions about dependence and variance. Classical standard-error formulas assume uncorrelated errors with constant variance; exact small-sample t-based inference additionally assumes normal errors. A fitted association alone does not establish causation.

## References

[1] James, G., Witten, D., Hastie, T., Tibshirani, R., & Taylor, J. (2023). *An Introduction to Statistical Learning: with Applications in Python*. Springer. Section 3.1, especially Section 3.1.1, pp. 70-72; Section 3.1.3 for model accuracy; Section 3.6.2 for the Python lab. [Official book website](https://www.statlearning.com/).

[2] Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer. Section 3.2, especially Equations (3.1)-(3.6), pp. 44-45. [Official book website](https://hastie.su.domains/ElemStatLearn/main.html).

Page numbers refer to the printed book pages, not PDF viewer page numbers. The explanations and Python example above are original summaries based on these references.
