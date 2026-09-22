# Simple Linear Regression in Python

Simple linear regression models a numerical response using one predictor. Ordinary least squares (OLS) estimates the intercept and slope by minimizing the residual sum of squares (RSS).

## Linear regression in summation form

A linear regression model assumes that the regression function $E(Y|X)$ is linear in the inputs $X_1, \ldots, X_p$:

$$
f(X) = \beta_0 + \sum_{j=1}^{p} X_j \beta_j.
$$

The linear model either assumes that the regression function $E(Y|X)$ is linear, or that the linear model is a reasonable approximation. Simple linear regression is the special case $p = 1$, with a single predictor.

## Residual sum of squares (RSS)

![OLS fit and residuals for Salary vs. Years of Experience: observed points, the fitted line, and the vertical residual segment at each point.](simple_linear_regression_fit.png)

*Fit on this repo's `Salary_Data.csv`. Each grey vertical segment is one residual; RSS is the sum of their squared lengths.*

Let $\hat y_i = \hat\beta_0 + \hat\beta_1 x_i$ be the prediction for $y_i$ based on the $i$th value of $x$. Then $e_i = y_i - \hat y_i$ represents the $i$th residual. The residual sum of squares is

$$
\mathrm{RSS} = \sum_{i=1}^{n} e_i^2 = \sum_{i=1}^{n}(y_i - \hat\beta_0 - \hat\beta_1 x_i)^2.
$$

The least squares approach chooses $\hat\beta_0$ and $\hat\beta_1$ to minimize the RSS.

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

These formulas are the solution obtained by setting the two partial derivatives of RSS to zero. They require variation in $x$: if all predictor values are identical, the denominator is zero and the slope and intercept cannot be uniquely estimated.

## Interpretation

OLS can be computed without assuming normally distributed errors. Statistical interpretation and inference require additional assumptions: a correctly specified linear conditional mean, errors with conditional mean zero, and appropriate assumptions about dependence and variance. Classical standard-error formulas assume uncorrelated errors with constant variance; exact small-sample t-based inference additionally assumes normal errors. A fitted association alone does not establish causation.
