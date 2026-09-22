# Simple Linear Regression in Python

Simple linear regression models a numerical response using one predictor. Ordinary least squares (OLS) fits a straight line by minimizing the residual sum of squares (RSS). The [notebook](simple_linear_regression.ipynb) applies this method to salary and years of experience; the mathematics below applies to any numerical response and single numerical predictor.

## Linear regression in summation form

The general linear regression model represents the average response as

$$
\Large f(X) = E(Y \mid X) = \beta_0 + \sum_{j=1}^{p} X_j\beta_j.
$$

- $Y$: the response, also called the output, target, or dependent variable.
- $X$: the collection of predictors, also called inputs, features, or explanatory variables; $X_j$ is its $j$th predictor.
- $E(Y \mid X)$: the conditional mean, or average response among observations with the given predictor values. The model uses $f(X)$ to represent this mean.
- $\beta_0$: the intercept, the modeled mean response when all predictors are zero. Its units are the same as the response.
- $\beta_j$: the coefficient of predictor $j$, the change in the modeled mean response for a one-unit increase in that predictor while holding the others fixed. Its units are response units per predictor unit.
- $p$: the number of predictors; $j$ indexes them from $1$ to $p$.
- $\sum$: the summation symbol, which adds the terms over the indicated index range.

This equality states the linear conditional-mean assumption. If the true relationship is not linear, a straight line can still serve as an approximation, but it may miss systematic patterns. A fitted association alone does not establish causation.

For simple linear regression, there is only one predictor ($p=1$), so the fitted prediction becomes

$$
\Large \hat y = \hat\beta_0 + \hat\beta_1 x.
$$

Lowercase $x$ and $y$ denote observed values of the predictor and response. A hat indicates a quantity estimated from data: $\hat\beta_0$ and $\hat\beta_1$ are the fitted coefficients, and $\hat y$ is the predicted response. With one predictor, its coefficient is the slope of the line: a positive slope gives increasing predictions as the input increases, while a negative slope gives decreasing predictions. The intercept may have little practical meaning when zero is outside the observed input range.

## Residual sum of squares (RSS)

<img src="simple_linear_regression_fit.png" alt="Salary versus years of experience, with observed points, the fitted OLS line, and vertical residual segments." width="520">

*Fit on this repo's `Salary_Data.csv`. Each grey vertical segment connects an observed point to the fitted line at the same input value.*

The difference between an observation and its prediction is

$$
\Large e_i = y_i - \hat y_i.
$$

Here, $e$ is the residual. The subscript $i$ identifies an observation and applies consistently to inputs, responses, predictions, and residuals. A positive residual means the point lies above the line and the model underpredicts; a negative residual means the point lies below the line and the model overpredicts. These are vertical differences in response units, not perpendicular distances to the line.

To measure the overall discrepancy, square each residual and add the results:

$$
\Large \mathrm{RSS} = \sum_{i=1}^{n} e_i^2
= \sum_{i=1}^{n}\left(y_i - \hat\beta_0 - \hat\beta_1 x_i\right)^2.
$$

Here, $n$ is the number of training observations. RSS is a single measure of how closely the line fits those observations. Squaring prevents positive and negative residuals from canceling and penalizes large residuals more strongly: doubling a residual makes its contribution four times as large.

OLS chooses the fitted coefficients that produce the smallest RSS. Moving the intercept shifts the line vertically; changing the slope tilts it. The best-fitting line balances these adjustments across all training points. A low training RSS alone does not guarantee accurate predictions on new data, so the notebook reserves a test set for evaluating predictions.

Minimizing RSS does not require normally distributed errors. To interpret the line as the conditional mean, however, the underlying errors (deviations from that mean, rather than fitted residuals) must have conditional mean zero. Uncertainty estimates also need assumptions about dependence and variance: classical standard-error formulas assume uncorrelated errors with constant variance, and exact small-sample t-based inference additionally assumes normal errors.

## Least-squares coefficient formulas

First calculate the training-data means:

$$
\Large \bar x = \frac{1}{n}\sum_{i=1}^{n}x_i,
\qquad
\bar y = \frac{1}{n}\sum_{i=1}^{n}y_i.
$$

The bars denote arithmetic means: $\bar x$ is the average training input and $\bar y$ is the average training response. Subtracting these means centers the observations, so the following calculation measures how inputs and responses vary around their respective averages.

The fitted slope is

$$
\Large \hat\beta_1 =
\frac{\sum_{i=1}^{n}(x_i - \bar x)(y_i - \bar y)}
{\sum_{i=1}^{n}(x_i - \bar x)^2}.
$$

The numerator measures how the input and response vary together. Observations where both are above their means, or both below, contribute positively; observations where they are on opposite sides contribute negatively. The denominator measures the total squared spread of the inputs. Dividing by this spread converts the joint variation into a change in response per unit of input.

Once the slope is known, calculate the fitted intercept:

$$
\Large \hat\beta_0 = \bar y - \hat\beta_1\bar x.
$$

This places the fitted line through the point $(\bar x,\bar y)$. Substituting the average input into the fitted equation therefore gives the average training response.

These formulas minimize RSS by setting its partial derivatives with respect to the two coefficients to zero and solving the resulting equations. They require variation in the inputs: if all input values are identical, the slope denominator is zero. Many slope-and-intercept combinations then give the same predictions at that single input value, so the two coefficients cannot be uniquely estimated.
