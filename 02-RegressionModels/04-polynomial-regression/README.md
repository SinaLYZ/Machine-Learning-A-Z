# Polynomial Regression in Python

Polynomial regression models a curved relationship between one numerical predictor and a numerical response by including powers of the predictor. It still uses ordinary least squares (OLS) to minimize the residual sum of squares (RSS). The [notebook](polynomial_regression.ipynb) compares a straight line with a degree-4 polynomial using the included [position salary dataset](Position_Salaries.csv).

## Polynomial regression in summation form

The model represents the average response as

$$
\Large f(X) = E(Y \mid X) = \beta_0 + \sum_{j=1}^{d}\beta_j X^j.
$$

- $X$: the input variable, also called the predictor, feature, or explanatory variable.
- $Y$: the response, also called the output, target, or dependent variable.
- $E(Y \mid X)$: the conditional mean, or average response among observations with the given input value. The model uses $f(X)$ to represent this mean.
- $\beta_0$: the intercept, the modeled mean response at an input of zero. Its units are the same as the response.
- $\beta_j$: the coefficient multiplying the input raised to power $j$. Its units are response units divided by input units raised to that power.
- $d$: the polynomial degree, the highest power included in the model; $j$ indexes the powers from $1$ to $d$.
- $\sum$: the summation symbol, which adds the terms over the indicated index range.

The equality states that the conditional mean has the specified polynomial form. When the true relationship has a different shape, the polynomial can still serve as an approximation. Degree 1 gives a straight line, degree 2 adds a quadratic term, and higher degrees allow more changes in curvature.

Polynomial regression is **linear in its coefficients**: each coefficient multiplies a known function of the input, and the resulting terms are added. The input can appear as a square or a fourth power without making the fitting problem nonlinear in the coefficients. This lets us use the same least-squares machinery as multiple linear regression, treating the powers as transformed feature columns.

For a degree-4 fit, the prediction equation is

$$
\Large \hat y = \hat\beta_0 + \hat\beta_1x + \hat\beta_2x^2 + \hat\beta_3x^3 + \hat\beta_4x^4.
$$

Lowercase $x$ and $y$ denote observed input and response values. A hat indicates a quantity estimated from data: the hatted coefficients are fitted parameters, and $\hat y$ is the predicted response. The intercept may have little practical meaning if zero lies outside the observed input range.

Unlike separate predictors that can vary independently, powers of the same input change together. A polynomial coefficient therefore cannot be interpreted as the overall change in the response for a one-unit increase in the original input. The fitted curve's local slope is

$$
\Large \frac{\mathrm d\hat y}{\mathrm dx}
= \sum_{j=1}^{d} j\hat\beta_jx^{j-1}.
$$

The derivative on the left measures the instantaneous change in the prediction per unit of input at a particular input value. Its upright differential symbols denote differentiation, not the polynomial degree. The slope generally changes along the curve: a small input increase can have a different predicted impact at low and high input values. For an exact one-unit change, compare the curve's predictions at the two input values. These are fitted associations, not evidence of causation.

## Residual sum of squares (RSS)

The difference between an observed response and its prediction is

$$
\Large e_i = y_i - \hat y_i.
$$

Here, $e$ is the residual. The subscript $i$ identifies an observation and applies consistently to inputs, responses, predictions, and residuals. A positive residual means the model underpredicts; a negative residual means it overpredicts. On a plot, these are vertical differences between the observed points and the curve, not perpendicular distances to the curve.

Square the residuals and add them to measure the overall training discrepancy:

$$
\Large \mathrm{RSS} = \sum_{i=1}^{n}e_i^2
= \sum_{i=1}^{n}\left(y_i - \hat\beta_0 - \sum_{j=1}^{d}\hat\beta_jx_i^j\right)^2.
$$

Here, $n$ is the number of training observations. The inner sum combines the polynomial terms for one observation; the outer sum adds the squared discrepancies across observations. Squaring prevents positive and negative residuals from canceling and penalizes larger discrepancies more strongly: doubling a residual makes its contribution four times as large.

For a chosen degree, OLS fits all coefficients jointly to minimize RSS. Increasing the degree cannot increase the minimum training RSS on the same observations, because the extra coefficient can be set to zero to recover the lower-degree fit. However, extra flexibility can fit noise as well as the underlying relationship. Choose the degree using validation data or cross-validation rather than training error alone.

High-degree curves can oscillate near the edges of the observed range and grow rapidly outside it. A close fit within the data therefore does not justify extrapolating far beyond the observed inputs. Even a curve that passes through every training point may predict new observations poorly.

Computing OLS does not require normally distributed errors. Interpreting the polynomial as the conditional mean requires the underlying errors (deviations from that mean, rather than fitted residuals) to have conditional mean zero. Classical standard-error formulas assume uncorrelated errors with constant variance, and exact small-sample t-based inference additionally assumes normal errors.

## Least-squares coefficient formulas

Arrange the polynomial features into a matrix:

$$
\Large
\mathbf X =
\begin{bmatrix}
1 & x_1 & x_1^2 & \cdots & x_1^d \\
1 & x_2 & x_2^2 & \cdots & x_2^d \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_n & x_n^2 & \cdots & x_n^d
\end{bmatrix}.
$$

The bold $\mathbf X$ is the design matrix, with one row per training observation and one column per polynomial term, including the constant term. It has $n$ rows and $d+1$ columns. Unlike the original predictor $X$, it contains the transformed inputs for the entire training dataset. The column of ones supplies the intercept.

Predictions for all training observations can then be written as

$$
\Large \hat{\mathbf y} = \mathbf X\hat{\boldsymbol\beta}.
$$

- $\hat{\boldsymbol\beta}$: the column vector of fitted coefficients, ordered from the intercept through the coefficient of the highest power.
- $\mathbf y$ and $\hat{\mathbf y}$: the column vectors collecting the observed and predicted training responses, respectively.

Multiplying a row of the design matrix by the coefficient vector evaluates the fitted polynomial at that row's input value. This is the same matrix operation used in multiple linear regression; only the construction of the feature columns differs.

Setting the partial derivatives of RSS with respect to the coefficients to zero gives the **normal equations**:

$$
\Large \mathbf X^{\mathsf T}\mathbf X\hat{\boldsymbol\beta}
= \mathbf X^{\mathsf T}\mathbf y.
$$

The superscript $\mathsf T$ denotes a transpose, which exchanges a matrix's rows and columns. These equations balance the residuals against each polynomial feature column. Because the design matrix includes a column of ones, the fitted training residuals sum to zero, apart from numerical rounding.

If the design matrix has full column rank, the unique coefficient solution is

$$
\Large \hat{\boldsymbol\beta}
= (\mathbf X^{\mathsf T}\mathbf X)^{-1}\mathbf X^{\mathsf T}\mathbf y.
$$

The superscript $-1$ denotes a matrix inverse. Full column rank means no column is an exact linear combination of the others. For this single-input polynomial basis, this requires at least $d+1$ distinct training input values. Otherwise, different coefficient vectors can give the same fitted training predictions, and the inverse does not exist.

Even with enough distinct inputs, large powers can make the matrix numerically ill-conditioned: small rounding or data changes can lead to large coefficient changes. Centering and scaling the input before generating powers can help. Numerical least-squares methods are preferable to explicitly calculating the inverse shown in the formula.

## Connection to this notebook

The notebook uses `Level` to predict `Salary`, excluding the descriptive `Position` column. It fits both a straight-line baseline and a degree-4 polynomial to all 10 observations, plots their fitted predictions, and predicts salary at level 6.5.

`PolynomialFeatures(degree=4)` expands each level into a constant column and powers through the fourth degree. `LinearRegression` then fits the transformed features. The constant column is redundant with its default fitted intercept; a cleaner parameterization would use `include_bias=False` when generating features and retain the fitted intercept. The mathematical design matrix above represents the constant term only once.

The dense plotting grid evaluates the same fitted curve at more input values, making it look smooth; it does not add training observations or improve the fit. Because this notebook uses the whole dataset without a held-out test set or evaluation metrics, its plots illustrate the fitted relationship rather than establish accuracy on unseen data.
