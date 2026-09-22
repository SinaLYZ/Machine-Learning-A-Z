# Simple Linear Regression

Mathematical foundations and variable definitions for the [simple linear regression notebook](simple_linear_regression.ipynb). The notebook applies these ideas to the included [salary dataset](Salary_Data.csv).

## Main Math for Simple Linear Regression

Simple linear regression models the relationship between one input variable $x$ and an output variable $y$ using a straight line.

### Notation and variable definitions

- $x$: the input variable, also called the predictor, feature, explanatory variable, or independent variable. Its value is used to predict the response; "independent" here does not imply statistical independence or causation.
- $y$: the observed output variable, also called the response, target, or dependent variable.
- $\hat{y}$: the model's predicted value of $y$ for a given $x$. The hat distinguishes a prediction from an observed value.
- $b_0$: the fitted intercept, the value of $\hat{y}$ when $x=0$. It has the same units as $y$.
- $b_1$: the fitted slope, the change in $\hat{y}$ for a one-unit increase in $x$. Its units are units of $y$ per unit of $x$.
- $n$: the number of observations in the training data used to fit the line.
- $i$: an observation index, ranging from $1$ to $n$.
- $x_i$: the observed input value for observation $i$.
- $y_i$: the observed response value for observation $i$.
- $\hat{y}_i$: the predicted response for observation $i$, calculated as $b_0+b_1x_i$.
- $e_i$: the residual for observation $i$, calculated as the observed response minus its fitted prediction, $y_i-\hat{y}_i$.
- $\bar{x}$: the arithmetic mean of the training input values $x_1,\ldots,x_n$.
- $\bar{y}$: the arithmetic mean of the training response values $y_1,\ldots,y_n$.

The symbol $\sum_{i=1}^{n}$ means to add a quantity over all $n$ training observations. The notation $\min_{b_0,b_1}$ means to choose the intercept and slope that make the following expression as small as possible.

### 1. The regression line

$$
\hat{y} = b_0 + b_1x
$$

- **$b_0$ — intercept:** predicted response when the input is zero ($x=0$).
- **$b_1$ — slope:** change in the predicted response for each one-unit increase in the input.

The intercept may have limited practical meaning if $x=0$ falls outside the range of the training inputs or has no meaningful interpretation for the input variable.

### 2. Residuals: prediction errors

For training observation $i$, the residual is:

$$
e_i = y_i - \hat{y}_i
    = y_i - (b_0 + b_1x_i)
$$

A positive residual means the model underestimates the observed response; a negative residual means it overestimates the observed response. A zero residual means the prediction equals the observed response.

### 3. Finding the best-fitting line

**Ordinary Least Squares (OLS)** chooses $b_0$ and $b_1$ to minimize the sum of squared residuals:

$$
\min_{b_0,b_1}
\sum_{i=1}^{n}\left[y_i-(b_0+b_1x_i)\right]^2
$$

Here, $n$ is the number of training observations. Squaring prevents positive and negative errors from canceling and gives larger errors more weight.

### 4. Calculating the slope and intercept

First, calculate the training-data means:

$$
\bar{x} = \frac{1}{n}\sum_{i=1}^{n}x_i,
\qquad
\bar{y} = \frac{1}{n}\sum_{i=1}^{n}y_i
$$

Then:

$$
b_1 =
\frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}
{\sum_{i=1}^{n}(x_i-\bar{x})^2}
$$

$$
b_0 = \bar{y} - b_1\bar{x}
$$

These formulas require the training $x$-values to vary; otherwise, the denominator is zero and the slope cannot be uniquely determined.

### 5. Connection to this notebook

For this particular dataset, $x$ represents years of experience and $y$ represents salary. These are examples of an input and a response; the definitions and equations above apply to any simple linear regression with one numeric predictor.

- `X_train`: the training input array, with one row per training observation and one feature column. Its entries correspond to the $x_i$ values.
- `y_train`: the observed training responses, corresponding to the $y_i$ values.
- `X_test`: the input array for held-out test observations.
- `y_test`: the observed responses for those test observations.
- `regressor`: the model object that stores the fitted intercept and slope after training.
- `b0` and `b1`: Python names for the fitted parameters $b_0$ and $b_1$.
- `y_pred`: the predictions for `X_test`, corresponding to $\hat{y}$ values on the test data.

```python
regressor.fit(X_train, y_train)
```

This fits the line using only the training data. The fitted parameters are available as:

```python
b0 = regressor.intercept_
b1 = regressor.coef_[0]
```

Predictions follow the same equation:

```python
y_pred = regressor.predict(X_test)
# Equivalent to:
# y_pred = b0 + b1 * X_test[:, 0]
```

The test data is reserved for evaluating predictions on observations that were not used to fit the line.
