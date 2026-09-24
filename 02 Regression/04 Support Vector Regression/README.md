# Support Vector Regression in Python

Support vector regression (SVR) extends the support vector approach to a quantitative response. Unlike ordinary, multiple, and polynomial regression, it does not fit by minimizing the residual sum of squares (RSS); it minimizes a different loss that ignores small residuals entirely. ISLP mentions SVR only in a single paragraph, without equations (end of Chapter 9, page 386), so the math below follows ESL, Section 12.3.6.

## Linear model and loss function

ESL first writes the same linear model used elsewhere in the book:

```math
\Large f(x) = x^T\beta + \beta_0.
```

- $x$: the input vector, containing the predictor values for one observation.
- $f$: the prediction function, which maps an input to a numerical response.
- $\beta$: the vector of coefficients applied to the predictors.
- $\beta_0$: the intercept, the predicted response when all inputs are zero.
- $T$: the transpose symbol; the product here adds the input values multiplied by their corresponding coefficients.

With one predictor, this is a straight line. The coefficients control how predictions change with the inputs, while the intercept shifts the predictions up or down.

To estimate $\beta$ and $\beta_0$, ESL considers minimizing

```math
\Large H(\beta, \beta_0) = \sum_{i=1}^{N} V(y_i - f(x_i)) + \frac{\lambda}{2}\|\beta\|^2,
```

- $H$: the objective minimized during training.
- $y$: the observed response.
- $N$: the number of training observations.
- $i$: an observation index, used consistently for inputs, responses, and the solution weights below.
- $\sum$: adds the expression over the indicated observations.
- $\|\beta\|^2$: the sum of the squared coefficients, measuring the squared length of the coefficient vector.

$V$ is a chosen error measure and $\lambda$ is a nonnegative tuning parameter that controls the trade-off between fitting the data and keeping $\beta$ small.

The first term measures prediction discrepancies; the second discourages large coefficients. The intercept is not penalized. Increasing the regularization parameter places more emphasis on smaller coefficients, even if some predictions become less accurate. It is typically chosen by cross-validation.

Support vector regression uses the $\epsilon$-insensitive error measure

```math
\Large V_\epsilon(r) =
\begin{cases}
0 & \text{if } \vert r \vert < \epsilon, \\
\vert r \vert - \epsilon & \text{otherwise},
\end{cases}
```

Here, $r$ is a residual: the observed response minus its prediction. The symbol $\epsilon$ is a nonnegative tolerance in response units, and $\vert r\vert$ is the residual's magnitude regardless of sign.

The loss ignores residuals smaller than $\epsilon$ in absolute value; only residuals larger than $\epsilon$ contribute. At the boundary, the second branch also gives zero loss.

Imagine a tube extending the tolerance above and below the fitted curve. A point inside this tube incurs no penalty. Outside it, only the excess distance beyond the boundary is counted. For example, with a tolerance of 2, a residual magnitude of 5 contributes a loss of 3. The penalty grows linearly beyond the tube rather than quadratically as in squared-error loss. This tube describes a fitting tolerance, not a confidence interval.

## Solution and support vectors

If $\hat\beta, \hat\beta_0$ minimize $H$, the solution has the form

```math
\Large \hat\beta = \sum_{i=1}^{N}(\hat\alpha_i^* - \hat\alpha_i)x_i,
\qquad
\hat f(x) = \sum_{i=1}^{N}(\hat\alpha_i^* - \hat\alpha_i)\langle x, x_i\rangle + \hat\beta_0,
```

A hat marks a fitted quantity. The starred and unstarred $\alpha$ symbols label two families of nonnegative optimization weights; the star is a label, not multiplication. The brackets $\langle x,x_i\rangle$ denote an inner product, the sum of products of corresponding input coordinates.

The weights $\hat\alpha_i, \hat\alpha_i^*$ are nonnegative and solve a constrained quadratic program. Typically only a subset of the values $(\hat\alpha_i^* - \hat\alpha_i)$ are nonzero; the corresponding observations are called the support vectors. As with the support vector classifier, the solution depends on the training inputs only through their inner products $\langle x_i, x_{i'}\rangle$, so the method generalizes to richer feature spaces by replacing the inner product with a kernel.

The first expression writes the fitted coefficient vector as a weighted combination of training inputs. The second substitutes that combination into the prediction function, so each observation contributes according to its signed weight and its inner product with the new input.

Support vectors can lie on the tube boundary as well as outside it; a boundary point can have zero loss and still help determine the fit. Because predictions depend on the observations with nonzero weight differences, observations already strictly inside the tube can have no direct contribution to the fitted prediction sum. This explains how SVR can be less sensitive to observations that are already well predicted than ordinary least squares is.
## Connection to this notebook

<img src="figures/support_vector_regression_fit.png" alt="Observed salaries in red and the fitted RBF support vector regression curve in blue, plotted against position level." width="520">

*Saved plot from the notebook in the original position-level and salary units. The epsilon tube is not drawn.*

The [notebook](support_vector_regression.ipynb) uses an RBF kernel to fit a curved relationship between position level and salary. It scales the input and target separately before fitting, then converts predictions back to salary units. The dense plotting grid makes the fitted curve appear smooth; it does not add training observations. All 10 observations are used for fitting, so this plot illustrates the fit rather than performance on a held-out test set.
