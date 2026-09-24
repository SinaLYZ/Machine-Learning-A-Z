# Support Vector Regression in Python

Support vector regression (SVR) extends the support vector approach to a quantitative response. Unlike ordinary, multiple, and polynomial regression, it does not fit by minimizing the residual sum of squares (RSS); it minimizes a different loss that ignores small residuals entirely. ISLP mentions SVR only in a single paragraph, without equations (end of Chapter 9, page 386), so the math below follows ESL, Section 12.3.6.

## Linear model and loss function

ESL first writes the same linear model used elsewhere in the book:

$$
f(x) = x^T\beta + \beta_0.
$$

To estimate $\beta$ and $\beta_0$, ESL considers minimizing

$$
H(\beta, \beta_0) = \sum_{i=1}^{N} V(y_i - f(x_i)) + \frac{\lambda}{2}\|\beta\|^2,
$$

where $V$ is a chosen error measure and $\lambda$ is a nonnegative tuning parameter that controls the trade-off between fitting the data and keeping $\beta$ small.

Support vector regression uses the $\epsilon$-insensitive error measure

$$
V_\epsilon(r) =
\begin{cases}
0 & \text{if } \vert r \vert < \epsilon, \\
\vert r \vert - \epsilon & \text{otherwise},
\end{cases}
$$

which ignores residuals smaller than $\epsilon$ in absolute value; only residuals larger than $\epsilon$ contribute to the loss.

## Solution and support vectors

If $\hat\beta, \hat\beta_0$ minimize $H$, the solution has the form

$$
\hat\beta = \sum_{i=1}^{N}(\hat\alpha_i^* - \hat\alpha_i)x_i,
\qquad
\hat f(x) = \sum_{i=1}^{N}(\hat\alpha_i^* - \hat\alpha_i)\langle x, x_i\rangle + \hat\beta_0,
$$

where $\hat\alpha_i, \hat\alpha_i^*$ are nonnegative and solve a constrained quadratic program. Typically only a subset of the values $(\hat\alpha_i^* - \hat\alpha_i)$ are nonzero; the corresponding observations are called the support vectors. As with the support vector classifier, the solution depends on the training inputs only through their inner products $\langle x_i, x_{i'}\rangle$, so the method generalizes to richer feature spaces by replacing the inner product with a kernel.

## Interpretation

The parameter $\epsilon$ sets a margin of residuals that are treated as free; only points with residuals larger than $\epsilon$ become support vectors and influence the fit. The parameter $\lambda$ plays the same regularization role it plays elsewhere in the book, and is typically chosen by cross-validation. Because the fit is determined only by the support vectors, SVR can be less sensitive to observations that are already well predicted than ordinary least squares is.