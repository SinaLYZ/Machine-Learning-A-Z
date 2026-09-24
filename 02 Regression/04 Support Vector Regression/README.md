# Support Vector Regression in Python

Support vector regression (SVR) predicts a numerical response by balancing model complexity against errors outside a chosen tolerance. The [notebook](support_vector_regression.ipynb) uses an RBF kernel to predict salary from position level in the included [dataset](Position_Salaries.csv).

## Linear model and residuals

Start with a linear prediction function:

$$
\Large f(x) = x^{\mathsf T}\beta + \beta_0.
$$

- $x$: the input vector, containing the predictor values for one observation.
- $f$: the prediction function, which maps an input to a numerical response.
- $\beta$: the vector of weights multiplying the predictors.
- $\beta_0$: the intercept, the predicted response when all input values are zero.
- $\mathsf T$: the transpose symbol; here, the product adds the predictor values multiplied by their corresponding weights.

With one predictor, this is a straight line. Kernels, introduced below, allow the same approach to produce curved predictions.

The prediction discrepancy is

$$
\Large r_i = y_i - f(x_i).
$$

Here, $y$ denotes the observed response and $r$ the residual. A subscript $i$ identifies an observation throughout the equations. Positive residuals indicate underprediction; negative residuals indicate overprediction. These differences are measured in response units.

## The epsilon-insensitive loss

SVR assigns the following loss to a residual:

$$
\Large V_\epsilon(r) = \max(0, |r| - \epsilon).
$$

- $V_\epsilon$: the epsilon-insensitive loss function.
- $\epsilon$: a nonnegative tolerance, measured in the same units as the response.
- $|r|$: the absolute value of the residual, its magnitude regardless of sign.
- $\max$: selects the larger of its two arguments.

Imagine a band extending epsilon units above and below the prediction curve. An observation inside this band, or exactly on its edge, has zero loss. Outside the band, only the excess distance beyond the edge is penalized. For example, with a tolerance of 2, residual magnitudes of 1 and 2 have zero loss, while a magnitude of 5 has loss 3.

The band has total vertical width twice the tolerance. It expresses an acceptable training discrepancy; it is not a confidence interval or a prediction interval. Outside the band, the loss grows linearly rather than quadratically, so it gives large residuals less rapidly increasing penalties than squared-error loss. This does not make the model immune to outliers.

## Balancing loss and model complexity

One way to express the training objective is

$$
\Large H(\beta,\beta_0)
= \sum_{i=1}^{n} V_\epsilon\!\left(y_i-f(x_i)\right)
+ \frac{\lambda}{2}\|\beta\|^2.
$$

- $H$: the objective minimized during fitting.
- $n$: the number of training observations.
- $\sum$: adds the terms over the indicated index range.
- $\lambda$: a positive regularization parameter controlling the penalty on the weights.
- $\|\beta\|^2$: the squared Euclidean length of the weight vector, equal to the sum of its squared entries.

The first term measures discrepancies outside the tolerance band. The second discourages large weights; the intercept is not penalized. A larger regularization parameter gives more emphasis to keeping the weights small, even if that leaves more observations outside the band. For a single linear predictor, a smaller weight gives a flatter line.

The equivalent parameterization used for epsilon-SVR with `C` is

$$
\Large \min_{\beta,\beta_0}
\left[\frac{1}{2}\|\beta\|^2
+ C\sum_{i=1}^{n}V_\epsilon\!\left(y_i-f(x_i)\right)\right].
$$

Here, $C$ is a positive parameter weighting the loss, and $\min$ means to choose the indicated parameters to make the expression as small as possible. For these exact summed-loss objectives, dividing the first objective by its regularization parameter gives $C=1/\lambda$. Larger values of `C` emphasize fitting observations outside the band; smaller values emphasize regularization. This reciprocal relationship changes if an alternative objective averages the loss or uses a different normalization.

Epsilon and regularization have different jobs: one sets which residuals are free, while the other controls how strongly the remaining discrepancies compete with model simplicity. Fitting SVR does not require a normal-error assumption, and its loss does not generally target the conditional mean in the same way squared-error loss does.

## Kernels and support vectors

A fitted kernel SVR predicts using a weighted combination of training-input similarities:

$$
\Large \hat f(x) = \sum_{i=1}^{n} a_i K(x_i,x) + \hat\beta_0.
$$

- A hat marks a fitted quantity, such as the learned function or intercept.
- $a_i$: a signed coefficient learned for a training observation by solving the constrained SVR optimization problem.
- $K$: a kernel, which computes an inner product in a possibly transformed feature space.

The observations with nonzero signed coefficients are the **support vectors**. Other training observations contribute zero to this prediction sum. With positive epsilon, observations strictly inside the fitted tube have zero coefficients in the exact optimum; support vectors can lie on the tube boundary as well as outside it. A point can therefore have zero loss and still help determine the fit. This distinction is why support vectors should not be described only as points with errors larger than epsilon.

A linear kernel computes an ordinary input dot product and recovers a linear function. Other kernels make the model linear in a transformed feature space while allowing a nonlinear curve in the original input. The weight penalty then measures complexity in that feature space, rather than simply the slope of the plotted curve.

The notebook uses the radial basis function (RBF) kernel:

$$
\Large K(x_i,x) = \exp\!\left(-\gamma\|x_i-x\|^2\right).
$$

Here, $\gamma$ is a positive parameter controlling how quickly similarity decreases with squared input distance; $\exp$ is the exponential function. The norm notation now measures the Euclidean distance between two inputs.

Nearby inputs have higher similarity. Increasing gamma makes each contribution more localized, allowing sharper changes; decreasing it spreads contributions over a broader range. Neither setting guarantees good predictions: gamma, `C`, and epsilon interact. Far from all training inputs, RBF similarities approach zero, so predictions approach the fitted intercept rather than continuing a learned linear trend.

## Scaling and connection to this notebook

The notebook standardizes both position level and salary before fitting. For a single predictor, the transformations are

$$
\Large z_x = \frac{x-\mu_x}{s_x},
\qquad
z_y = \frac{y-\mu_y}{s_y}.
$$

Here, $z$ denotes a standardized value, $\mu$ a training-data mean, and $s$ a training-data standard deviation; the subscripts identify the predictor or response. Separate scalers are necessary because the two variables have different units and distributions.

Input scaling changes the distances used by the RBF kernel. Target scaling changes the units in which epsilon is interpreted: on a standardized target, a tolerance of 0.1 corresponds to one tenth of the target's fitted standard deviation, not 0.1 salary units.

The notebook sets `kernel='rbf'` and leaves the other parameters at the pinned scikit-learn version's defaults: `C=1.0`, `epsilon=0.1`, and `gamma='scale'`. See the [scikit-learn 1.5 SVR reference](https://scikit-learn.org/1.5/modules/generated/sklearn.svm.SVR.html) for those settings, and the [SVR mathematical formulation](https://scikit-learn.org/1.5/modules/svm.html#regression) for the constrained optimization behind the fit.

To predict salary at level 6.5, the notebook first transforms the level with the input scaler, obtains a prediction in standardized target units, and converts it back:

$$
\Large \hat y = \mu_y + s_y\hat z_y.
$$

This reverses target standardization so the result is expressed in salary units. The same transformation is used to plot predictions alongside the original observations.

The notebook fits all 10 observations and has no held-out evaluation set. Its smooth plotting grid illustrates the fitted curve; it does not add training information or establish accuracy on unseen data. For evaluation, split before fitting either scaler and choose hyperparameters using validation data or cross-validation, fitting preprocessing only on the training portion of each split.
