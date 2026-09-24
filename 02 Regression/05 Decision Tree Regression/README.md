# Decision Tree Regression in Python

Decision tree regression models a numerical response by partitioning the predictor space into regions and predicting the same value for every observation that falls into a region, rather than fitting one global equation as ordinary, multiple, and polynomial regression do. ISLP presents this in Section 8.1.1, "Regression Trees" (pages 331–337), with a worked example predicting a baseball player's salary; ESL covers the same construction in Section 9.2.2, "Regression Trees" (pages 307–308). The equations below follow ISLP's notation.

## Regions and predictions

The predictor space — the set of possible values for the predictors $X_1, X_2, \ldots, X_p$ — is divided into $J$ distinct, non-overlapping regions $R_1, R_2, \ldots, R_J$. Every observation that falls into a given region receives the same prediction: the mean of the response values for the training observations in that region.

The regions are chosen to minimize the residual sum of squares

```math
\Large \sum_{j=1}^{J}\sum_{i\in R_j}(y_i-\hat y_{R_j})^2,
```

- $J$: the number of regions the predictor space is divided into.
- $R_j$: the $j$th region, one of $J$ non-overlapping subsets of the predictor space.
- $y_i$: the observed response for training observation $i$.
- $\hat y_{R_j}$: the mean response for the training observations within the $j$th region.

Squaring the differences between each observation's response and its region's mean prevents positive and negative discrepancies from canceling. Considering every possible partition into $J$ regions is computationally infeasible, so the regions are built one split at a time instead.

## Recursive binary splitting

Starting with all of the data, a splitting predictor $j$ and split point $s$ divide the predictor space into a pair of half-planes:

```math
\Large R_1(j,s) = \{X\mid X_j < s\} \quad\text{and}\quad R_2(j,s) = \{X\mid X_j \ge s\}.
```

- $j$: the index of the predictor selected for the split.
- $s$: the cutpoint at which that predictor is split.
- $R_1(j,s)$: the half-plane containing predictor values below the cutpoint.
- $R_2(j,s)$: the half-plane containing predictor values at or above the cutpoint.

The values of $j$ and $s$ are chosen to minimize

```math
\Large \sum_{i:\,x_i\in R_1(j,s)}(y_i-\hat y_{R_1})^2 + \sum_{i:\,x_i\in R_2(j,s)}(y_i-\hat y_{R_2})^2,
```

where $\hat y_{R_1}$ and $\hat y_{R_2}$ are the mean responses for the training observations in $R_1(j,s)$ and $R_2(j,s)$, respectively. Having found the best split, the same splitting process is repeated on each of the two resulting regions, and then on each region produced afterward, until a stopping criterion is reached, such as no region containing more than a set minimum number of observations. This top-down, greedy approach is called recursive binary splitting: greedy because each split is chosen to be the best one available at that step, without looking ahead to whether a different split would lead to a better tree further down.

## Tree size and cost-complexity pruning

A tree grown until every region is very small tends to overfit the training data and perform poorly on new observations. The preferred strategy is instead to grow one large tree $T_0$ and then prune it back to a smaller subtree. Cost-complexity pruning (also called weakest link pruning) considers a sequence of subtrees indexed by a nonnegative tuning parameter $\alpha$: for each value of $\alpha$, there is a subtree $T \subset T_0$ minimizing

```math
\Large \sum_{m=1}^{\vert T\vert}\sum_{i:\,x_i\in R_m}(y_i-\hat y_{R_m})^2 + \alpha\vert T\vert.
```

- $\vert T\vert$: the number of terminal nodes (leaves) of the subtree $T$.
- $R_m$: the region of predictor space corresponding to the $m$th terminal node.
- $\hat y_{R_m}$: the mean response for the training observations in $R_m$.
- $\alpha$: a nonnegative tuning parameter controlling the trade-off between the size of the tree and its fit to the training data.

When $\alpha=0$, the subtree $T$ is simply the full tree $T_0$, since the expression then reduces to the training sum of squares. As $\alpha$ increases, there is a price for having many terminal nodes, so the expression tends to be minimized by a smaller subtree. In practice, $\alpha$ is chosen using cross-validation: the large tree is grown, cost-complexity pruning produces a sequence of subtrees as a function of $\alpha$, K-fold cross-validation estimates the error for each value of $\alpha$, and the subtree corresponding to the value of $\alpha$ with the lowest cross-validated error is returned.

ESL develops the same construction in Section 9.2.2: the region-mean prediction is written with $f(x)=\sum_m c_mI(x\in R_m)$ and $\hat c_m=\mathrm{ave}(y_i\mid x_i\in R_m)$ (Equations 9.10–9.11), the half-plane split uses $\le$ and $>$ in place of $<$ and $\ge$ (Equation 9.12), the same greedy minimization chooses the split (Equations 9.13–9.14), and an equivalent cost-complexity criterion is built from the per-node observation count $N_m$ and squared-error impurity $Q_m(T)$ (Equations 9.15–9.16), penalized by the same $\alpha\vert T\vert$ term.

## Connection to this notebook

The [notebook](decision_tree_regression.ipynb) fits `DecisionTreeRegressor(random_state=0)` from scikit-learn to all 10 observations in `Position_Salaries.csv`, using `Level` to predict `Salary`. It does not set `ccp_alpha`, so the fitted tree is not cost-complexity pruned; scikit-learn instead limits its growth through its own default stopping rule rather than the pruning procedure described above. The notebook predicts the salary at level 6.5, then evaluates the fitted tree on a dense grid of level values to plot it. Because every region's prediction is a single constant, the fitted curve is a step function rather than a smooth one; the dense grid only reveals that shape more clearly; it does not add training observations or improve the fit.