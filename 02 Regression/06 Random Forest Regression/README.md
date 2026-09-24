# Random Forest Regression in Python

Random forest regression predicts a numerical response by averaging the predictions of many decorrelated regression trees, extending bagging with an extra source of randomness in how each tree is grown. ESL introduces random forests (Breiman, 2001) in Chapter 15, "Random Forests," and gives the regression and classification rules side by side in Algorithm 15.1, "Random Forest for Regression or Classification" (Section 15.2, pages 587–589). The equations below are the regression case only.

## Growing decorrelated trees

Each tree is grown on an independent bootstrap sample of the training data. Growing a tree from a bootstrap sample proceeds recursively: at each terminal node, a random sample of $m$ variables is drawn from the full set of $p$ variables, the best variable and split point are chosen among that sample of $m$, and the node is split into two daughter nodes. This continues until a minimum node size $n_{\min}$ is reached. Restricting each split to a random subset of the predictors is what decorrelates the trees; without it, a strong predictor would dominate the top split of most bootstrapped trees, and the trees would end up highly correlated with one another.

Section 15.3, "Details of Random Forests" (page 592), gives the inventors' recommended defaults for $m$ and the minimum node size, stated separately for the two settings this algorithm covers: for regression, the default value for $m$ is $\lfloor p/3\rfloor$ and the minimum node size is five; for classification, the default value for $m$ is $\lfloor\sqrt p\rfloor$ and the minimum node size is one. Because the regression and classification defaults differ, the recommended $m$ for a regression task is $\lfloor p/3\rfloor$, not $\lfloor\sqrt p\rfloor$.

## Prediction by averaging

After $B$ such trees $T_1,\ldots,T_B$ are grown, the random forest regression predictor at a point $x$ is

```math
\Large \hat f_{\mathrm{rf}}^B(x) = \frac{1}{B}\sum_{b=1}^{B}T_b(x).
```

- $B$: the number of trees grown.
- $T_b(x)$: the prediction of the $b$th random-forest tree at $x$.
- $\hat f_{\mathrm{rf}}^B(x)$: the random forest's predicted response at $x$, the average of the $B$ individual tree predictions.

ESL restates this same rule as Equation 15.2, $\hat f_{\mathrm{rf}}^B(x)=\frac{1}{B}\sum_{b=1}^{B}T(x;\Theta_b)$, using $\Theta_b$ to characterize the $b$th tree through its split variables, cutpoints at each node, and terminal-node values. The same algorithm's classification setting instead has each tree cast a vote for a predicted class, and the forest predicts by majority vote rather than by averaging.

Averaging reduces variance: if the $B$ trees were independent and identically distributed with variance $\sigma^2$, their average would have variance $\sigma^2/B$. Bootstrapped trees are identically distributed but not independent; if each pair has positive correlation $\rho$, the variance of their average is instead

```math
\Large \rho\sigma^2+\frac{1-\rho}{B}\sigma^2.
```

- $\rho$: the pairwise correlation between trees.
- $\sigma^2$: the variance of an individual tree.

As $B$ increases, the second term shrinks toward zero, but the first term, set by the correlation $\rho$, remains and limits how much averaging can reduce variance. Random forests improve on plain bagging by reducing $\rho$ through the random selection of split variables described above, without increasing $\sigma^2$ by very much.

## Connection to this notebook

The [notebook](random_forest_regression.ipynb) fits `RandomForestRegressor(n_estimators=10, random_state=0)` from scikit-learn to all 10 observations in `Position_Salaries.csv`, using `Level` to predict `Salary`. With a single predictor, $p=1$, so there is nothing to randomly subsample at each split; the averaging in the equations above is doing the work here, not the decorrelation step. The notebook predicts the salary at level 6.5, then evaluates the fitted forest on a dense grid of level values to plot it. Because each of the ten trees predicts its own piecewise-constant step function and the forest averages all ten, the plotted curve has more, smaller steps than the single tree in the decision-tree-regression notebook, rather than becoming a smooth curve.