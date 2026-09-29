# Machine-Learning-A-Z

A collection of Python notebooks and datasets for learning machine learning step by step, covering data preprocessing, regression, classification, model selection, clustering, and market basket analysis.

## Topics

| Topic | Contents |
| --- | --- |
| [Data preprocessing](01%20Data%20Preprocessing/) | Missing values, categorical encoding, train/test splitting, feature scaling |
| [Regression](02%20Regression/) | Simple linear, multiple linear, polynomial, support vector, decision tree, random forest |
| [Regression model selection](03%20Regression%20Model%20Selection/) | Five regression models evaluated with test-set R-squared |
| [Classification](04%20Classification/) | Logistic regression, K-NN, linear SVM, kernel SVM, naive Bayes, decision tree, random forest |
| [Classification model selection](05%20Classification%20Model%20Selection/) | Seven classifier templates in notebook and Python script form |
| [Clustering](06%20Clustering/) | K-means, hierarchical clustering |
| [Apriori](07%20Apriori/) | Association rules, support, confidence, lift |
| [Eclat](Eclat/) | Product associations summarized by support; currently uses Apriori |

## Project structure

```text
Machine-Learning-A-Z/
|-- 01 Data Preprocessing/
|-- 02 Regression/
|   |-- 01 Simple Linear Regreesion/
|   |-- 02 Multiple Linear Regression/
|   |-- 03 Polynomial Regression/
|   |-- 04 Support Vector Regression/
|   |-- 05 Decision Tree Regression/
|   `-- 06 Random Forest Regression/
|-- 03 Regression Model Selection/
|-- 04 Classification/
|   |-- 01 Logistic Regression/
|   |-- 02 K-Nearest Neighbors/
|   |-- 03 Support Vector Machine/
|   |-- 04 Kernel SVM/
|   |-- 05 Naive Bayes/
|   |-- 06 Decision Tree/
|   `-- 07 Random Forest/
|-- 05 Classification Model Selection/
|-- 06 Clustering/
|   |-- 01 K-Means Clusterting/
|   `-- 02 Hierarchical Clustering/
|-- 07 Apriori/
|-- Eclat/
|-- requirements.txt
|-- LICENSE
`-- README.md
```

Numbered folders indicate the learning order; `Eclat/` is currently unnumbered. The tree above preserves the existing folder names. Each topic keeps its notebooks and datasets together with their existing filenames, so relative CSV paths stay the same. Topic explanations live in `README.md`, and supporting images live in `figures/`.

Regression model selection contains five notebooks and its own `Data.csv`. Classification model selection contains seven notebooks, seven Python scripts, and a separate `Data.csv`.

## Getting started

Use Python 3.10 or newer. Core dependency versions are pinned in `requirements.txt`.

From the repository root, create a virtual environment:

```sh
python -m venv .venv
```

Activate it using the command for your shell:

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```sh
source .venv/bin/activate
```

Install the dependencies and launch JupyterLab:

```sh
python -m pip install -r requirements.txt
python -m jupyterlab
```

The Apriori and Eclat notebooks also import `apyori`, which is not included in `requirements.txt`. Install it in the same environment before running those notebooks:

```sh
python -m pip install apyori
```

Open a topic folder from the table above, choose a notebook, select the Python kernel from your virtual environment, and run the cells from top to bottom. Follow preprocessing with regression and classification, then explore model selection, clustering, and market basket analysis. See the notes below for templates and cells that need adjustment before a full run.

The CSV paths are relative to each notebook's directory. Keep the notebook beside its dataset and ensure its working directory is that topic folder if you encounter a `FileNotFoundError`.

## Data preprocessing

The first notebook demonstrates how to:

1. Load a CSV with pandas and separate features (`X`) from the target (`y`).
2. Fill missing numeric values with `SimpleImputer` using the column mean.
3. One-hot encode country names with `ColumnTransformer` and `OneHotEncoder`.
4. Encode the purchase target with `LabelEncoder`.
5. Split the data into 80% training and 20% test sets with `random_state=1`.
6. Standardize numeric features with `StandardScaler`, fitting the scaler on the training set and applying it to the test set.

**Learning note:** The current example fits the imputer and categorical encoder before splitting the data. For model evaluation, split first and fit preprocessing on the training data only to avoid leaking information from the test set. A scikit-learn pipeline can help enforce this separation. This guidance also applies to the multiple linear regression notebook, which fits its startup categorical encoder before splitting.

## Simple linear regression

This notebook uses `Salary_Data.csv` to predict `Salary` from `YearsExperience`:

1. Split the data into two-thirds training and one-third test data with `random_state=0`.
2. Fit scikit-learn's `LinearRegression` model on the training set.
3. Predict salaries for the test set.
4. Plot the training and test observations alongside the fitted regression line.

The current notebook does not report evaluation metrics or export the trained model.

## Multiple linear regression

This notebook uses `50_Startups.csv` to predict `Profit` from `R&D Spend`, `Administration`, `Marketing Spend`, and `State`:

1. One-hot encode `State` using `ColumnTransformer` and `OneHotEncoder`, retaining the numeric spending features.
2. Split the data into 80% training and 20% test sets with `random_state=0`.
3. Fit a `LinearRegression` model on the training set.
4. Print test-set predictions beside the actual profits.

The current notebook does not include significance tests, backward elimination, or model export.

## Polynomial regression

The fourth notebook uses `Position_Salaries.csv` to explore predicting `Salary` from `Level`. The descriptive `Position` column is excluded from the model inputs.

1. Load the dataset and separate position level (`X`) from salary (`y`).
2. Fit a baseline `LinearRegression` model on the whole dataset.
3. Generate degree-4 polynomial features with `PolynomialFeatures` and fit a second `LinearRegression` model to those features.
4. Visualize both models, including a smoother polynomial curve using a grid with step size `0.1`.
5. Predict the salary for position level `6.5` with both models.

The notebook uses the whole dataset without a train/test split, and doesn't yet report evaluation metrics.

## Support vector regression (SVR)

The fifth notebook uses its own copy of `Position_Salaries.csv` to predict `Salary` from `Level` with support vector regression:

1. Load the dataset and reshape the salary target for scaling.
2. Standardize position levels and salaries with separate `StandardScaler` instances.
3. Fit an `SVR` model with an RBF kernel on the whole scaled dataset.
4. Predict the salary for position level `6.5`, scaling the input and inverse-transforming the prediction to the original salary units.
5. Plot the fitted model in the original units, including a smoother curve using a position-level grid with step size `0.1`.

Like the polynomial example, this notebook uses the whole dataset without a train/test split; the plots illustrate the fit rather than performance on unseen data.

## Decision tree regression

The sixth notebook uses `Position_Salaries.csv` to predict salary from position level:

1. Use `Level` as the feature and `Salary` as the target, excluding the descriptive position name.
2. Fit `DecisionTreeRegressor(random_state=0)` on all 10 rows.
3. Predict the salary at level `6.5`.
4. Plot predictions on a dense grid to show the constant predictions within each tree leaf.

A perfect training fit does not establish accuracy on unseen positions; this example has no held-out test set.

## Random forest regression

The seventh notebook uses `Position_Salaries.csv` to predict salary from position level with an ensemble of decision trees:

1. Use `Level` as the feature and `Salary` as the target, excluding the descriptive position name.
2. Fit `RandomForestRegressor(n_estimators=10, random_state=0)` on all 10 rows.
3. Predict the salary at level `6.5`.
4. Plot predictions on a grid with step size `0.01` to visualize the ensemble's piecewise-constant fitted curve.

Like the other position-level notebooks, this example trains on the whole dataset without a held-out test set. It does not yet report evaluation metrics.

## Regression model selection

The five notebooks in [03 Regression Model Selection](03%20Regression%20Model%20Selection/) use the included `Data.csv` to compare multiple linear regression, degree-4 polynomial regression, RBF support vector regression, decision tree regression, and random forest regression.

Each notebook uses an 80/20 train/test split with `random_state=0`, prints predictions beside actual targets, and computes test-set R^2. The SVR notebook fits separate feature and target scalers on the training data and converts predictions back to the original target units.


## Classification

The seven classification notebooks use `Social_Network_Ads.csv` to predict `Purchased` from `Age` and `EstimatedSalary`. They split the data into 75% training and 25% test sets with `random_state=0`, fit `StandardScaler` on training features only, and transform the test features with the same scaler.

- **Logistic regression:** Fits `LogisticRegression(random_state=0)`, predicts a purchase for age 30 and estimated salary 87,000, and plots training/test decision boundaries. The current notebook does not report accuracy or a confusion matrix.
- **K-NN:** Fits `KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)` (Euclidean distance), predicts the same example and the test set, reports a confusion matrix and accuracy, and plots training/test decision boundaries.

- **Linear SVM and kernel SVM:** Fit `SVC` with linear and RBF kernels, respectively, and report confusion matrices and accuracy.
- **Naive Bayes:** Fits `GaussianNB` and reports a confusion matrix and accuracy.
- **Decision tree:** Fits `DecisionTreeClassifier` with the entropy criterion and `random_state=0`.
- **Random forest:** Fits 10 trees with the entropy criterion and `random_state=0`.

The tree and forest examples also report confusion matrices and accuracy. All seven notebooks include training/test decision-boundary plots.

The decision-boundary plots create dense grids in the original age and salary units. If plotting is slow or uses too much memory, increase the grid step sizes, especially on the salary axis, or skip the plotting cells while exploring predictions and metrics.

## Classification model selection

[05 Classification Model Selection](05%20Classification%20Model%20Selection/) provides notebook and `.py` versions of logistic regression, K-NN, linear SVM, kernel SVM, naive Bayes, decision tree, and random forest classifiers. The templates use a 75/25 split with `random_state=0`, fit feature scaling on the training set, and compute a confusion matrix and test accuracy.

These files currently load `ENTER_THE_NAME_OF_YOUR_DATASET_HERE.csv`. Replace the placeholder with `Data.csv` to use the included table, or supply your own dataset. They take all columns except the last as features and the last as the target. The included table contains a sample identifier column; review whether to exclude it when adapting the templates. Run scripts from their folder so relative dataset paths resolve.

## Clustering

Both clustering notebooks use `Mall_Customers.csv`, selecting annual income and spending score as inputs:

- [K-means](06%20Clustering/01%20K-Means%20Clusterting/k_means_clustering.ipynb) plots within-cluster sums of squares for 1 through 10 clusters, then fits five clusters with k-means++ initialization and plots the centroids.
- [Hierarchical clustering](06%20Clustering/02%20Hierarchical%20Clustering/hierarchical_clustering.ipynb) builds a Ward-linkage dendrogram, then fits five agglomerative clusters and plots the groups.

The hierarchical notebook currently passes `affinity='euclidean'` to `AgglomerativeClustering`. The pinned scikit-learn 1.5.2 API uses `metric='euclidean'`, so this cell needs that parameter-name update before it can run with the pinned environment.

## Market basket analysis

[Apriori](07%20Apriori/apriori.ipynb) and [Eclat](Eclat/eclat.ipynb) each include a copy of `Market_Basket_Optimisation.csv`, containing 7,501 transactions with up to 20 item slots per row and no header. Both notebooks currently call `apyori.apriori` with minimum support 0.003, confidence 0.2, lift 3, and itemset length 2.

The Apriori notebook creates a table of associations with support, confidence, and lift, and selects the ten highest-lift entries. Its standalone `resultsDataFrame` cell refers to an undefined name; the table is actually stored in `resultsinDataFrame`.

The Eclat notebook displays product associations and support. Despite the folder name, its current implementation uses Apriori rather than a separate Eclat algorithm. Both examples convert all 20 slots to strings, including missing slots, so empty entries should be filtered when adapting transaction preparation.

## Datasets

All datasets are included in the repository; no separate download is needed. Each of the seven classification topic folders has its own copy of `Social_Network_Ads.csv`. Polynomial regression, SVR, decision tree regression, and random forest regression each include a copy of `Position_Salaries.csv` in their topic folder.

| Dataset | Rows | Columns | Purpose |
| --- | --- | --- | --- |
| [Data.csv](01%20Data%20Preprocessing/Data.csv) | 10 | `Country`, `Age`, `Salary`, `Purchased` | Practice handling missing values and categorical features; `Purchased` is the target |
| [Salary_Data.csv](02%20Regression/01%20Simple%20Linear%20Regreesion/Salary_Data.csv) | 30 | `YearsExperience`, `Salary` | Practice predicting salary from years of experience |
| [50_Startups.csv](02%20Regression/02%20Multiple%20Linear%20Regression/50_Startups.csv) | 50 | `R&D Spend`, `Administration`, `Marketing Spend`, `State`, `Profit` | Practice predicting profit from multiple numeric and categorical features |
| [Position_Salaries.csv](02%20Regression/03%20Polynomial%20Regression/Position_Salaries.csv) | 10 | `Position`, `Level`, `Salary` | Explore linear and polynomial salary prediction from position level |
| [Position_Salaries.csv (SVR)](02%20Regression/04%20Support%20Vector%20Regression/Position_Salaries.csv) | 10 | `Position`, `Level`, `Salary` | Practice feature and target scaling for support vector regression |
| [Position_Salaries.csv (decision tree)](02%20Regression/05%20Decision%20Tree%20Regression/Position_Salaries.csv) | 10 | `Position`, `Level`, `Salary` | Explore decision tree salary predictions |
| [Position_Salaries.csv (random forest)](02%20Regression/06%20Random%20Forest%20Regression/Position_Salaries.csv) | 10 | `Position`, `Level`, `Salary` | Explore salary predictions from an ensemble of decision trees |
| [Data.csv (regression model selection)](03%20Regression%20Model%20Selection/Data.csv) | 9,568 | `AT`, `V`, `AP`, `RH`, `PE` | Compare regression models using the first four columns to predict `PE` |
| [Social_Network_Ads.csv (logistic regression)](04%20Classification/01%20Logistic%20Regression/Social_Network_Ads.csv) | 400 | `Age`, `EstimatedSalary`, `Purchased` | Predict purchases with logistic regression |
| [Social_Network_Ads.csv (K-NN)](04%20Classification/02%20K-Nearest%20Neighbors/Social_Network_Ads.csv) | 400 | `Age`, `EstimatedSalary`, `Purchased` | Predict purchases with nearest neighbors |
| [Data.csv (classification model selection)](05%20Classification%20Model%20Selection/Data.csv) | 683 | Sample identifier, nine cell-measurement features, `Class` | Compare classifier templates |
| [Mall_Customers.csv (K-means)](06%20Clustering/01%20K-Means%20Clusterting/Mall_Customers.csv) | 200 | Customer ID, genre, age, annual income, spending score | Group customers with K-means |
| [Mall_Customers.csv (hierarchical)](06%20Clustering/02%20Hierarchical%20Clustering/Mall_Customers.csv) | 200 | Customer ID, genre, age, annual income, spending score | Group customers with hierarchical clustering |
| [Market_Basket_Optimisation.csv (Apriori)](07%20Apriori/Market_Basket_Optimisation.csv) | 7,501 | Up to 20 item slots; no header | Mine product associations |
| [Market_Basket_Optimisation.csv (Eclat folder)](Eclat/Market_Basket_Optimisation.csv) | 7,501 | Up to 20 item slots; no header | Summarize product associations by support |

The preprocessing `Data.csv` contains one missing age and one missing salary, which the preprocessing notebook fills using column means.

## License

[MIT](LICENSE)
