# Machine-Learning-A-Z

A collection of Python notebooks and datasets for learning machine learning step by step, starting with data preprocessing and progressing to regression and classification.

## Topics and progress

| Topic | Contents | Status |
| --- | --- | --- |
| [Data preprocessing](01-DataPreprocessing/data_preprocessing.ipynb) | Missing values, categorical encoding, train/test splitting, and feature scaling | Notebook available |
| [Simple linear regression](02-RegressionModels/02-simple-linear-regression/simple_linear_regression.ipynb) | Predicting salary from experience and plotting training/test data | Notebook available |
| [Multiple linear regression](02-RegressionModels/03-multiple-linear-regression/multiple_linear_regression.ipynb) | Encoding state and predicting startup profit from multiple features | Notebook available |
| [Polynomial regression](02-RegressionModels/04-polynomial-regression/polynomial_regression.ipynb) | Comparing linear and degree-4 polynomial salary predictions | Notebook available |
| [Support vector regression (SVR)](02-RegressionModels/05-support-vector-regression/support_vector_regression.ipynb) | Scaling features and targets and fitting an RBF kernel model | Notebook available |
| [Decision tree regression](02-RegressionModels/06-decision-tree-regression/decision_tree_regression.ipynb) | Predicting position salaries with a decision tree | Notebook available |
| [Random forest regression](02-RegressionModels/07-random-forest-regression/random_forest_regression.ipynb) | Predicting position salaries with 10 decision trees | Notebook available |
| [Regression model selection](02-RegressionModels/08-regression-model-selection/) | Comparing five regression models using test-set R^2 | Notebooks available |
| [Additional model-selection notebooks](03-Regression%20Model%20Selection/) | The same five model types, with dataset placeholders in four notebooks | Dataset filenames need configuration |
| [Logistic regression](04-Classification/09-logistic-regression/logistic_regression.ipynb) | Predicting purchases and plotting decision boundaries | Notebook available |
| [K-nearest neighbors (K-NN)](04-Classification/10-K-NN/k_nearest_neighbors.ipynb) | Classifying purchases with five neighbors, a confusion matrix, and accuracy | Notebook available |

## Project structure

```text
Machine-Learning-A-Z/
|-- 01-DataPreprocessing/
|   |-- Data.csv
|   `-- data_preprocessing.ipynb
|-- 02-RegressionModels/
|   |-- 02-simple-linear-regression/
|   |-- 03-multiple-linear-regression/
|   |-- 04-polynomial-regression/
|   |-- 05-support-vector-regression/
|   |-- 06-decision-tree-regression/
|   |-- 07-random-forest-regression/
|   `-- 08-regression-model-selection/
|-- 03-Regression Model Selection/
|-- 04-Classification/
|   |-- 09-logistic-regression/
|   `-- 10-K-NN/
|-- requirements.txt
|-- LICENSE
`-- README.md
```

Each topic folder contains its notebooks and datasets. Data preprocessing is shared by regression and classification, so it stays at the repository root.

## Getting started

Use Python 3.10 or newer with the pinned dependencies. Dependency versions are pinned in `requirements.txt`.

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

Open a notebook from the topics table, select the Python kernel from your virtual environment, and run the cells from top to bottom. Start with data preprocessing, then work through simple linear regression, multiple linear regression, polynomial regression, support vector regression, decision tree regression, and random forest regression. Then explore regression model selection, logistic regression, and K-NN. Notebooks can be run independently; see the model-selection section for the dataset placeholders that need updating first.

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

The five notebooks in [02-RegressionModels/08-regression-model-selection](02-RegressionModels/08-regression-model-selection/) use the included `Data.csv` to compare multiple linear regression, degree-4 polynomial regression, RBF support vector regression, decision tree regression, and random forest regression.

Each notebook uses an 80/20 train/test split with `random_state=0`, prints predictions beside actual targets, and computes test-set R^2. The SVR notebook fits separate feature and target scalers on the training data and converts predictions back to the original target units.

[03-Regression Model Selection](03-Regression%20Model%20Selection/) contains another set of these notebooks and a copy of the dataset. Its SVR notebook already loads `Data.csv`; the other four use `ENTER_THE_NAME_OF_YOUR_DATASET_HERE.csv`. Replace that placeholder with `Data.csv` to use the included data, or with your own dataset filename. These examples treat all columns except the last as numeric features and the last column as the target.

## Classification

Both classification notebooks use `Social_Network_Ads.csv` to predict `Purchased` from `Age` and `EstimatedSalary`. They split the data into 75% training and 25% test sets with `random_state=0`, fit `StandardScaler` on training features only, and transform the test features with the same scaler.

- **Logistic regression:** Fits `LogisticRegression(random_state=0)`, predicts a purchase for age 30 and estimated salary 87,000, and plots training/test decision boundaries. The current notebook does not report accuracy or a confusion matrix.
- **K-NN:** Fits `KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)` (Euclidean distance), predicts the same example and the test set, reports a confusion matrix and accuracy, and plots training/test decision boundaries.

The decision-boundary plots create dense grids in the original age and salary units. If plotting is slow or uses too much memory, increase the grid step sizes, especially on the salary axis, or skip the plotting cells while exploring predictions and metrics.

## Datasets

All datasets are included in the repository; no separate download is needed. Polynomial regression, SVR, decision tree regression, and random forest regression each include a copy of `Position_Salaries.csv` in their topic folder.

| Dataset | Rows | Columns | Purpose |
| --- | --- | --- | --- |
| [Data.csv](01-DataPreprocessing/Data.csv) | 10 | `Country`, `Age`, `Salary`, `Purchased` | Practice handling missing values and categorical features; `Purchased` is the target |
| [Salary_Data.csv](02-RegressionModels/02-simple-linear-regression/Salary_Data.csv) | 30 | `YearsExperience`, `Salary` | Practice predicting salary from years of experience |
| [50_Startups.csv](02-RegressionModels/03-multiple-linear-regression/50_Startups.csv) | 50 | `R&D Spend`, `Administration`, `Marketing Spend`, `State`, `Profit` | Practice predicting profit from multiple numeric and categorical features |
| [Position_Salaries.csv](02-RegressionModels/04-polynomial-regression/Position_Salaries.csv) | 10 | `Position`, `Level`, `Salary` | Explore linear and polynomial salary prediction from position level |
| [Position_Salaries.csv (SVR)](02-RegressionModels/05-support-vector-regression/Position_Salaries.csv) | 10 | `Position`, `Level`, `Salary` | Practice feature and target scaling for support vector regression |
| [Position_Salaries.csv (decision tree)](02-RegressionModels/06-decision-tree-regression/Position_Salaries.csv) | 10 | `Position`, `Level`, `Salary` | Explore decision tree salary predictions |
| [Position_Salaries.csv (random forest)](02-RegressionModels/07-random-forest-regression/Position_Salaries.csv) | 10 | `Position`, `Level`, `Salary` | Explore salary predictions from an ensemble of decision trees |
| [Data.csv (regression model selection)](02-RegressionModels/08-regression-model-selection/Data.csv) | 9,568 | `AT`, `V`, `AP`, `RH`, `PE` | Compare regression models using the first four columns to predict `PE` |
| [Data.csv (additional model-selection notebooks)](03-Regression%20Model%20Selection/Data.csv) | 9,568 | `AT`, `V`, `AP`, `RH`, `PE` | Included dataset for the additional model-selection notebooks |
| [Social_Network_Ads.csv (logistic regression)](04-Classification/09-logistic-regression/Social_Network_Ads.csv) | 400 | `Age`, `EstimatedSalary`, `Purchased` | Predict purchases with logistic regression |
| [Social_Network_Ads.csv (K-NN)](04-Classification/10-K-NN/Social_Network_Ads.csv) | 400 | `Age`, `EstimatedSalary`, `Purchased` | Predict purchases with nearest neighbors |

`Data.csv` contains one missing age and one missing salary, which the preprocessing notebook fills using column means.

## License

[MIT](LICENSE)
