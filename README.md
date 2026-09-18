# Machine-Learning-A-Z

A collection of Python notebooks and datasets for learning machine learning step by step, starting with data preprocessing and progressing to regression.

## Topics and progress

| Topic | Contents | Status |
| --- | --- | --- |
| [Data preprocessing](01-data-preprocessing/data_preprocessing.ipynb) | Missing values, categorical encoding, train/test splitting, and feature scaling | Notebook available |
| [Simple linear regression](02-simple-linear-regression/simple-linear-regression.ipynb) | Predicting salary from experience and visualizing the fitted regression line | Notebook available |
| [Multiple linear regression](03-multiple-linear-regression/multiple_linear_regression.ipynb) | Predicting startup profit from spending and state, with categorical encoding | Notebook available |
| [Polynomial regression](04-polynomial-regression/polynomial_regression.ipynb) | Comparing linear and degree-4 polynomial salary predictions and visualizing fitted curves | Notebook available |
| [Support vector regression (SVR)](05-support-vector-regression/support_vector_regression.ipynb) | Scaling features and targets, fitting an RBF kernel model, and predicting salaries | Notebook available |

## Project structure

```text
Machine-Learning-A-Z/
├── 01-data-preprocessing/
│   ├── Data.csv
│   └── data_preprocessing.ipynb
├── 02-simple-linear-regression/
│   ├── Salary_Data.csv
│   └── simple-linear-regression.ipynb
├── 03-multiple-linear-regression/
│   ├── 50_Startups.csv
│   └── multiple_linear_regression.ipynb
├── 04-polynomial-regression/
│   ├── Position_Salaries.csv
│   └── polynomial_regression.ipynb
├── 05-support-vector-regression/
│   ├── Position_Salaries.csv
│   └── support_vector_regression.ipynb
└── README.md
```

## Getting started

Use Python 3 and JupyterLab. The preprocessing notebook records Python 3.13.1 in its metadata; dependency versions are not currently pinned.

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

Install the notebook dependencies and launch JupyterLab:

```sh
python -m pip install numpy pandas matplotlib scikit-learn jupyterlab
python -m jupyterlab
```

Open a notebook from the topics table, select the Python kernel from your virtual environment, and run the cells from top to bottom. Start with data preprocessing, then work through simple linear regression, multiple linear regression, polynomial regression, and support vector regression. Each notebook loads its own dataset and can also be run independently.

The CSV paths are relative to each notebook's directory. Keep the notebook beside its dataset and ensure its working directory is that topic folder if you encounter a `FileNotFoundError`.

## Data preprocessing

The first notebook demonstrates how to:

1. Load a CSV with pandas and separate features (`X`) from the target (`y`).
2. Fill missing numeric values with `SimpleImputer` using the column mean.
3. One-hot encode country names with `ColumnTransformer` and `OneHotEncoder`.
4. Encode the purchase target with `LabelEncoder`.
5. Split the data into 80% training and 20% test sets with `random_state=1`.
6. Standardize numeric features with `StandardScaler`, fitting the scaler on the training set and applying it to the test set.

**Learning note:** The current example fits the imputer and categorical encoder before splitting the data. For model evaluation, split first and fit preprocessing on the training data only to avoid leaking information from the test set. A scikit-learn pipeline can help enforce this separation.

## Simple linear regression

The second notebook uses `Salary_Data.csv` to predict `Salary` from `YearsExperience`:

1. Split the data into 80% training and 20% test sets with `random_state=0`.
2. Fit scikit-learn's `LinearRegression` model on the training set.
3. Predict salaries for the test set.
4. Plot the training and test observations alongside the fitted regression line using Matplotlib.

## Multiple linear regression

The third notebook uses `50_Startups.csv` to predict `Profit` from `R&D Spend`, `Administration`, `Marketing Spend`, and `State`:

1. One-hot encode `State` using `ColumnTransformer` and `OneHotEncoder`, retaining the numeric spending features.
2. Split the data into 80% training and 20% test sets with `random_state=0`.
3. Fit a `LinearRegression` model and display predicted and actual test profits side by side.
4. Predict profit for a single manually encoded example.
5. Display the model's coefficients and intercept.

The single-example prediction supplies the encoded state columns first, followed by the three spending values. New inputs must follow the same feature order as the training data. This notebook also fits its categorical encoder before the train/test split; the preprocessing guidance above applies here as well.

The regression notebooks demonstrate fitting and prediction; they do not yet calculate evaluation metrics such as R² or mean squared error.

## Polynomial regression

The fourth notebook uses `Position_Salaries.csv` to explore predicting `Salary` from `Level`. The descriptive `Position` column is excluded from the model inputs.

1. Load the dataset and separate position level (`X`) from salary (`y`).
2. Fit a baseline `LinearRegression` model on the whole dataset.
3. Generate degree-4 polynomial features with `PolynomialFeatures` and fit a second `LinearRegression` model to those features.
4. Visualize both models, including a smoother polynomial curve using a grid with step size `0.1`.
5. Predict the salary for position level `6.5` with both models.

The notebook uses the whole dataset without a train/test split.

## Support vector regression (SVR)

The fifth notebook uses its own copy of `Position_Salaries.csv` to predict `Salary` from `Level` with support vector regression:

1. Load the dataset and reshape the salary target for scaling.
2. Standardize position levels and salaries with separate `StandardScaler` instances.
3. Fit an `SVR` model with an RBF kernel on the whole scaled dataset.
4. Predict the salary for position level `6.5`, scaling the input and inverse-transforming the prediction to the original salary units.
5. Plot the fitted model in the original units, including a smoother curve using a position-level grid with step size `0.1`.

Like the polynomial example, this notebook uses the whole dataset without a train/test split; the plots illustrate the fit rather than performance on unseen data.

## Datasets

All datasets are included in the repository; no separate download is needed. Polynomial regression and SVR each include a copy of `Position_Salaries.csv` in their topic folder.

| Dataset | Rows | Columns | Purpose |
| --- | --- | --- | --- |
| [Data.csv](01-data-preprocessing/Data.csv) | 10 | `Country`, `Age`, `Salary`, `Purchased` | Practice handling missing values and categorical features; `Purchased` is the target |
| [Salary_Data.csv](02-simple-linear-regression/Salary_Data.csv) | 30 | `YearsExperience`, `Salary` | Practice predicting salary from years of experience |
| [50_Startups.csv](03-multiple-linear-regression/50_Startups.csv) | 50 | `R&D Spend`, `Administration`, `Marketing Spend`, `State`, `Profit` | Practice predicting profit from multiple numeric and categorical features |
| [Position_Salaries.csv](04-polynomial-regression/Position_Salaries.csv) | 10 | `Position`, `Level`, `Salary` | Explore linear and polynomial salary prediction from position level |
| [Position_Salaries.csv (SVR)](05-support-vector-regression/Position_Salaries.csv) | 10 | `Position`, `Level`, `Salary` | Practice feature and target scaling for support vector regression |

`Data.csv` contains one missing age and one missing salary, which the preprocessing notebook fills using column means.
