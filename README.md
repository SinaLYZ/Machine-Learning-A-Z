# Machine-Learning-A-Z

A collection of Python notebooks and datasets for learning machine learning step by step, starting with data preprocessing and progressing to regression.

## Interactive demo

`app.py` is a Streamlit app with five interactive regression demos. Models train directly from the included CSVs; running the notebooks or generating saved model files first is not required.

| Tab | Model | Inputs and features |
| --- | --- | --- |
| Salary Predictor | Simple linear regression | Years of experience, salary prediction, test-set R², and observed data |
| Startup Profit Predictor | Multiple linear regression | Spending and state, profit prediction, and test-set R² |
| Polynomial Regression | Degree-4 polynomial regression | Position level, salary prediction, fitted curve, and linear baseline |
| Support Vector Regression | RBF SVR | Position level, feature and target scaling, salary prediction in dollars, and fitted curve |
| Decision Tree Regression | Decision tree | Position level, salary prediction, and piecewise-constant fitted curve |

With your virtual environment already activated, run these commands from the repository root:

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed in the terminal (usually `http://localhost:8501`). Launch with Streamlit: running `python app.py` produces missing `ScriptRunContext` warnings and does not start the web interface. Stop the server with `Ctrl+C`.

The first two demos report R² on a held-out 20% test set. The three position-level demos match their notebooks by training on all 10 rows and display **training R²**, not a test-set score. Each includes a position-level slider, fitted curve, selected prediction, and expandable source data. These small course datasets are learning examples.

See [DEMO.md](DEMO.md) for more details.

<!-- Once deployed: **Live demo:** https://<your-app>.streamlit.app -->

## Topics and progress

| Topic | Contents | Status |
| --- | --- | --- |
| [Data preprocessing](01-data-preprocessing/data_preprocessing.ipynb) | Missing values, categorical encoding, train/test splitting, and feature scaling | Notebook available |
| [Simple linear regression](02-simple-linear-regression/simple-linear-regression.ipynb) | Predicting salary from experience, with test-set R²/MAE/RMSE, the fitted equation, and a residual plot | Notebook available |
| [Multiple linear regression](03-multiple-linear-regression/multiple_linear_regression.ipynb) | Predicting startup profit from spending and state, with EDA, test-set metrics, a statsmodels significance summary, a VIF multicollinearity check, and backward elimination | Notebook available |
| [Polynomial regression](04-polynomial-regression/polynomial_regression.ipynb) | Comparing linear and degree-4 polynomial salary predictions and visualizing fitted curves | Notebook available |
| [Support vector regression (SVR)](05-support-vector-regression/support_vector_regression.ipynb) | Scaling features and targets, fitting an RBF kernel model, and predicting salaries | Notebook available |
| [Decision tree regression](06-decision-tree-regression/decision_tree_regression.ipynb) | Fitting a decision tree to position salaries and visualizing piecewise-constant predictions | Notebook available |

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
├── 06-decision-tree-regression/
│   ├── Position_Salaries.csv
│   └── decision_tree_regression.ipynb
├── models/                 # trained models saved by the notebooks (gitignored)
├── app.py                  # Streamlit demo — see DEMO.md
├── DEMO.md
├── requirements.txt
├── LICENSE
└── README.md
```

## Getting started

Requires Python 3. Dependency versions are pinned in `requirements.txt`.

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

Open a notebook from the topics table, select the Python kernel from your virtual environment, and run the cells from top to bottom. Start with data preprocessing, then work through simple linear regression, multiple linear regression, polynomial regression, support vector regression, and decision tree regression. Each notebook loads its own dataset and can also be run independently.

The CSV paths are relative to each notebook's directory. Keep the notebook beside its dataset and ensure its working directory is that topic folder if you encounter a `FileNotFoundError`.

## Data preprocessing

The first notebook demonstrates how to:

1. Load a CSV with pandas and separate features (`X`) from the target (`y`).
2. Fill missing numeric values with `SimpleImputer` using the column mean.
3. One-hot encode country names with `ColumnTransformer` and `OneHotEncoder`.
4. Encode the purchase target with `LabelEncoder`.
5. Split the data into 80% training and 20% test sets with `random_state=1`.
6. Standardize numeric features with `StandardScaler`, fitting the scaler on the training set and applying it to the test set.

**Learning note:** The current example fits the imputer and categorical encoder before splitting the data. For model evaluation, split first and fit preprocessing on the training data only to avoid leaking information from the test set. A scikit-learn pipeline can help enforce this separation. This still applies here and to the multiple linear regression notebook's categorical encoding. The app fits the startup encoder on its training split only.

## Simple linear regression

The second notebook uses `Salary_Data.csv` to predict `Salary` from `YearsExperience`:

1. Split the data into 80% training and 20% test sets with `random_state=0`.
2. Fit scikit-learn's `LinearRegression` model on the training set.
3. Predict salaries for the test set.
4. Report R², MAE, and RMSE on the test set, and print the fitted equation (`Salary = intercept + slope * YearsExperience`).
5. Plot the training and test observations alongside the fitted regression line, plus a residual plot to sanity-check the linear fit.
6. Save the trained model to `models/simple_linear_regression.joblib` for reuse outside the notebook. The Streamlit app trains directly from the CSV data.

## Multiple linear regression

The third notebook uses `50_Startups.csv` to predict `Profit` from `R&D Spend`, `Administration`, `Marketing Spend`, and `State`:

1. Explore correlations between the numeric features and profit before modeling.
2. One-hot encode `State` using `ColumnTransformer` and `OneHotEncoder`, retaining the numeric spending features.
3. Split the data into 80% training and 20% test sets with `random_state=0`.
4. Fit a `LinearRegression` model, report test-set R²/MAE/RMSE, and plot actual vs. predicted profit.
5. Refit with `statsmodels.OLS` to inspect each coefficient's p-value and confidence interval — scikit-learn's `LinearRegression` doesn't expose these.
6. Check for multicollinearity between the spending features with Variance Inflation Factor (VIF).
7. Run backward elimination to find which features remain statistically significant once the others are accounted for (R&D Spend, in this dataset).
8. Save the trained model and fitted encoder to `models/` for reuse outside the notebook. The Streamlit app trains directly from the CSV data.

The single-example prediction supplies the encoded state columns first, followed by the three spending values. New inputs must follow the same feature order as the training data.

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

The app reproduces this model and lets you explore fractional levels. A perfect training fit does not establish accuracy on unseen positions; this example has no held-out test set.

## Datasets

All datasets are included in the repository; no separate download is needed. Polynomial regression, SVR, and decision tree regression each include a copy of `Position_Salaries.csv` in their topic folder.

| Dataset | Rows | Columns | Purpose |
| --- | --- | --- | --- |
| [Data.csv](01-data-preprocessing/Data.csv) | 10 | `Country`, `Age`, `Salary`, `Purchased` | Practice handling missing values and categorical features; `Purchased` is the target |
| [Salary_Data.csv](02-simple-linear-regression/Salary_Data.csv) | 30 | `YearsExperience`, `Salary` | Practice predicting salary from years of experience |
| [50_Startups.csv](03-multiple-linear-regression/50_Startups.csv) | 50 | `R&D Spend`, `Administration`, `Marketing Spend`, `State`, `Profit` | Practice predicting profit from multiple numeric and categorical features |
| [Position_Salaries.csv](04-polynomial-regression/Position_Salaries.csv) | 10 | `Position`, `Level`, `Salary` | Explore linear and polynomial salary prediction from position level |
| [Position_Salaries.csv (SVR)](05-support-vector-regression/Position_Salaries.csv) | 10 | `Position`, `Level`, `Salary` | Practice feature and target scaling for support vector regression |
| [Position_Salaries.csv (decision tree)](06-decision-tree-regression/Position_Salaries.csv) | 10 | `Position`, `Level`, `Salary` | Explore decision tree salary predictions |

`Data.csv` contains one missing age and one missing salary, which the preprocessing notebook fills using column means.

## License

[MIT](LICENSE)
