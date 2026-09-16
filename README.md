# Machine-Learning-A-Z

A collection of Python notebooks and datasets for learning machine learning step by step, starting with data preprocessing and progressing to regression.

## Topics and progress

| Topic | Contents | Status |
| --- | --- | --- |
| [Data preprocessing](01-data-preprocessing/data_preprocessing.ipynb) | Missing values, categorical encoding, train/test splitting, and feature scaling | Notebook available |
| [Simple linear regression](02-simple-linear-regression/) | Salary dataset for exploring the relationship between experience and salary | Dataset available; notebook is an empty placeholder |
| Multiple linear regression | Reserved folder for the next topic | Planned |

## Project structure

```text
Machine-Learning-A-Z/
├── 01-data-preprocessing/
│   ├── Data.csv
│   └── data_preprocessing.ipynb
├── 02-simple-linear-regression/
│   ├── Salary_Data.csv
│   └── simple-linear-regression.ipynb
└── README.md
```

The local `03-multiple-linear-regression/` folder is currently empty. Git does not track empty folders, so it may not appear in a fresh clone.

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

Open `01-data-preprocessing/data_preprocessing.ipynb`, select the Python kernel from your virtual environment, and run the cells from top to bottom. The notebook displays the feature and target arrays before and after preprocessing.

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

## Datasets

Both datasets are included in the repository; no separate download is needed.

| Dataset | Rows | Columns | Purpose |
| --- | --- | --- | --- |
| [Data.csv](01-data-preprocessing/Data.csv) | 10 | `Country`, `Age`, `Salary`, `Purchased` | Practice handling missing values and categorical features; `Purchased` is the target |
| [Salary_Data.csv](02-simple-linear-regression/Salary_Data.csv) | 30 | `YearsExperience`, `Salary` | Practice predicting salary from years of experience |

`Data.csv` contains one missing age and one missing salary. The salary regression dataset is ready to use, but the regression implementation and results have not yet been added.
