# Interactive demo

`app.py` provides five tabs: simple linear regression for salary, multiple linear regression for startup profit, polynomial regression, support vector regression, and decision tree regression.

## Run locally

From the repository root, with your virtual environment activated:

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the URL printed in the terminal (usually `http://localhost:8501`). Use `Ctrl+C` to stop the server. Do not launch with `python app.py`; the interface and session state require the Streamlit runtime.

The app trains and caches its models from the included CSVs. Saved models and prior notebook execution are not required. Dataset paths are resolved relative to `app.py`.

## Explore the models

- **Salary Predictor:** change years of experience to predict salary.
- **Startup Profit Predictor:** change spending and state to predict profit. The categorical encoder is fitted only on training data.
- **Polynomial Regression:** change position level to compare a degree-4 curve with a linear baseline.
- **Support Vector Regression:** explore an RBF model with scaled inputs and targets; displayed predictions use original salary units.
- **Decision Tree Regression:** explore the constant salary predictions within each tree leaf.

The salary and startup models report held-out test-set R². Position-level models use all 10 rows, matching their notebooks, and report training R² only. Their charts illustrate fit, not generalization performance.

## Hosting configuration

For a Streamlit hosting service, use `app.py` as the entry point and install `requirements.txt`, which includes Streamlit. Include all five regression dataset folders with the app.
