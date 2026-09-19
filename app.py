"""Interactive demos for the five regression models in this repo.

Run locally:
    python -m pip install -r requirements.txt
    python -m streamlit run app.py

Deploy for free on Streamlit Community Cloud (share.streamlit.io) by pointing
it at this file in your GitHub repo - no server to manage.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

st.set_page_config(page_title="Regression Model Demos", page_icon="📈")
BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def train_salary_model():
    data = pd.read_csv(BASE_DIR / "02-simple-linear-regression/Salary_Data.csv")
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    model = LinearRegression().fit(X_train, y_train)
    test_r2 = r2_score(y_test, model.predict(X_test))
    return model, test_r2, data


@st.cache_resource
def train_profit_model():
    dataset = pd.read_csv(BASE_DIR / "03-multiple-linear-regression/50_Startups.csv")
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    ct = ColumnTransformer(
        transformers=[("encoder", OneHotEncoder(handle_unknown="ignore"), [3])],
        remainder="passthrough",
    )
    model = LinearRegression().fit(ct.fit_transform(X_train), y_train)
    test_r2 = r2_score(y_test, model.predict(ct.transform(X_test)))
    states = sorted(dataset["State"].unique())
    return model, ct, test_r2, states


@st.cache_resource
def train_position_model(kind):
    folders = {
        "Polynomial Regression": "04-polynomial-regression",
        "Support Vector Regression": "05-support-vector-regression",
        "Decision Tree Regression": "06-decision-tree-regression",
    }
    data = pd.read_csv(BASE_DIR / folders[kind] / "Position_Salaries.csv")
    X = data[["Level"]].to_numpy()
    y = data["Salary"].to_numpy()
    if kind == "Polynomial Regression":
        model = make_pipeline(PolynomialFeatures(degree=4), LinearRegression())
    elif kind == "Support Vector Regression":
        model = TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(), SVR(kernel="rbf")),
            transformer=StandardScaler(),
        )
    else:
        model = DecisionTreeRegressor(random_state=0)
    model.fit(X, y)
    return model, data


def show_position_demo(kind):
    descriptions = {
        "Polynomial Regression": "Degree-4 polynomial features with linear regression.",
        "Support Vector Regression": "RBF support vector regression with scaled levels and salaries; predictions are converted back to dollars.",
        "Decision Tree Regression": "Decision tree regression with random_state=0; predictions are constant within each leaf.",
    }
    st.subheader(kind)
    st.caption(descriptions[kind])
    model, data = train_position_model(kind)
    st.info("Trained on all 10 positions, matching the notebook. Training R² measures fit to these examples, not performance on unseen data.")
    level = st.slider(
        "Position level", float(data["Level"].min()),
        float(data["Level"].max()), 6.5, 0.1, key=kind,
    )
    prediction = model.predict([[level]])[0]
    col1, col2 = st.columns(2)
    col1.metric("Predicted salary", f"${prediction:,.0f}")
    col2.metric("Training R²", f"{model.score(data[['Level']].to_numpy(), data['Salary']):.3f}")
    grid = np.linspace(data["Level"].min(), data["Level"].max(), 901).reshape(-1, 1)
    fig, ax = plt.subplots()
    ax.scatter(data["Level"], data["Salary"], label="Observed salaries")
    ax.plot(grid[:, 0], model.predict(grid), label="Fitted model")
    if kind == "Polynomial Regression":
        baseline = LinearRegression().fit(data[["Level"]].to_numpy(), data["Salary"])
        ax.plot(grid[:, 0], baseline.predict(grid), linestyle="--", label="Linear baseline")
    ax.scatter([level], [prediction], marker="*", s=160, label="Your prediction", zorder=3)
    ax.set(xlabel="Position level", ylabel="Salary ($)")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    with st.expander("View position salary data"):
        st.dataframe(data, hide_index=True)


st.title("📈 Regression Model Demos")
st.caption(
    "Live predictions from the five regression models in "
    "[Machine-Learning-A-Z](.) — trained on the same course datasets used in the notebooks."
)

position_models = ["Polynomial Regression", "Support Vector Regression", "Decision Tree Regression"]
tab1, tab2, *position_tabs = st.tabs(["Salary Predictor", "Startup Profit Predictor", *position_models])

with tab1:
    st.subheader("Predict salary from years of experience")
    st.caption("Simple linear regression, trained on `Salary_Data.csv`.")

    model, test_r2, data = train_salary_model()
    st.metric("Test-set R²", f"{test_r2:.3f}")

    years = st.slider("Years of experience", 0.0, 15.0, 5.0, 0.1)
    predicted_salary = model.predict([[years]])[0]
    st.metric("Predicted salary", f"${predicted_salary:,.0f}")

    chart_data = data.rename(columns={"YearsExperience": "x", "Salary": "y"})
    st.scatter_chart(chart_data, x="x", y="y", x_label="Years of Experience", y_label="Salary")

with tab2:
    st.subheader("Predict startup profit from spending and state")
    st.caption("Multiple linear regression, trained on `50_Startups.csv`.")

    model, ct, test_r2, states = train_profit_model()
    st.metric("Test-set R²", f"{test_r2:.3f}")

    col1, col2 = st.columns(2)
    with col1:
        rd_spend = st.number_input("R&D Spend ($)", min_value=0.0, value=100000.0, step=1000.0)
        admin_spend = st.number_input("Administration Spend ($)", min_value=0.0, value=100000.0, step=1000.0)
    with col2:
        marketing_spend = st.number_input("Marketing Spend ($)", min_value=0.0, value=100000.0, step=1000.0)
        state = st.selectbox("State", states)

    new_row = pd.DataFrame(
        [[rd_spend, admin_spend, marketing_spend, state]],
        columns=["R&D Spend", "Administration", "Marketing Spend", "State"],
    )
    encoded_row = ct.transform(new_row.values)
    predicted_profit = model.predict(encoded_row)[0]
    st.metric("Predicted profit", f"${predicted_profit:,.0f}")

    st.caption(
        "R&D Spend is by far the strongest driver of profit in this dataset "
        "(see the backward elimination section of the notebook) — try changing it "
        "versus the other two spending fields to see the difference in effect size."
    )

for tab, kind in zip(position_tabs, position_models):
    with tab:
        show_position_demo(kind)

st.divider()
st.caption(
    "Source: [Machine-Learning-A-Z on GitHub](.) · "
    "Course: Udemy's Machine Learning A-Z · Built with scikit-learn + Streamlit."
)
