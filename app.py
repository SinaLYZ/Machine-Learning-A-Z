"""Interactive demo for the two linear regression models in this repo.

Run locally:
    pip install -r requirements.txt streamlit
    streamlit run app.py

Deploy for free on Streamlit Community Cloud (share.streamlit.io) by pointing
it at this file in your GitHub repo - no server to manage.
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="Linear Regression Demo", page_icon="📈")


@st.cache_resource
def train_salary_model():
    data = pd.read_csv("02-simple-linear-regression/Salary_Data.csv")
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
    dataset = pd.read_csv("03-multiple-linear-regression/50_Startups.csv")
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    ct = ColumnTransformer(
        transformers=[("encoder", OneHotEncoder(), [3])], remainder="passthrough"
    )
    X_encoded = np.array(ct.fit_transform(X))

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=0
    )
    model = LinearRegression().fit(X_train, y_train)
    test_r2 = r2_score(y_test, model.predict(X_test))
    states = sorted(dataset["State"].unique())
    return model, ct, test_r2, states


st.title("📈 Linear Regression Demo")
st.caption(
    "Live predictions from the two linear regression models in "
    "[Machine-Learning-A-Z](.) — trained on the same course datasets used in the notebooks."
)

tab1, tab2 = st.tabs(["Salary Predictor", "Startup Profit Predictor"])

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

st.divider()
st.caption(
    "Source: [Machine-Learning-A-Z on GitHub](.) · "
    "Course: Udemy's Machine Learning A-Z · Built with scikit-learn + Streamlit."
)
