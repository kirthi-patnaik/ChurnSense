
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(
    page_title="ChurnSense",
    page_icon="📊",
    layout="wide"
)

# ---------------- PATHS ----------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "churn_model.pkl"
DATA_PATH = BASE_DIR / "sample_customers.csv"

# ---------------- LOAD FILES ----------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

model = load_model()
df = load_data()

# ---------------- SIDEBAR ----------------

st.sidebar.title("📊 ChurnSense")
st.sidebar.caption("AI-Powered Customer Churn Analytics")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🔮 Predict",
        "📁 Batch",
        "📊 Analytics"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Machine Learning Churn Prediction System")


# =========================================================
# DASHBOARD PAGE
# =========================================================

if page == "🏠 Dashboard":

    st.title("🏠 Customer Churn Dashboard")
    st.caption("Customer risk monitoring and churn analytics")

    probabilities = model.predict_proba(df)[:, 1]
    predictions = model.predict(df)

    total_customers = len(df)
    predicted_churn = int(predictions.sum())
    churn_rate = predictions.mean()
    avg_probability = probabilities.mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

    col2.metric(
        "At-Risk Customers",
        f"{predicted_churn:,}"
    )

    col3.metric(
        "Predicted Churn Rate",
        f"{churn_rate:.1%}"
    )

    col4.metric(
        "Average Churn Risk",
        f"{avg_probability:.1%}"
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Customer Risk Distribution")

        risk_labels = pd.cut(
            probabilities,
            bins=[0, 0.30, 0.70, 1],
            labels=["Low Risk", "Medium Risk", "High Risk"],
            include_lowest=True
        )

        risk_counts = risk_labels.value_counts()

        st.bar_chart(risk_counts)

    with col2:

        st.subheader("Prediction Distribution")

        prediction_counts = pd.Series(
            predictions
        ).map({
            0: "Stay",
            1: "Churn"
        }).value_counts()

        st.bar_chart(prediction_counts)

    st.divider()

    st.subheader("High-Risk Customer Watchlist")

    dashboard_df = df.copy()

    dashboard_df["Churn Probability"] = probabilities

    dashboard_df["Risk Level"] = pd.cut(
        probabilities,
        bins=[0, 0.30, 0.70, 1],
        labels=["Low", "Medium", "High"],
        include_lowest=True
    )

    high_risk = dashboard_df.sort_values(
        "Churn Probability",
        ascending=False
    ).head(10)

    st.dataframe(
        high_risk,
        use_container_width=True
    )


# =========================================================
# PREDICT PAGE
# =========================================================

elif page == "🔮 Predict":

    st.title("🔮 Single Customer Prediction")
    st.caption("Predict churn risk for an individual customer")

    customer_id = st.number_input(
        "Select Customer Row",
        min_value=0,
        max_value=len(df) - 1,
        value=0,
        step=1
    )

    customer = df.iloc[[customer_id]]

    st.subheader("Customer Profile")

    profile_col1, profile_col2, profile_col3 = st.columns(3)

    profile_col1.metric(
        "Tenure",
        f"{customer['Tenure Months'].iloc[0]} months"
    )

    profile_col2.metric(
        "Monthly Charges",
        f"${customer['Monthly Charges'].iloc[0]:.2f}"
    )

    profile_col3.metric(
        "Total Charges",
        f"${customer['Total Charges'].iloc[0]:.2f}"
    )

    with st.expander("View Complete Customer Features"):

        st.dataframe(
            customer,
            use_container_width=True
        )

    if st.button(
        "Predict Churn",
        type="primary",
        use_container_width=True
    ):

        prediction = model.predict(customer)[0]

        probability = model.predict_proba(
            customer
        )[0][1]

        if probability < 0.30:

            risk = "Low Risk"

        elif probability < 0.70:

            risk = "Medium Risk"

        else:

            risk = "High Risk"

        st.divider()

        st.subheader("Prediction Results")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Churn Probability",
            f"{probability:.2%}"
        )

        col2.metric(
            "Risk Level",
            risk
        )

        col3.metric(
            "Prediction",
            "CHURN" if prediction == 1 else "STAY"
        )

        st.progress(float(probability))

        if prediction == 1:

            st.error(
                "⚠️ Customer is predicted to churn."
            )

            st.subheader("💡 Retention Recommendations")

            st.markdown("""
            - Contact the customer proactively.
            - Offer a personalised loyalty discount.
            - Review billing and service concerns.
            - Recommend a long-term contract.
            - Provide targeted service bundles.
            """)

        else:

            st.success(
                "✅ Customer is predicted to stay."
            )

            st.subheader("💡 Engagement Recommendations")

            st.markdown("""
            - Maintain current service quality.
            - Continue customer engagement.
            - Consider premium service upselling.
            - Offer loyalty rewards.
            """)


# =========================================================
# BATCH PAGE
# =========================================================

elif page == "📁 Batch":

    st.title("📁 Batch Churn Prediction")
    st.caption("Upload customer data and generate churn predictions")

    uploaded_file = st.file_uploader(
        "Upload Customer CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        batch_df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data")

        st.dataframe(
            batch_df.head(),
            use_container_width=True
        )

        expected_columns = list(
            model.named_steps["model"].feature_names_in_
        )

        missing_columns = [
            col
            for col in expected_columns
            if col not in batch_df.columns
        ]

        if missing_columns:

            st.error(
                "CSV does not contain all required model features."
            )

            st.write(
                "Missing columns:",
                missing_columns
            )

        else:

            batch_df = batch_df[expected_columns]

            if st.button(
                "Run Batch Prediction",
                type="primary"
            ):

                predictions = model.predict(batch_df)

                probabilities = model.predict_proba(
                    batch_df
                )[:, 1]

                results = batch_df.copy()

                results["Prediction"] = predictions

                results["Churn Status"] = pd.Series(
                    predictions
                ).map({
                    0: "Stay",
                    1: "Churn"
                })

                results["Churn Probability"] = probabilities

                results["Risk Level"] = pd.cut(
                    probabilities,
                    bins=[0, 0.30, 0.70, 1],
                    labels=[
                        "Low",
                        "Medium",
                        "High"
                    ],
                    include_lowest=True
                )

                st.success(
                    f"Predictions completed for {len(results)} customers."
                )

                st.subheader("Prediction Results")

                st.dataframe(
                    results,
                    use_container_width=True
                )

                csv = results.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "⬇️ Download Prediction Results",
                    data=csv,
                    file_name="churn_predictions.csv",
                    mime="text/csv"
                )


# =========================================================
# ANALYTICS PAGE
# =========================================================

elif page == "📊 Analytics":

    st.title("📊 Model Analytics")
    st.caption("Machine learning model performance and insights")

    st.subheader("Model Information")

    model_name = type(
        model.named_steps["model"]
    ).__name__

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Model",
        model_name
    )

    col2.metric(
        "Input Features",
        len(
            model.named_steps[
                "model"
            ].feature_names_in_
        )
    )

    col3.metric(
        "Training Algorithm",
        "Random Forest"
    )

    st.divider()

    st.subheader("Model Configuration")

    estimator = model.named_steps["model"]

    config_df = pd.DataFrame({
        "Parameter": [
            "Number of Trees",
            "Maximum Depth",
            "Minimum Samples Split",
            "Random State"
        ],
        "Value": [
            estimator.n_estimators,
            estimator.max_depth,
            estimator.min_samples_split,
            estimator.random_state
        ]
    })

    st.dataframe(
        config_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Feature Importance")

    feature_names = estimator.feature_names_in_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": estimator.feature_importances_
    }).sort_values(
        "Importance",
        ascending=False
    )

    top_features = importance_df.head(10)

    st.bar_chart(
        top_features.set_index("Feature")
    )

    st.subheader("Top Predictive Features")

    st.dataframe(
        top_features,
        use_container_width=True,
        hide_index=True
    )
