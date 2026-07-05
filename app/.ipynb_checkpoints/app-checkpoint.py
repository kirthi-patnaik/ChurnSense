import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Telco Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# Load model and data
model = joblib.load("churn_model.pkl")
df = pd.read_csv("sample_customers.csv")

st.title("📊 Telco Customer Churn Predictor")
st.markdown("Predict customer churn using Machine Learning")

# Customer selector
customer_id = st.number_input(
    "Select Customer Row",
    min_value=0,
    max_value=len(df)-1,
    value=0
)

customer = df.iloc[[customer_id]]

st.subheader("Customer Profile")

st.dataframe(customer)

if st.button("Predict Churn"):

    prediction = model.predict(customer)[0]
    probability = model.predict_proba(customer)[0][1]

    st.subheader("Prediction Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Churn Probability",
            f"{probability:.2%}"
        )

    with col2:

        if probability < 0.30:
            risk = "Low"
        elif probability < 0.70:
            risk = "Medium"
        else:
            risk = "High"

        st.metric("Risk Level", risk)

    with col3:
        st.metric(
            "Prediction",
            "Churn" if prediction == 1 else "Stay"
        )

    st.divider()

    if prediction == 1:

        st.error(
            f"⚠ Customer likely to churn ({probability:.2%})"
        )

        st.subheader("Retention Recommendations")

        st.write("""
        • Offer loyalty discounts

        • Contact customer support proactively

        • Review billing concerns

        • Promote long-term contract plans

        • Offer bundled internet and streaming services
        """)

    else:

        st.success(
            f"✅ Customer likely to stay ({probability:.2%})"
        )

        st.subheader("Customer Status")

        st.write("""
        • Customer appears stable

        • Continue engagement campaigns

        • Maintain service quality

        • Upsell premium plans if appropriate
        """)