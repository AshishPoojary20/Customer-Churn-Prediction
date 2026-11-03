import streamlit as st
import pandas as pd
import joblib

from pathlib import Path


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Churn Prediction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. PROJECT DIRECTORY
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"
DATASET_DIR = BASE_DIR / "dataset"


# =========================================================
# 3. LOAD MACHINE LEARNING MODELS
# =========================================================

@st.cache_resource
def load_models():

    model = joblib.load(
        MODEL_DIR / "random_forest_model.pkl"
    )

    preprocessor = joblib.load(
        MODEL_DIR / "preprocessor.pkl"
    )

    return model, preprocessor


try:

    model, preprocessor = load_models()

except Exception as e:

    st.error(f"Unable to load machine learning models: {e}")

    st.stop()


# =========================================================
# 4. LOAD CUSTOMER DATASET
# =========================================================

@st.cache_data
def load_data():

    file_path = DATASET_DIR / "cleaned_customer_churn.csv"

    df = pd.read_csv(file_path)

    return df


try:

    analytics_df = load_data()

except Exception as e:

    st.error(f"Unable to load customer dataset: {e}")

    st.stop()


# =========================================================
# 5. APPLICATION HEADER
# =========================================================

st.title("📊 Customer Churn Prediction System")

st.write(
    """
    An intelligent customer churn prediction system
    developed using Machine Learning.

    Predict customer behaviour and analyze customer
    retention patterns using historical telecom data.
    """
)

st.divider()


# =========================================================
# 6. SIDEBAR
# =========================================================

st.sidebar.title("📊 Navigation")

st.sidebar.write(
    "Customer Churn Prediction"
)

st.sidebar.info(
    "Machine Learning Model: Random Forest"
)

st.sidebar.success(
    "Application Status: Active"
)


# =========================================================
# 7. APPLICATION TABS
# =========================================================

prediction_tab, dashboard_tab = st.tabs(
    [
        "🔮 Churn Prediction",
        "📊 Business Analytics Dashboard"
    ]
)


# =========================================================
# 8. CUSTOMER CHURN PREDICTION
# =========================================================

with prediction_tab:

    st.header("Customer Information")

    st.write(
        "Enter customer details to predict the "
        "possibility of customer churn."
    )

    with st.form("customer_prediction_form"):

        col1, col2, col3 = st.columns(3)

        # -------------------------------------------------
        # PERSONAL INFORMATION
        # -------------------------------------------------

        with col1:

            st.subheader("Personal Information")

            gender = st.selectbox(
                "Gender",
                ["Male", "Female"]
            )

            senior = st.selectbox(
                "Senior Citizen",
                [0, 1]
            )

            partner = st.selectbox(
                "Partner",
                ["Yes", "No"]
            )

            dependents = st.selectbox(
                "Dependents",
                ["Yes", "No"]
            )

            tenure = st.number_input(
                "Tenure (Months)",
                min_value=0,
                max_value=72,
                value=12
            )

        # -------------------------------------------------
        # SERVICE INFORMATION
        # -------------------------------------------------

        with col2:

            st.subheader("Service Information")

            phone = st.selectbox(
                "Phone Service",
                ["Yes", "No"]
            )

            multiple = st.selectbox(
                "Multiple Lines",
                [
                    "Yes",
                    "No",
                    "No phone service"
                ]
            )

            internet = st.selectbox(
                "Internet Service",
                [
                    "DSL",
                    "Fiber optic",
                    "No"
                ]
            )

            security = st.selectbox(
                "Online Security",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )

            backup = st.selectbox(
                "Online Backup",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )

        # -------------------------------------------------
        # ADDITIONAL SERVICES
        # -------------------------------------------------

        with col3:

            st.subheader("Additional Services")

            protection = st.selectbox(
                "Device Protection",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )

            support = st.selectbox(
                "Tech Support",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )

            streaming_tv = st.selectbox(
                "Streaming TV",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )

            streaming_movies = st.selectbox(
                "Streaming Movies",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )

        st.divider()

        # -------------------------------------------------
        # SUBSCRIPTION INFORMATION
        # -------------------------------------------------

        st.subheader("Subscription Information")

        col4, col5 = st.columns(2)

        with col4:

            contract = st.selectbox(
                "Contract",
                [
                    "Month-to-month",
                    "One year",
                    "Two year"
                ]
            )

            paperless = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"]
            )

            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

        with col5:

            monthly = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                max_value=200.0,
                value=70.0
            )

            total = st.number_input(
                "Total Charges",
                min_value=0.0,
                value=float(monthly * tenure)
            )

        st.divider()

        submitted = st.form_submit_button(
            "🔮 Predict Customer Churn",
            use_container_width=True
        )

    # =====================================================
    # 9. GENERATE PREDICTION
    # =====================================================

    if submitted:

        try:

            customer_data = pd.DataFrame([{

                "gender": gender,

                "SeniorCitizen": senior,

                "Partner": partner,

                "Dependents": dependents,

                "tenure": tenure,

                "PhoneService": phone,

                "MultipleLines": multiple,

                "InternetService": internet,

                "OnlineSecurity": security,

                "OnlineBackup": backup,

                "DeviceProtection": protection,

                "TechSupport": support,

                "StreamingTV": streaming_tv,

                "StreamingMovies": streaming_movies,

                "Contract": contract,

                "PaperlessBilling": paperless,

                "PaymentMethod": payment,

                "MonthlyCharges": monthly,

                "TotalCharges": total

            }])

            # Apply preprocessing

            processed_data = preprocessor.transform(
                customer_data
            )

            # Generate prediction

            prediction = model.predict(
                processed_data
            )[0]

            # Calculate probabilities

            probabilities = model.predict_proba(
                processed_data
            )[0]

            # Identify churn class

            churn_classes = list(model.classes_)

            churn_index = next(
                (
                    i for i, value in enumerate(churn_classes)
                    if str(value).strip().lower()
                    in ["1", "yes", "true", "churn", "churned"]
                ),
                None
            )

            if churn_index is None:

                raise ValueError(
                    "Unable to identify the churn class. "
                    "Check the target encoding used during model training."
                )

            churn_probability = (
                probabilities[churn_index] * 100
            )

            is_churn = (
                str(prediction).strip().lower()
                in ["1", "yes", "true", "churn", "churned"]
            )

            # Display result

            st.divider()

            st.header("Prediction Result")

            if is_churn:

                st.error(
                    "⚠️ Customer is likely to Churn"
                )

            else:

                st.success(
                    "✅ Customer is likely to Stay"
                )

            st.metric(
                "Churn Probability",
                f"{churn_probability:.2f}%"
            )

            st.progress(
                float(churn_probability / 100)
            )

            st.caption(
                "Probability represents the model's "
                "estimated likelihood of customer churn."
            )

            # Recommendations

            st.subheader("Customer Retention Recommendation")

            if churn_probability >= 70:

                st.warning(
                    "High Risk: Consider offering special "
                    "discounts and personalized retention offers."
                )

            elif churn_probability >= 40:

                st.info(
                    "Medium Risk: Monitor customer satisfaction "
                    "and improve customer engagement."
                )

            else:

                st.success(
                    "Low Risk: Continue providing quality "
                    "services to maintain customer satisfaction."
                )

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )


# =========================================================
# 10. BUSINESS ANALYTICS DASHBOARD
# =========================================================

with dashboard_tab:

    st.header("📊 Business Analytics Dashboard")

    st.write(
        "Analyze customer churn, retention patterns, "
        "and important business performance indicators."
    )

    st.divider()

    # -----------------------------------------------------
    # DATA VALIDATION
    # -----------------------------------------------------

    required_columns = [
        "Churn",
        "Contract"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in analytics_df.columns
    ]

    if missing_columns:

        st.error(
            f"Missing required dataset columns: {missing_columns}"
        )

        st.stop()

    # Normalize churn labels

    churn_labels = (
        analytics_df["Churn"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # -----------------------------------------------------
    # KEY PERFORMANCE INDICATORS
    # -----------------------------------------------------

    total_customers = len(analytics_df)

    churned_customers = (
        churn_labels == "yes"
    ).sum()

    retained_customers = (
        churn_labels == "no"
    ).sum()

    if total_customers > 0:

        churn_rate = (
            churned_customers / total_customers
        ) * 100

        retention_rate = (
            retained_customers / total_customers
        ) * 100

    else:

        churn_rate = 0

        retention_rate = 0

    st.subheader("Key Performance Indicators")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Customers",
            f"{total_customers:,}"
        )

    with col2:

        st.metric(
            "Churned Customers",
            f"{churned_customers:,}"
        )

    with col3:

        st.metric(
            "Retained Customers",
            f"{retained_customers:,}"
        )

    with col4:

        st.metric(
            "Churn Rate",
            f"{churn_rate:.2f}%"
        )

    st.divider()

    # -----------------------------------------------------
    # CUSTOMER CHURN DISTRIBUTION
    # -----------------------------------------------------

    st.subheader("Customer Churn Distribution")

    churn_distribution = (
        analytics_df["Churn"]
        .value_counts()
        .rename_axis("Churn")
        .reset_index(name="Customers")
    )

    st.bar_chart(
        churn_distribution,
        x="Churn",
        y="Customers"
    )

    st.caption(
        "Distribution of customers who stayed "
        "and customers who churned."
    )

    st.divider()

    # -----------------------------------------------------
    # CHURN RATE BY CONTRACT TYPE
    # -----------------------------------------------------

    st.subheader("Churn Rate by Contract Type")

    contract_churn = analytics_df.copy()

    contract_churn["_Churned"] = (
        contract_churn["Churn"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("yes")
        .astype(int)
    )

    contract_churn_rate = (
        contract_churn
        .groupby("Contract")["_Churned"]
        .mean()
        * 100
    )

    contract_churn_rate = (
        contract_churn_rate
        .rename("Churn Rate (%)")
    )

    st.bar_chart(
        contract_churn_rate
    )

    st.caption(
        "Month-to-month customers generally have "
        "a higher churn rate than long-term contract customers."
    )

    st.divider()

    # -----------------------------------------------------
    # CUSTOMER RETENTION OVERVIEW
    # -----------------------------------------------------

    st.subheader("Customer Retention Overview")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Retention Rate",
            f"{retention_rate:.2f}%"
        )

    with col2:

        st.metric(
            "Churn Rate",
            f"{churn_rate:.2f}%"
        )

    st.progress(
        float(retention_rate / 100)
    )

    st.caption(
        "Percentage of customers retained by the company."
    )

    st.divider()

    # -----------------------------------------------------
    # CUSTOMER CONTRACT DISTRIBUTION
    # -----------------------------------------------------

    st.subheader("Customer Distribution by Contract")

    contract_distribution = (
        analytics_df["Contract"]
        .value_counts()
        .rename_axis("Contract")
        .reset_index(name="Customers")
    )

    st.bar_chart(
        contract_distribution,
        x="Contract",
        y="Customers"
    )

    st.divider()

    # -----------------------------------------------------
    # RANDOM FOREST FEATURE IMPORTANCE
    # -----------------------------------------------------

    st.subheader("Top Important Features")

    if hasattr(model, "feature_importances_"):

        try:

            feature_names = (
                preprocessor.get_feature_names_out()
            )

            importance_values = model.feature_importances_

            if len(feature_names) == len(importance_values):

                feature_importance = pd.DataFrame({

                    "Feature": feature_names,

                    "Importance": importance_values

                })

                feature_importance = (
                    feature_importance
                    .sort_values(
                        by="Importance",
                        ascending=False
                    )
                    .head(15)
                )

                st.bar_chart(
                    feature_importance.set_index("Feature")
                )

            else:

                st.info(
                    "Feature names do not match model importance values."
                )

        except Exception as e:

            st.info(
                f"Feature importance could not be displayed: {e}"
            )

    else:

        st.info(
            "Feature importance is not available for this model."
        )

    st.divider()

    # -----------------------------------------------------
    # CUSTOMER DATA TABLE
    # -----------------------------------------------------

    st.subheader("Customer Dataset Preview")

    st.dataframe(
        analytics_df.head(20),
        use_container_width=True
    )

    # -----------------------------------------------------
    # DOWNLOAD CUSTOMER DATA
    # -----------------------------------------------------

    st.subheader("Download Customer Data")

    csv_data = analytics_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Customer Data (CSV)",
        data=csv_data,
        file_name="customer_churn_report.csv",
        mime="text/csv",
        use_container_width=True
    )


# =========================================================
# 11. FOOTER
# =========================================================

st.divider()

st.divider()

st.markdown(
    "<h4 style='text-align:center;'>"
    "Customer Churn Prediction System"
    "</h4>",
    unsafe_allow_html=True
)

st.caption(
    "Developed using Python, Machine Learning, "
    "Pandas, Scikit-learn and Streamlit."
)

st.caption(
    "Customer Churn Prediction Project | 2026"
)