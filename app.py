import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

st.set_page_config(
    page_title="Telecom Churn Prediction & Retention",
    page_icon="📡",
    layout="wide",
)

sns.set_style("whitegrid")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RENAME_MAP = {
    "CustomerID": "customerID",
    "Gender": "gender",
    "Senior Citizen": "SeniorCitizen",
    "Tenure Months": "tenure",
    "Phone Service": "PhoneService",
    "Multiple Lines": "MultipleLines",
    "Internet Service": "InternetService",
    "Online Security": "OnlineSecurity",
    "Online Backup": "OnlineBackup",
    "Device Protection": "DeviceProtection",
    "Tech Support": "TechSupport",
    "Streaming TV": "StreamingTV",
    "Streaming Movies": "StreamingMovies",
    "Paperless Billing": "PaperlessBilling",
    "Payment Method": "PaymentMethod",
    "Monthly Charges": "MonthlyCharges",
    "Total Charges": "TotalCharges",
    "Churn Label": "Churn",
}

DROP_COLS = [
    "Count", "Country", "State", "City", "Zip Code", "Lat Long",
    "Latitude", "Longitude", "Churn Value", "Churn Score", "CLTV", "Churn Reason",
]


@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:
    if file.name.endswith(".xlsx") or file.name.endswith(".xls"):
        df = pd.read_excel(file)
    else:
        df = pd.read_csv(file)

    df = df.rename(columns=RENAME_MAP)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    if "Churn" in df.columns and df["Churn"].dtype == object:
        df["Churn"] = df["Churn"].astype(str).str.strip()

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    return df


@st.cache_data(show_spinner=False)
def generate_sample_data(n=2000, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    gender = rng.choice(["Male", "Female"], n)
    senior = rng.choice([0, 1], n, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], n)
    dependents = rng.choice(["Yes", "No"], n, p=[0.3, 0.7])
    tenure = rng.integers(0, 73, n)
    phone_service = rng.choice(["Yes", "No"], n, p=[0.9, 0.1])
    multiple_lines = rng.choice(["Yes", "No", "No phone service"], n)
    internet_service = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.44, 0.21])
    online_security = rng.choice(["Yes", "No", "No internet service"], n)
    tech_support = rng.choice(["Yes", "No", "No internet service"], n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.24, 0.21])
    paperless = rng.choice(["Yes", "No"], n, p=[0.59, 0.41])
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], n
    )
    monthly_charges = np.round(rng.uniform(18, 120, n), 2)
    total_charges = np.round(monthly_charges * tenure + rng.uniform(0, 50, n), 2)

    churn_prob = (
        0.5 * (contract == "Month-to-month")
        + 0.25 * (internet_service == "Fiber optic")
        + 0.2 * (payment_method == "Electronic check")
        + 0.15 * (tenure < 12)
        + 0.1 * (monthly_charges > 80)
        - 0.2 * (contract == "Two year")
        - 0.15 * (tenure > 48)
    )
    churn_prob = 1 / (1 + np.exp(-3 * (churn_prob - 0.62)))
    churn = rng.binomial(1, np.clip(churn_prob, 0.02, 0.95))
    churn = np.where(churn == 1, "Yes", "No")

    df = pd.DataFrame({
        "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": phone_service, "MultipleLines": multiple_lines,
        "InternetService": internet_service, "OnlineSecurity": online_security, "TechSupport": tech_support,
        "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges, "TotalCharges": total_charges, "Churn": churn,
    })
    return df


@st.cache_resource(show_spinner=True)
def train_models(df: pd.DataFrame):
    data = df.copy()
    data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})
    data = data.dropna(subset=["Churn"])

    binary_cols = [c for c in data.select_dtypes(include="object").columns if data[c].nunique() == 2]
    multi_cols = [c for c in data.select_dtypes(include="object").columns if c not in binary_cols]

    encoders = {}
    for c in binary_cols:
        le = LabelEncoder()
        data[c] = le.fit_transform(data[c].astype(str))
        encoders[c] = le

    data = pd.get_dummies(data, columns=multi_cols, drop_first=True)

    X = data.drop(columns=["Churn"])
    y = data["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Neural Network (MLP)": MLPClassifier(
            hidden_layer_sizes=(64, 32, 16), max_iter=500, random_state=42, early_stopping=True
        ),
    }

    results = {}
    fitted = {}
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_proba = model.predict_proba(X_test_s)[:, 1]
        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_proba),
        }
        fitted[name] = model

    results_df = pd.DataFrame(results).T.sort_values("ROC-AUC", ascending=False)

    test_out = X_test.copy()
    test_out["Actual_Churn"] = y_test.values
    best_name = results_df.index[0]
    test_out["Churn_Probability"] = fitted[best_name].predict_proba(X_test_s)[:, 1]

    def tier(p):
        return "High Risk" if p >= 0.7 else ("Medium Risk" if p >= 0.4 else "Low Risk")

    test_out["Risk_Tier"] = test_out["Churn_Probability"].apply(tier)

    return {
        "fitted": fitted,
        "results_df": results_df,
        "best_name": best_name,
        "scaler": scaler,
        "encoders": encoders,
        "feature_columns": X.columns.tolist(),
        "test_out": test_out,
        "X_test": X_test,
    }


def retention_offer(tenure, monthly_charges, risk_tier, contract):
    """Rule-based fallback retention message (works with no API key)."""
    if tenure < 12:
        return (
            f"We noticed you're fairly new with us ({tenure} months). "
            f"As a welcome, here's 20% off your next 6 months if you move to a 1-year plan, "
            f"plus a free service upgrade."
        )
    elif monthly_charges > 80:
        return (
            f"We appreciate your loyalty over {tenure} months! To keep your monthly cost more "
            f"comfortable, we'd like to offer you a bundle discount reducing your bill by 15%."
        )
    elif contract == "Month-to-month":
        return (
            f"Thanks for being with us for {tenure} months on a flexible plan. "
            f"Switch to a 1-year contract and get 2 months free plus priority support."
        )
    else:
        return (
            f"Thanks for being with us for {tenure} months. "
            f"We'd love to offer you a free tech support upgrade and 10% off your next 3 bills."
        )


def genai_offer(profile: dict, api_key: str) -> str:
    """Live GenAI offer using Anthropic API (only called if key provided)."""
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    prompt = f"""You are a telecom customer retention specialist.
A customer is at risk of churning. Here is their profile:
{profile}

Write a short, warm, personalized retention message (3-4 sentences) that:
1. Acknowledges their loyalty / usage pattern
2. Addresses a likely pain point suggested by the data
3. Offers ONE concrete retention incentive
4. Has a friendly closing call-to-action

Keep it under 80 words."""
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


# ---------------------------------------------------------------------------
# Sidebar — Data source
# ---------------------------------------------------------------------------

st.sidebar.title("📡 Telecom Churn App")
st.sidebar.markdown("Upload the Telco Customer Churn dataset (CSV or XLSX), or use sample data to explore.")

data_source = st.sidebar.radio("Data source", ["Use sample data", "Upload my own file"])

if data_source == "Upload my own file":
    uploaded_file = st.sidebar.file_uploader("Upload CSV or XLSX", type=["csv", "xlsx", "xls"])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
    else:
        st.sidebar.info("Upload a file to continue, or switch to sample data.")
        st.stop()
else:
    df = generate_sample_data()
    st.sidebar.success("Using generated sample data (2000 customers).")

st.sidebar.markdown("---")
api_key = st.sidebar.text_input("Anthropic API Key (optional, for live GenAI offers)", type="password")
st.sidebar.caption("Leave blank to use rule-based offers instead of live Claude API calls.")

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------

st.title("Telecom Customer Churn Prediction & Retention Recommendation")
st.caption("AI-driven churn prediction with personalized retention recommendations.")

tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "🤖 Model Comparison", "🎯 Predict a Customer", "💌 Retention Offers"])

# --- TAB 1: EDA ---
with tab1:
    st.subheader("Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Customers", f"{len(df):,}")
    if "Churn" in df.columns:
        churn_rate = (df["Churn"] == "Yes").mean() * 100
        c2.metric("Churn Rate", f"{churn_rate:.1f}%")
    c3.metric("Features", f"{df.shape[1]}")

    st.dataframe(df.head(10), use_container_width=True)

    if "Churn" in df.columns:
        colA, colB = st.columns(2)
        with colA:
            st.markdown("**Churn Distribution**")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.countplot(x="Churn", data=df, palette="Set2", ax=ax)
            st.pyplot(fig)

        with colB:
            if "Contract" in df.columns:
                st.markdown("**Churn by Contract Type**")
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.countplot(x="Contract", hue="Churn", data=df, palette="Set2", ax=ax)
                plt.xticks(rotation=15)
                st.pyplot(fig)

        if "tenure" in df.columns:
            st.markdown("**Tenure by Churn**")
            fig, ax = plt.subplots(figsize=(8, 3.5))
            sns.boxplot(x="Churn", y="tenure", data=df, palette="Set2", ax=ax)
            st.pyplot(fig)
    else:
        st.warning("No 'Churn' column found — EDA charts related to churn are skipped.")

# --- TAB 2: Model Comparison ---
with tab2:
    st.subheader("Model Training & Comparison")
    if "Churn" not in df.columns:
        st.error("Dataset needs a 'Churn' column to train models.")
    else:
        with st.spinner("Training models (Logistic Regression, Random Forest, Neural Network)..."):
            bundle = train_models(df)

        st.markdown("**Performance on held-out test set**")
        st.dataframe(bundle["results_df"].style.background_gradient(cmap="Greens"), use_container_width=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        bundle["results_df"][["ROC-AUC", "F1", "Recall"]].plot(kind="bar", ax=ax)
        plt.xticks(rotation=15)
        plt.ylabel("Score")
        st.pyplot(fig)

        st.success(f"Best model by ROC-AUC: **{bundle['best_name']}**")

        st.markdown("**Top 10 Highest-Risk Customers (test set)**")
        top_risk = bundle["test_out"].sort_values("Churn_Probability", ascending=False).head(10)
        show_cols = [c for c in ["tenure", "MonthlyCharges", "Actual_Churn", "Churn_Probability", "Risk_Tier"] if c in top_risk.columns]
        st.dataframe(top_risk[show_cols], use_container_width=True)

# --- TAB 3: Predict a Customer ---
with tab3:
    st.subheader("Predict Churn for a Single Customer")
    if "Churn" not in df.columns:
        st.error("Dataset needs a 'Churn' column to train/predict.")
    else:
        bundle = train_models(df)

        with st.form("predict_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                tenure_in = st.slider("Tenure (months)", 0, 72, 6)
                monthly_in = st.slider("Monthly Charges", 18.0, 120.0, 75.0)
                total_in = st.number_input("Total Charges", 0.0, 10000.0, float(monthly_in * tenure_in))
            with col2:
                contract_in = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                internet_in = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
                payment_in = st.selectbox(
                    "Payment Method",
                    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                )
            with col3:
                paperless_in = st.selectbox("Paperless Billing", ["Yes", "No"])
                partner_in = st.selectbox("Partner", ["Yes", "No"])
                senior_in = st.selectbox("Senior Citizen", [0, 1])

            submitted = st.form_submit_button("Predict Churn Risk")

        if submitted:
            row = pd.DataFrame([{c: 0 for c in bundle["feature_columns"]}])
            for c in ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]:
                if c in row.columns:
                    row.at[0, c] = {"tenure": tenure_in, "MonthlyCharges": monthly_in,
                                     "TotalCharges": total_in, "SeniorCitizen": senior_in}[c]

            dummy_flags = {
                f"Contract_{contract_in}": 1,
                f"InternetService_{internet_in}": 1,
                f"PaymentMethod_{payment_in}": 1,
            }
            for col, val in dummy_flags.items():
                if col in row.columns:
                    row.at[0, col] = val

            for c, le in bundle["encoders"].items():
                if c == "PaperlessBilling" and c in row.columns:
                    row.at[0, c] = 1 if paperless_in == "Yes" else 0
                elif c == "Partner" and c in row.columns:
                    row.at[0, c] = 1 if partner_in == "Yes" else 0

            row_scaled = bundle["scaler"].transform(row[bundle["feature_columns"]])
            model = bundle["fitted"][bundle["best_name"]]
            proba = model.predict_proba(row_scaled)[0, 1]
            tier = "High Risk" if proba >= 0.7 else ("Medium Risk" if proba >= 0.4 else "Low Risk")

            st.metric("Churn Probability", f"{proba*100:.1f}%", delta=tier)

            st.session_state["last_prediction"] = {
                "tenure": tenure_in, "monthly_charges": monthly_in,
                "contract": contract_in, "proba": proba, "tier": tier,
            }

            if tier == "High Risk":
                st.error(f"⚠️ {tier} — recommend proactive retention outreach.")
            elif tier == "Medium Risk":
                st.warning(f"⚠️ {tier} — monitor and consider a light-touch offer.")
            else:
                st.success(f"✅ {tier} — customer looks stable.")

# --- TAB 4: Retention Offers ---
with tab4:
    st.subheader("Personalized Retention Offer")
    pred = st.session_state.get("last_prediction")
    if not pred:
        st.info("Go to the 'Predict a Customer' tab first and run a prediction.")
    else:
        st.write(f"**Tenure:** {pred['tenure']} months | **Monthly Charges:** ${pred['monthly_charges']:.2f} | "
                  f"**Contract:** {pred['contract']} | **Risk:** {pred['tier']} ({pred['proba']*100:.1f}%)")

        if st.button("Generate Retention Offer"):
            if api_key:
                with st.spinner("Calling Claude API..."):
                    try:
                        msg = genai_offer({
                            "tenure_months": pred["tenure"],
                            "monthly_charges": pred["monthly_charges"],
                            "contract": pred["contract"],
                            "churn_probability": round(pred["proba"], 3),
                            "risk_tier": pred["tier"],
                        }, api_key)
                        st.success(msg)
                    except Exception as e:
                        st.error(f"API call failed ({e}). Showing rule-based offer instead.")
                        st.info(retention_offer(pred["tenure"], pred["monthly_charges"], pred["tier"], pred["contract"]))
            else:
                st.info(retention_offer(pred["tenure"], pred["monthly_charges"], pred["tier"], pred["contract"]))

st.markdown("---")
st.caption("Built for hackathon demo — Telecom Customer Churn Prediction & Retention Recommendation (Classification)")
