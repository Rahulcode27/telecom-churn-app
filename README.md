[README (1).md](https://github.com/user-attachments/files/31254100/README.1.md)
# 📡 Telecom Customer Churn Prediction & Retention Recommendation

An AI-driven system that predicts which telecom customers are likely to churn and recommends personalized retention actions — combining traditional ML, deep learning, and GenAI-powered personalization in a single interactive dashboard.

🔗 **Live Demo:** _[add your Streamlit Cloud URL here after deployment]_

---

## 📋 Problem Statement

Telecom operators lose significant revenue when customers switch to competitors. This project builds an AI/ML solution that predicts customer churn and recommends personalized retention actions to proactively reduce revenue loss.

## 🎯 Objective

Develop an AI-driven churn prediction and retention recommendation system that identifies customers likely to leave and recommends proactive interventions to improve customer retention.

**Type:** Classification

---

## ✨ Features

- **📊 Exploratory Data Analysis** — churn distribution, churn by contract type, tenure vs. churn visualizations
- **🤖 Multiple ML Models** — Logistic Regression, Random Forest, and a Neural Network (MLP), trained and compared side-by-side using Accuracy, Precision, Recall, F1, and ROC-AUC
- **🎯 Real-time Prediction** — enter a customer's details in a form and instantly get their churn probability and risk tier (Low / Medium / High)
- **💌 Personalized Retention Offers** — generates a tailored retention message per customer, either via the Claude (Anthropic) API for live GenAI personalization, or a rule-based fallback when no API key is provided
- **📁 Flexible Data Input** — works with either the Kaggle `WA_Fn-UseC_-Telco-Customer-Churn.csv` format or the full IBM Telco `.xlsx` dataset (column names are auto-mapped), or generated sample data if no file is uploaded

---

## 🗂️ Dataset

**Source:** [IBM Telco Customer Churn Dataset — Kaggle](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset)

Key fields used: `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `InternetService`, `PaymentMethod`, `PaperlessBilling`, `Partner`, `SeniorCitizen`, and the target `Churn`.

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data Processing | pandas, numpy |
| Machine Learning | scikit-learn (Logistic Regression, Random Forest, MLP Neural Network) |
| Visualization | matplotlib, seaborn |
| GenAI | Anthropic Claude API |
| App / Dashboard | Streamlit |
| Deployment | Streamlit Community Cloud |

---

## 🚀 Run Locally

```bash
# 1. Clone this repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## ☁️ Deployment

This app is deployed on [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Push `app.py` and `requirements.txt` to a public GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app**, select this repository, set the main file to `app.py`
4. Click **Deploy**

---

## 📖 How to Use the App

1. **Choose a data source** in the sidebar — upload your own Telco dataset (CSV/XLSX) or use the built-in sample data
2. **(Optional)** Paste an Anthropic API key in the sidebar to enable live, LLM-generated retention offers — otherwise a rule-based offer is shown
3. Explore the **EDA** tab for churn patterns
4. Check the **Model Comparison** tab to see how each model performs and which customers are highest-risk
5. Go to **Predict a Customer** to score an individual customer
6. Head to **Retention Offers** to generate a personalized retention message for that customer

---

## 🔮 Future Enhancements

- Add XGBoost / LightGBM and a full Keras deep learning model
- SHAP-based explainability for individual predictions
- Agentic AI workflow (CrewAI / LangGraph) chaining a churn-analysis agent → retention-strategy agent → outreach-copywriter agent
- A/B testing framework to measure real-world retention offer effectiveness
- Persist predictions and offers to a database for tracking over time

---

## 📄 License

This project was built for hackathon/educational purposes.
