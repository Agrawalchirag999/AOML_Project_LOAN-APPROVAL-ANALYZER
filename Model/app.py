import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px

# Load all trained models
models = {
    "KNN": joblib.load("KNN_classifier.model"),
    "Decision Tree": joblib.load("Decison_Tree_Classifier.model"),
    "Random Forest": joblib.load("Random_Forest_Classifier.model"),
    "XGBoost": joblib.load("best_xgb_loan_model (1).h5"),
    "CATBoost": joblib.load("CatBoost_classifier (1).model"),
    "LightGBM": joblib.load("LightGBM_Optimized.model"),
    "ENSEMBLE": joblib.load("Ensemble.model")
}

# Apply Custom CSS for Styling
st.markdown(
    """
    <style>
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #f8f9fa;
            padding: 20px;
        }

        /* Main Title */
        .title {
            color: #007bff;
            text-align: center;
            font-size: 34px;
            font-weight: bold;
        }
        
        /* Styled Buttons */
        div.stButton > button {
            background-color: #007bff;
            color: white;
            font-size: 16px;
            border-radius: 8px;
            padding: 12px 24px;
            transition: 0.3s;
        }
        div.stButton > button:hover {
            background-color: #0056b3;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title with color
st.markdown('<h1 class="title">🏦 Loan Approval Prediction</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🔍 Loan Analyzer")

with st.sidebar.expander("📌 User Information", expanded=True):
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Married = st.selectbox("Married", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])

with st.sidebar.expander("🏦 Loan Information", expanded=True):
    ApplicantIncome = st.number_input("Applicant Income", min_value=0, step=500)
    CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0, step=500)
    LoanAmount = st.number_input("Loan Amount (in thousands)", min_value=0, step=10)
    Loan_Amount_Term = st.selectbox("Loan Term (Months)", [360, 180, 120, 60])
    Credit_History = st.selectbox("Credit History", [1, 0])
    Property_Area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

# One-hot encoding for Property_Area
Property_Area_Rural = 1 if Property_Area == "Rural" else 0
Property_Area_Semiurban = 1 if Property_Area == "Semiurban" else 0
Property_Area_Urban = 1 if Property_Area == "Urban" else 0

# Prepare input data
input_data = np.array([[1 if Gender == "Male" else 0,
                        1 if Married == "Yes" else 0,
                        int(Dependents[0]) if Dependents != "3+" else 3,
                        1 if Education == "Graduate" else 0,
                        1 if Self_Employed == "Yes" else 0,
                        ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
                        Credit_History, Property_Area_Rural, Property_Area_Semiurban, Property_Area_Urban]])

if st.button("🔍 Predict Loan Approval"):
    results = {}
    for model_name, model in models.items():
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1] if hasattr(model, 'predict_proba') else None
        results[model_name] = {
            "Approval Probability": round(probability * 100, 2) if probability else "N/A",
            "Prediction": "✅ Approved" if prediction == 1 else "❌ Rejected"
        }

    results_df = pd.DataFrame(results).T.reset_index()
    results_df.columns = ["Model", "Approval Probability (%)", "Prediction"]

    st.success("✅ Loan Prediction Completed!")
    st.dataframe(results_df, use_container_width=True)

    fig = px.bar(results_df, x="Model", y="Approval Probability (%)", color="Prediction", 
                 title="Loan Approval Probability by Model", text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

    # Get the final decision from ENSEMBLE model
    final_decision = models["ENSEMBLE"].predict(input_data)[0]
    final_prediction = "✅ Approved" if final_decision == 1 else "❌ Rejected"

    st.markdown(
    f"""
    <div style="
        background: linear-gradient(135deg, #007bff, #0056b3);
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.25);
        margin-top: 20px;
    ">
        <h2 style="color:white; font-size:24px; margin-bottom:10px;">
            🏦 Final Loan Decision
        </h2>
        <h1 style="color:#FFD700; font-size:42px;">
            {final_prediction}
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)