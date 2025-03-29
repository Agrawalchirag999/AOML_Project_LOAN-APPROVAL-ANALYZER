# Loan Approval Prediction

## 📌 Problem Statement

Banks and financial institutions process thousands of loan applications daily. A wrong approval can lead to financial losses, while an unnecessary rejection can deny deserving applicants access to funds. Manual loan assessments are time-consuming, inconsistent, and prone to errors. This project leverages machine learning to automate and enhance loan approval decisions by analyzing applicant details such as income, credit history, and financial stability. By accurately predicting loan approvals, financial institutions can minimize risks, improve efficiency, and provide fair lending opportunities to applicants.

## 📊 Dataset Details

- **Source:** [https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)
- **Description:** The dataset contains applicant details such as demographic information, financial status, and loan-related parameters.
- **Features:**
  - `Loan_ID`: Unique identifier for the loan application.
  - `Gender`: Male/Female.
  - `Married`: Marital status.
  - `Dependents`: Number of dependents.
  - `Education`: Education level.
  - `Self_Employed`: Employment type.
  - `ApplicantIncome`: Applicant's income.
  - `CoapplicantIncome`: Co-applicant's income.
  - `LoanAmount`: Loan amount requested.
  - `Loan_Amount_Term`: Loan repayment term.
  - `Credit_History`: Creditworthiness score.
  - `Property_Area`: Location type (Urban, Semiurban, Rural).
  - `Loan_Status`: Target variable (Approved/Not Approved).

## 🏗 Solution Approach

1. **Data Preprocessing**
   - Handling missing values.
   - Encoding categorical variables.
   - Standardizing numerical features.
2. **Model Training & Evaluation**
   - Trained multiple models: KNN, Decision Tree, Random Forest, XGBoost, CatBoost, LightGBM, and an Ensemble model.
   - Optimized models using hyperparameter tuning (Optuna) for best performance.
   - Evaluated models using Accuracy, Precision, Recall, and F1-score.
3. **Deployment**
   - Built a user-friendly interface using Streamlit.
   - Allows users to input loan details and get a real-time prediction with probabilities and graphical representation.

## 📈 Results & Model Comparison

| Model         | Accuracy | F1 Score |
| ------------- | -------- | -------- |
| KNN           | 96.6%    | 0.979    |
| Decision Tree | 88.0%    | 0.912    |
| Random Forest | 91.0%    | 0.895    |
| XGBoost       | 99.985%  | 1.0      |
| CatBoost      | 99.8%    | 0.999    |
| LightGBM      | 98.5%    | 0.96     |
| Ensemble      |  98.99%  | 0.98     |

## 📊 Streamlit Interface & Final Decision

- Users input their details and get approval probability for each model.
- A graphical representation shows model performance.
- The final loan decision is made based on the best-performing model.
 ![image](https://github.com/user-attachments/assets/d3c98f4e-c591-4d7e-826e-05c0954c9b26)

 ![image](https://github.com/user-attachments/assets/60f7fc74-2dbf-4335-81bf-e47c7caf5067)

## 🚀 How to Run the Project

1. Clone the repository:
   ```bash
   git clone <repository-link>
   cd loan-approval-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## 👥 Contributors

- Chirag Agrawal (J016)
- Jeet Shorey (J024)

## 🙌 Acknowledgments

Special thanks to Manan Jhaveri.

