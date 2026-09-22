import os
import joblib
import pandas as pd
import numpy as np
from preprocessing import feature_engineering

class CreditRiskPredictor:
    def __init__(self, model_path=os.path.join("models", "credit_risk_best_pipeline.joblib")):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Run model.py or main.py first.")
        self.pipeline = joblib.load(model_path)

    def predict_applicant(self, applicant_data: dict):
        """
        Accepts a dictionary of applicant data, runs feature engineering and pipeline prediction.
        Returns risk score, default probability, risk tier, and loan recommendation.
        """
        df_input = pd.DataFrame([applicant_data])
        df_engineered = feature_engineering(df_input)
        
        # Predict probability of default (class 1)
        prob_default = float(self.pipeline.predict_proba(df_engineered)[:, 1][0])
        default_pred = int(self.pipeline.predict(df_engineered)[0])
        
        # Convert default probability to a standard credit risk rating (0 - 1000, higher is riskier)
        risk_score = int(round(prob_default * 1000))
        
        # Determine Risk Tier and Underwriting Decision
        if prob_default < 0.12:
            risk_tier = "Low Risk"
            decision = "APPROVED (Instant Approval)"
            action_code = "AUTO_APPROVE"
        elif prob_default < 0.30:
            risk_tier = "Moderate Risk"
            decision = "APPROVED (Standard Pricing)"
            action_code = "APPROVE_STANDARD"
        elif prob_default < 0.50:
            risk_tier = "High Risk"
            decision = "MANUAL REVIEW REQUIRED (Higher Rate / Collateral Needed)"
            action_code = "REFER_UNDERWRITER"
        else:
            risk_tier = "Critical Default Risk"
            decision = "DECLINED (High Expected Loss)"
            action_code = "AUTO_REJECT"
            
        return {
            'applicant_id': applicant_data.get('customer_id', 'APPLICANT-SAMPLE'),
            'default_probability': round(prob_default, 4),
            'default_probability_pct': f"{prob_default * 100:.2f}%",
            'risk_score': risk_score,
            'risk_tier': risk_tier,
            'underwriting_decision': decision,
            'action_code': action_code
        }

if __name__ == "__main__":
    predictor = CreditRiskPredictor()
    
    # Sample Applicant 1: Strong Credit Profile
    sample_low_risk = {
        'customer_id': 'APPL-001',
        'person_age': 35,
        'person_income': 95000,
        'person_home_ownership': 'MORTGAGE',
        'person_emp_length': 8.5,
        'loan_intent': 'HOMEIMPROVEMENT',
        'loan_grade': 'A',
        'loan_amnt': 12000,
        'loan_int_rate': 7.5,
        'loan_percent_income': 0.126,
        'dti': 15.2,
        'revolving_utilization': 22.4,
        'credit_score': 760,
        'cb_person_default_on_file': 'N',
        'cb_person_cred_hist_length': 12
    }
    
    # Sample Applicant 2: Vulnerable Credit Profile
    sample_high_risk = {
        'customer_id': 'APPL-002',
        'person_age': 23,
        'person_income': 28000,
        'person_home_ownership': 'RENT',
        'person_emp_length': 1.0,
        'loan_intent': 'PERSONAL',
        'loan_grade': 'E',
        'loan_amnt': 18000,
        'loan_int_rate': 21.5,
        'loan_percent_income': 0.642,
        'dti': 42.5,
        'revolving_utilization': 88.0,
        'credit_score': 560,
        'cb_person_default_on_file': 'Y',
        'cb_person_cred_hist_length': 3
    }
    
    print("\n--- SAMPLE INFERENCE 1: LOW RISK APPLICANT ---")
    res1 = predictor.predict_applicant(sample_low_risk)
    for k, v in res1.items():
        print(f"  {k}: {v}")
        
    print("\n--- SAMPLE INFERENCE 2: HIGH RISK APPLICANT ---")
    res2 = predictor.predict_applicant(sample_high_risk)
    for k, v in res2.items():
        print(f"  {k}: {v}")
