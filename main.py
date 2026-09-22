import os
import sys

# Ensure src directory is in Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from dataset_generator import generate_loan_dataset
from database import CreditRiskDatabase
from eda import generate_eda_plots
from model import CreditRiskModelTrainer
from predict import CreditRiskPredictor

def run_pipeline():
    print("=" * 70)
    print("      CREDIT RISK & LOAN DEFAULT ANALYSIS - END-TO-END PIPELINE      ")
    print("=" * 70)

    # Step 1: Generate Synthetic Realistic Loan Dataset
    print("\n[STEP 1/5] Generating raw loan dataset...")
    os.makedirs("data", exist_ok=True)
    csv_path = os.path.join("data", "raw_loan_data.csv")
    df_raw = generate_loan_dataset(n_samples=10000, random_state=42)
    df_raw.to_csv(csv_path, index=False)
    print(f" -> Generated {len(df_raw)} records saved to {csv_path}")

    # Step 2: Database Ingestion & Relational SQL Analysis
    print("\n[STEP 2/5] Initializing SQLite Database & running SQL queries...")
    db = CreditRiskDatabase()
    db.setup_database(csv_path)
    sql_insights = db.run_sql_analysis()
    
    for title, df_res in sql_insights.items():
        print(f"\n  SQL Query Output: {title}")
        print(df_res.head(6).to_string(index=False))

    # Fetch full joined analytical table from SQL DB
    df_sql = db.fetch_analytical_dataset()

    # Step 3: Exploratory Data Analysis & Matplotlib Visualizations
    print("\n[STEP 3/5] Generating Matplotlib EDA visual figures...")
    generate_eda_plots(df_sql)

    # Step 4: Machine Learning Model Training & Evaluation
    print("\n[STEP 4/5] Training Scikit-learn Classifier Models...")
    trainer = CreditRiskModelTrainer()
    metrics_summary = trainer.train_and_evaluate(df_sql)

    # Step 5: Inference Verification
    print("\n[STEP 5/5] Testing Applicant Inference Engine...")
    predictor = CreditRiskPredictor()
    sample_applicant = {
        'customer_id': 'TEST-CUSTOMER-99',
        'person_age': 28,
        'person_income': 45000,
        'person_home_ownership': 'RENT',
        'person_emp_length': 3.0,
        'loan_intent': 'DEBTCONSOLIDATION',
        'loan_grade': 'C',
        'loan_amnt': 15000,
        'loan_int_rate': 13.5,
        'loan_percent_income': 0.333,
        'dti': 32.5,
        'revolving_utilization': 65.0,
        'credit_score': 640,
        'cb_person_default_on_file': 'N',
        'cb_person_cred_hist_length': 5
    }
    pred_res = predictor.predict_applicant(sample_applicant)
    
    print("\n=================== PIPELINE EXECUTION COMPLETE ===================")
    print("Sample Inference Result:")
    for k, v in pred_res.items():
        print(f"  {k}: {v}")
    print("\nAll figures, models, database tables, and reports generated successfully!")

if __name__ == "__main__":
    run_pipeline()
