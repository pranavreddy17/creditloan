import numpy as np
import pandas as pd
import os

def generate_loan_dataset(n_samples=10000, random_state=42):
    """
    Generates a realistic synthetic credit risk dataset with financial attributes
    and non-linear interactions leading to loan default probabilities.
    """
    np.random.seed(random_state)
    
    customer_ids = [f"CUST-{10000 + i}" for i in range(n_samples)]
    
    # Demographics & Income
    age = np.random.randint(20, 70, size=n_samples)
    
    # Log-normal distribution for income (realistic skew)
    person_income = np.random.lognormal(mean=10.8, sigma=0.6, size=n_samples).astype(int)
    person_income = np.clip(person_income, 15000, 350000)
    
    person_home_ownership = np.random.choice(
        ['RENT', 'MORTGAGE', 'OWN', 'OTHER'],
        size=n_samples,
        p=[0.48, 0.40, 0.10, 0.02]
    )
    
    person_emp_length = np.random.exponential(scale=5.0, size=n_samples).round(1)
    person_emp_length = np.minimum(person_emp_length, age - 18)
    person_emp_length = np.clip(person_emp_length, 0, 45)
    
    # Loan Details
    loan_intent = np.random.choice(
        ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'],
        size=n_samples,
        p=[0.20, 0.22, 0.18, 0.17, 0.13, 0.10]
    )
    
    # Loan Grade (A to G)
    loan_grade = np.random.choice(
        ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
        size=n_samples,
        p=[0.30, 0.32, 0.18, 0.11, 0.05, 0.03, 0.01]
    )
    
    # Loan Amount based on income and intent
    loan_amnt = (np.random.gamma(shape=3.0, scale=3000, size=n_samples)).astype(int)
    loan_amnt = np.clip(loan_amnt, 1000, 40000)
    
    # Base Interest Rate based on Loan Grade
    grade_int_map = {'A': 7.5, 'B': 10.5, 'C': 13.5, 'D': 16.5, 'E': 19.5, 'F': 22.0, 'G': 24.5}
    base_int_rate = np.array([grade_int_map[g] for g in loan_grade])
    loan_int_rate = np.round(base_int_rate + np.random.normal(0, 1.2, size=n_samples), 2)
    loan_int_rate = np.clip(loan_int_rate, 5.0, 29.99)
    
    # Credit History & Bureau Attributes
    cb_person_cred_hist_length = np.clip((age - 18 - np.random.randint(0, 5, size=n_samples)), 1, 35)
    
    # FICO Credit Score (300 to 850)
    # Higher income/emp_length gives slightly higher FICO, lower loan grade gives lower FICO
    grade_fico_base = {'A': 750, 'B': 700, 'C': 650, 'D': 610, 'E': 570, 'F': 530, 'G': 500}
    fico_base = np.array([grade_fico_base[g] for g in loan_grade])
    credit_score = np.round(fico_base + np.random.normal(0, 35, size=n_samples)).astype(int)
    credit_score = np.clip(credit_score, 300, 850)
    
    # Historical default flag on credit bureau
    cb_person_default_on_file = np.where(credit_score < 620, np.random.choice(['Y', 'N'], size=n_samples, p=[0.45, 0.55]),
                                         np.random.choice(['Y', 'N'], size=n_samples, p=[0.08, 0.92]))
    
    # Debt-to-Income (DTI) Ratio and Loan-to-Income Ratio
    loan_percent_income = np.round(loan_amnt / person_income, 3)
    dti = np.round(np.random.beta(a=2, b=5, size=n_samples) * 50 + (loan_percent_income * 20), 2)
    dti = np.clip(dti, 2.0, 65.0)
    
    revolving_utilization = np.clip(np.random.beta(a=2, b=3, size=n_samples) * 100 + (850 - credit_score) * 0.1, 0.0, 100.0).round(2)
    
    # Calculate Default Probability based on risk formula (logistic function)
    # Factors increasing default: high DTI, low credit score, high interest rate, high loan_percent_income, historical default
    log_odds = (
        -4.5
        + 0.06 * (loan_int_rate)
        + 3.2 * (loan_percent_income)
        + 0.04 * (dti)
        + 0.03 * (revolving_utilization)
        - 0.008 * (credit_score - 600)
        - 0.04 * (person_emp_length)
        + (1.2 if 'Y' else 0.0) * np.where(cb_person_default_on_file == 'Y', 1, 0)
        + np.where(person_home_ownership == 'RENT', 0.4, 0.0)
        - np.where(person_home_ownership == 'OWN', 0.5, 0.0)
    )
    
    prob_default = 1 / (1 + np.exp(-log_odds))
    loan_status = (np.random.uniform(0, 1, size=n_samples) < prob_default).astype(int)
    
    # Introduce a small, realistic percentage of missing values (1-3%)
    mask_emp = np.random.rand(n_samples) < 0.02
    person_emp_length = person_emp_length.astype(object)
    person_emp_length[mask_emp] = np.nan
    
    mask_int = np.random.rand(n_samples) < 0.03
    loan_int_rate = loan_int_rate.astype(object)
    loan_int_rate[mask_int] = np.nan

    df = pd.DataFrame({
        'customer_id': customer_ids,
        'person_age': age,
        'person_income': person_income,
        'person_home_ownership': person_home_ownership,
        'person_emp_length': person_emp_length,
        'loan_intent': loan_intent,
        'loan_grade': loan_grade,
        'loan_amnt': loan_amnt,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_percent_income,
        'dti': dti,
        'revolving_utilization': revolving_utilization,
        'credit_score': credit_score,
        'cb_person_default_on_file': cb_person_default_on_file,
        'cb_person_cred_hist_length': cb_person_cred_hist_length,
        'loan_status': loan_status
    })
    
    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_loan_dataset(10000)
    csv_path = os.path.join("data", "raw_loan_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Dataset generated successfully with {len(df)} records at: {csv_path}")
    print(f"Default rate: {df['loan_status'].mean():.2%}")
