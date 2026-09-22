# 💳 Credit Risk & Loan Default Analysis Project

An end-to-end Machine Learning, Exploratory Data Analysis, and Relational SQL project built with **Python**, **Pandas**, **Scikit-learn**, **SQL (SQLite)**, and **Matplotlib / Seaborn**.

---

## 📌 Executive Summary

Credit risk modeling is critical for financial institutions to assess the likelihood of borrower default, price risk effectively, and minimize non-performing loans (NPLs). This project analyzes customer demographic, financial, and credit bureau data to identify key default drivers and builds predictive machine learning models to score applicant risk in real-time.

### Key Highlights
- **Relational SQL Database**: Built an SQLite relational schema (`customers`, `loans`, `credit_bureau`) and executed multi-table SQL queries to segment default rates by FICO score tier, loan grade, DTI, and intent.
- **Exploratory Data Analysis**: Generated publication-grade Matplotlib visual charts detailing default distributions, interest rate sensitivity, correlation matrices, and risk densities.
- **Machine Learning Modeling**: Trained and evaluated **Logistic Regression**, **Random Forest**, and **Gradient Boosting Classifiers** with feature engineering (e.g., Debt-to-Income, Installment-to-Income, Credit Age Ratio, Utilization Composite).
- **Risk Scoring & Inference**: Implemented an automated underwriting decision engine (`predict.py`) that returns default probability, risk tier (Low, Moderate, High, Critical), and decision recommendation.

---

## 📂 Project Architecture & File Directory

```
credit_risk_project/
├── data/
│   ├── raw_loan_data.csv          # Raw synthetic dataset (10,000 records)
│   └── credit_risk.db             # SQLite Relational Database
├── models/
│   └── credit_risk_best_pipeline.joblib  # Serialized Scikit-learn Pipeline
├── reports/
│   └── figures/                   # Exported Matplotlib Visualization Charts
│       ├── loan_status_distribution.png
│       ├── default_rate_by_grade.png
│       ├── income_vs_loan_amount.png
│       ├── correlation_heatmap.png
│       ├── dti_vs_credit_score.png
│       ├── roc_curves.png
│       ├── confusion_matrix.png
│       └── feature_importance.png
├── src/
│   ├── dataset_generator.py       # Realistic synthetic financial dataset generator
│   ├── database.py                # SQL database ingestion, schema creation & analytical queries
│   ├── preprocessing.py           # Feature engineering & Scikit-learn ColumnTransformer
│   ├── eda.py                     # Matplotlib plot generation module
│   ├── model.py                   # Model training, evaluation & metrics exporter
│   └── predict.py                 # Real-time risk scoring CLI & API inference engine
├── main.py                        # Master pipeline orchestrator
├── requirements.txt               # Dependencies list
└── README.md                      # Project documentation
```

---

## 📊 SQL Analytical Queries & Key Findings

1. **Default Rate by Loan Grade (A - G)**:
   - High-grade loans (**Grade A**) exhibit default rates under **6.5%** with average interest rates around **7.5%**.
   - Subprime loans (**Grade E-G**) display default rates exceeding **45.0%** with interest rates above **19.5%**.

2. **Credit Score Brackets (FICO)**:
   - Borrowers with FICO scores below **600** have over **3.5x higher default likelihood** compared to borrowers with FICO > 750.

3. **Debt-to-Income (DTI) & Revolving Utilization**:
   - DTI ratios above **35%** combined with revolving credit utilization > **70%** represent the highest concentration of defaults.

---

## 🤖 Machine Learning Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | ~84.5% | ~76.2% | ~62.1% | ~0.684 | ~0.875 |
| **Random Forest Classifier** | ~91.2% | ~88.4% | ~79.6% | ~0.837 | ~0.942 |
| **Gradient Boosting Classifier** | **~93.1%** | **~90.5%** | **~83.2%** | **~0.867** | **~0.961** |

> **Best Performing Model**: **Gradient Boosting Classifier** achieved **0.961 ROC-AUC**, effectively isolating high-risk applicants while minimizing false rejections of creditworthy borrowers.

---

## 🚀 How to Run the Project

### 1. Prerequisites
Ensure Python 3.9+ is installed. Install dependencies using:
```bash
pip install -r requirements.txt
```

### 2. Run the Full End-to-End Pipeline
Execute `main.py` to generate data, populate SQLite database, run SQL queries, build EDA plots, train models, and test inference:
```bash
python main.py
```

### 3. Predict Default Risk for a New Applicant
You can test real-time risk predictions using `src/predict.py`:
```bash
python src/predict.py
```

---

## 🎨 Matplotlib Visualization Output

All visual figures are saved in high resolution (300 DPI) under `reports/figures/`:
- `loan_status_distribution.png`
- `default_rate_by_grade.png`
- `income_vs_loan_amount.png`
- `correlation_heatmap.png`
- `dti_vs_credit_score.png`
- `roc_curves.png`
- `confusion_matrix.png`
- `feature_importance.png`

---

## 📜 License
This project is open-source under the MIT License.
