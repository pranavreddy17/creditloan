import sqlite3
import pandas as pd
import os

class CreditRiskDatabase:
    def __init__(self, db_path=os.path.join("data", "credit_risk.db")):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def setup_database(self, csv_path=os.path.join("data", "raw_loan_data.csv")):
        """Ingests raw CSV data and populates a relational SQLite database schema."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Source file {csv_path} not found. Run dataset_generator.py first.")

        df = pd.read_csv(csv_path)

        # Split into normalized relational tables
        customers_df = df[['customer_id', 'person_age', 'person_income', 'person_home_ownership', 'person_emp_length']].drop_duplicates()
        loans_df = df[['customer_id', 'loan_intent', 'loan_grade', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'loan_status']]
        bureau_df = df[['customer_id', 'dti', 'revolving_utilization', 'credit_score', 'cb_person_default_on_file', 'cb_person_cred_hist_length']]

        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create Schema
            cursor.execute("DROP TABLE IF EXISTS customers;")
            cursor.execute("DROP TABLE IF EXISTS loans;")
            cursor.execute("DROP TABLE IF EXISTS credit_bureau;")

            cursor.execute("""
            CREATE TABLE customers (
                customer_id TEXT PRIMARY KEY,
                person_age INTEGER,
                person_income REAL,
                person_home_ownership TEXT,
                person_emp_length REAL
            );
            """)

            cursor.execute("""
            CREATE TABLE loans (
                loan_id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT,
                loan_intent TEXT,
                loan_grade TEXT,
                loan_amnt REAL,
                loan_int_rate REAL,
                loan_percent_income REAL,
                loan_status INTEGER,
                FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
            );
            """)

            cursor.execute("""
            CREATE TABLE credit_bureau (
                customer_id TEXT PRIMARY KEY,
                dti REAL,
                revolving_utilization REAL,
                credit_score INTEGER,
                cb_person_default_on_file TEXT,
                cb_person_cred_hist_length INTEGER,
                FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
            );
            """)

            # Insert Data
            customers_df.to_sql('customers', conn, if_exists='append', index=False)
            loans_df.to_sql('loans', conn, if_exists='append', index=False)
            bureau_df.to_sql('credit_bureau', conn, if_exists='append', index=False)
            
            conn.commit()
            print(f"Database schema created successfully at: {self.db_path}")

    def run_sql_analysis(self):
        """Executes analytical SQL queries to extract key credit risk metrics."""
        queries = {
            "Default Rate by Loan Grade": """
                SELECT 
                    loan_grade,
                    COUNT(*) AS total_loans,
                    SUM(loan_status) AS total_defaults,
                    ROUND(AVG(loan_status) * 100, 2) AS default_rate_pct,
                    ROUND(AVG(loan_int_rate), 2) AS avg_interest_rate,
                    ROUND(AVG(loan_amnt), 2) AS avg_loan_amount
                FROM loans
                GROUP BY loan_grade
                ORDER BY loan_grade;
            """,
            
            "Default Rate by Credit Score Tier": """
                SELECT 
                    CASE 
                        WHEN cb.credit_score >= 750 THEN '1. Excellent (750+)'
                        WHEN cb.credit_score >= 700 THEN '2. Good (700-749)'
                        WHEN cb.credit_score >= 650 THEN '3. Fair (650-699)'
                        WHEN cb.credit_score >= 600 THEN '4. Poor (600-649)'
                        ELSE '5. Critical (<600)'
                    END AS credit_tier,
                    COUNT(*) AS total_borrowers,
                    SUM(l.loan_status) AS total_defaults,
                    ROUND(AVG(l.loan_status) * 100, 2) AS default_rate_pct,
                    ROUND(AVG(cb.dti), 2) AS avg_dti
                FROM credit_bureau cb
                JOIN loans l ON cb.customer_id = l.customer_id
                GROUP BY credit_tier
                ORDER BY credit_tier;
            """,
            
            "Loan Intent & Home Ownership Default Matrix": """
                SELECT 
                    c.person_home_ownership,
                    l.loan_intent,
                    COUNT(*) AS loan_count,
                    ROUND(AVG(l.loan_status) * 100, 2) AS default_rate_pct,
                    ROUND(AVG(l.loan_percent_income) * 100, 2) AS avg_loan_to_income_pct
                FROM customers c
                JOIN loans l ON c.customer_id = l.customer_id
                GROUP BY c.person_home_ownership, l.loan_intent
                HAVING COUNT(*) > 50
                ORDER BY default_rate_pct DESC;
            """,
            
            "High Risk Debt-to-Income & Prior Default Analysis": """
                SELECT 
                    cb.cb_person_default_on_file AS prior_default_on_file,
                    COUNT(*) AS borrower_count,
                    ROUND(AVG(l.loan_status) * 100, 2) AS default_rate_pct,
                    ROUND(AVG(cb.revolving_utilization), 2) AS avg_revolving_utilization,
                    ROUND(AVG(cb.dti), 2) AS avg_dti
                FROM credit_bureau cb
                JOIN loans l ON cb.customer_id = l.customer_id
                GROUP BY cb.cb_person_default_on_file;
            """
        }

        results = {}
        with self.get_connection() as conn:
            for title, query in queries.items():
                df_res = pd.read_sql_query(query, conn)
                results[title] = df_res

        return results

    def fetch_analytical_dataset(self):
        """Fetches the full joined analytical dataset via SQL query."""
        query = """
            SELECT 
                c.customer_id,
                c.person_age,
                c.person_income,
                c.person_home_ownership,
                c.person_emp_length,
                l.loan_intent,
                l.loan_grade,
                l.loan_amnt,
                l.loan_int_rate,
                l.loan_percent_income,
                cb.dti,
                cb.revolving_utilization,
                cb.credit_score,
                cb.cb_person_default_on_file,
                cb.cb_person_cred_hist_length,
                l.loan_status
            FROM customers c
            JOIN loans l ON c.customer_id = l.customer_id
            JOIN credit_bureau cb ON c.customer_id = cb.customer_id;
        """
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn)

if __name__ == "__main__":
    db = CreditRiskDatabase()
    db.setup_database()
    insights = db.run_sql_analysis()
    
    print("\n=================== SQL ANALYTICAL INSIGHTS ===================")
    for title, df in insights.items():
        print(f"\n--- {title} ---")
        print(df.to_string(index=False))
