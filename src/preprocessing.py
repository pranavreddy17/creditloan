import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def feature_engineering(df):
    """Adds calculated domain-specific financial features."""
    df_feat = df.copy()
    
    # 1. Estimated monthly installment (36-month term assumption)
    monthly_rate = (df_feat['loan_int_rate'].fillna(df_feat['loan_int_rate'].median()) / 100) / 12
    # Simple loan payment approximation
    df_feat['est_monthly_installment'] = (df_feat['loan_amnt'] * (1 + monthly_rate * 36)) / 36
    
    # 2. Installment to monthly income ratio
    monthly_income = df_feat['person_income'] / 12
    df_feat['installment_to_income_ratio'] = np.where(monthly_income > 0, df_feat['est_monthly_installment'] / monthly_income, 0)
    
    # 3. Credit history length ratio to age
    df_feat['cred_hist_to_age_ratio'] = df_feat['cb_person_cred_hist_length'] / np.maximum(df_feat['person_age'], 18)
    
    # 4. Debt Utilization Composite Score
    df_feat['risk_utilization_composite'] = (df_feat['dti'] * df_feat['revolving_utilization']) / 100.0
    
    return df_feat

def get_preprocessor(numeric_features, categorical_features):
    """Constructs a Scikit-Learn ColumnTransformer preprocessor pipeline."""
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, numeric_features),
            ('cat', cat_pipeline, categorical_features)
        ]
    )
    
    return preprocessor

def prepare_model_data(df, target_col='loan_status', test_size=0.2, random_state=42):
    """Performs feature engineering and splits data into Train & Test sets."""
    df_engineered = feature_engineering(df)
    
    drop_cols = ['customer_id', target_col]
    X = df_engineered.drop(columns=[col for col in drop_cols if col in df_engineered.columns])
    y = df_engineered[target_col]
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, numeric_features, categorical_features

if __name__ == "__main__":
    from database import CreditRiskDatabase
    db = CreditRiskDatabase()
    df = db.fetch_analytical_dataset()
    X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_model_data(df)
    print(f"Dataset split complete: Train shape={X_train.shape}, Test shape={X_test.shape}")
    print(f"Numerical features ({len(num_cols)}): {num_cols}")
    print(f"Categorical features ({len(cat_cols)}): {cat_cols}")
