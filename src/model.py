import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, PrecisionRecallDisplay
)
from sklearn.pipeline import Pipeline
from preprocessing import get_preprocessor, prepare_model_data

class CreditRiskModelTrainer:
    def __init__(self, models_dir="models", figures_dir=os.path.join("reports", "figures")):
        self.models_dir = models_dir
        self.figures_dir = figures_dir
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=120, learning_rate=0.08, max_depth=5, random_state=42)
        }
        self.best_model_name = None
        self.best_pipeline = None
        self.best_score = 0.0
        self.results = {}

    def train_and_evaluate(self, df):
        """Trains multiple ML models and evaluates classification metrics."""
        X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_model_data(df)
        preprocessor = get_preprocessor(num_cols, cat_cols)
        
        print("\n=================== TRAINING ML MODELS ===================")
        
        for name, classifier in self.models.items():
            print(f"Training {name}...")
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', classifier)
            ])
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_prob)
            
            self.results[name] = {
                'pipeline': pipeline,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'y_pred': y_pred,
                'y_prob': y_prob
            }
            
            print(f"[{name}] Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")
            
            if roc_auc > self.best_score:
                self.best_score = roc_auc
                self.best_model_name = name
                self.best_pipeline = pipeline

        print(f"\nBest Model selected: {self.best_model_name} with ROC-AUC = {self.best_score:.4f}")
        
        # Save best model
        model_path = os.path.join(self.models_dir, 'credit_risk_best_pipeline.joblib')
        joblib.dump(self.best_pipeline, model_path)
        print(f"Saved best model pipeline to: {model_path}")
        
        # Generate Evaluation Plots
        self.plot_roc_curves(y_test)
        self.plot_confusion_matrix(y_test)
        self.plot_feature_importance(preprocessor, X_train)
        
        return pd.DataFrame(self.results).T.drop(columns=['pipeline', 'y_pred', 'y_prob'])

    def plot_roc_curves(self, y_test):
        """Plots ROC curves for all trained models."""
        plt.figure(figsize=(9, 6))
        for name, res in self.results.items():
            fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
            plt.plot(fpr, tpr, label=f"{name} (AUC = {res['roc_auc']:.3f})", linewidth=2)
            
        plt.plot([0, 1], [0, 1], 'k--', label='Random Chance (AUC = 0.500)')
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity / Recall)')
        plt.title('Receiver Operating Characteristic (ROC) Curves Comparison', pad=15)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'roc_curves.png'), dpi=300)
        plt.close()

    def plot_confusion_matrix(self, y_test):
        """Plots Confusion Matrix for the best performing model."""
        best_res = self.results[self.best_model_name]
        cm = confusion_matrix(y_test, best_res['y_pred'])
        
        plt.figure(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Non-Default', 'Default'],
                    yticklabels=['Non-Default', 'Default'])
        plt.title(f'Confusion Matrix - {self.best_model_name}', pad=15)
        plt.xlabel('Predicted Loan Status')
        plt.ylabel('Actual Loan Status')
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'confusion_matrix.png'), dpi=300)
        plt.close()

    def plot_feature_importance(self, preprocessor, X_train):
        """Extracts and plots feature importances from the best tree-based model."""
        clf = self.best_pipeline.named_steps['classifier']
        
        if hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
            
            # Extract feature names after One-Hot Encoding
            num_cols = preprocessor.transformers_[0][2]
            cat_cols = preprocessor.transformers_[1][2]
            cat_onehot_cols = list(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_cols))
            all_feature_names = num_cols + cat_onehot_cols
            
            feat_imp_df = pd.DataFrame({
                'Feature': all_feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False).head(12)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=feat_imp_df, x='Importance', y='Feature', palette='crest')
            plt.title(f'Top 12 Feature Importances - {self.best_model_name}', pad=15)
            plt.xlabel('Gini Importance Score')
            plt.tight_layout()
            plt.savefig(os.path.join(self.figures_dir, 'feature_importance.png'), dpi=300)
            plt.close()

if __name__ == "__main__":
    from database import CreditRiskDatabase
    db = CreditRiskDatabase()
    df = db.fetch_analytical_dataset()
    
    trainer = CreditRiskModelTrainer()
    metrics_df = trainer.train_and_evaluate(df)
    print("\n--- Model Performance Metrics Summary ---")
    print(metrics_df)
