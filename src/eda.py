import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def generate_eda_plots(df, output_dir=os.path.join("reports", "figures")):
    """Generates and saves publication-quality Matplotlib EDA figures."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set Matplotlib style aesthetics
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 16
    })

    # Color Palette
    primary_color = '#1f77b4'
    default_color = '#d62728'
    non_default_color = '#2ca02c'
    palette = [non_default_color, default_color]

    print("Generating EDA visualizations with Matplotlib...")

    # 1. Loan Status Distribution Chart
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    counts = df['loan_status'].value_counts()
    labels = ['Non-Default (0)', 'Default (1)']
    
    ax[0].pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=palette, explode=(0, 0.08), shadow=True)
    ax[0].set_title('Overall Loan Default Ratio')
    
    sns.barplot(x=labels, y=counts.values, ax=ax[1], palette=palette, hue=labels, legend=False)
    ax[1].set_title('Customer Count by Default Status')
    ax[1].set_ylabel('Number of Borrowers')
    for i, v in enumerate(counts.values):
        ax[1].text(i, v + 100, f"{v:,}", ha='center', fontweight='bold')

    plt.tight_layout()
    fig1_path = os.path.join(output_dir, 'loan_status_distribution.png')
    plt.savefig(fig1_path, dpi=300)
    plt.close()

    # 2. Default Rate & Avg Interest Rate by Loan Grade
    fig, ax1 = plt.subplots(figsize=(10, 6))
    grade_df = df.groupby('loan_grade').agg(
        total=('loan_status', 'count'),
        defaults=('loan_status', 'sum'),
        default_rate=('loan_status', lambda x: x.mean() * 100),
        avg_int_rate=('loan_int_rate', 'mean')
    ).reset_index()

    ax2 = ax1.twinx()
    bars = sns.barplot(data=grade_df, x='loan_grade', y='default_rate', ax=ax1, color='#4c72b0', alpha=0.85)
    line = sns.lineplot(data=grade_df, x='loan_grade', y='avg_int_rate', ax=ax2, color='#c44e52', marker='o', linewidth=2.5, label='Avg Interest Rate (%)')

    ax1.set_title('Loan Default Rate & Interest Rate by Loan Grade (A - G)', pad=15)
    ax1.set_xlabel('Loan Grade')
    ax1.set_ylabel('Default Rate (%)', color='#4c72b0', fontweight='bold')
    ax2.set_ylabel('Average Interest Rate (%)', color='#c44e52', fontweight='bold')
    ax2.grid(False)

    for p in bars.patches:
        height = p.get_height()
        ax1.annotate(f'{height:.1f}%',
                     (p.get_x() + p.get_width() / 2., height / 2),
                     ha='center', va='center', color='white', fontweight='bold', fontsize=10)

    plt.tight_layout()
    fig2_path = os.path.join(output_dir, 'default_rate_by_grade.png')
    plt.savefig(fig2_path, dpi=300)
    plt.close()

    # 3. Income vs Loan Amount Scatter Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = sns.scatterplot(
        data=df, 
        x='person_income', 
        y='loan_amnt', 
        hue='loan_status', 
        palette={0: '#2ca02c', 1: '#d62728'},
        alpha=0.6, 
        s=30,
        ax=ax
    )
    ax.set_xscale('log')
    ax.set_title('Annual Income vs Loan Amount (Log Scale)', pad=15)
    ax.set_xlabel('Annual Income ($ - Log Scale)')
    ax.set_ylabel('Requested Loan Amount ($)')
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, ['Non-Default', 'Default'], title='Status', loc='upper left')

    plt.tight_layout()
    fig3_path = os.path.join(output_dir, 'income_vs_loan_amount.png')
    plt.savefig(fig3_path, dpi=300)
    plt.close()

    # 4. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='Blues', vmin=-1, vmax=1, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title('Financial Feature Correlation Matrix', pad=15)

    plt.tight_layout()
    fig4_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(fig4_path, dpi=300)
    plt.close()

    # 5. DTI Ratio & Credit Score Distribution
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.boxplot(data=df, x='loan_status', y='dti', palette={0: '#2ca02c', 1: '#d62728'}, ax=ax[0], hue='loan_status', legend=False)
    ax[0].set_title('Debt-to-Income (DTI) Ratio by Default Status')
    ax[0].set_xticks([0, 1])
    ax[0].set_xticklabels(['Non-Default', 'Default'])
    ax[0].set_xlabel('Loan Status')
    ax[0].set_ylabel('Debt-to-Income (%)')

    sns.kdeplot(data=df[df['loan_status']==0], x='credit_score', label='Non-Default', color='#2ca02c', fill=True, alpha=0.3, ax=ax[1])
    sns.kdeplot(data=df[df['loan_status']==1], x='credit_score', label='Default', color='#d62728', fill=True, alpha=0.3, ax=ax[1])
    ax[1].set_title('FICO Credit Score Density Distribution')
    ax[1].set_xlabel('Credit Score')
    ax[1].set_ylabel('Density')
    ax[1].legend()

    plt.tight_layout()
    fig5_path = os.path.join(output_dir, 'dti_vs_credit_score.png')
    plt.savefig(fig5_path, dpi=300)
    plt.close()

    print(f"All 5 EDA figures saved successfully to: {output_dir}")

if __name__ == "__main__":
    from database import CreditRiskDatabase
    db = CreditRiskDatabase()
    df = db.fetch_analytical_dataset()
    generate_eda_plots(df)
