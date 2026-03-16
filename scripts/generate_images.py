import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create images directory if it doesn't exist
os.makedirs('../images', exist_ok=True)

# Load data
try:
    df = pd.read_csv('../data/sample_dataset.csv')
except FileNotFoundError:
    print("Run this script from the scripts directory, or ensure sample_dataset.csv is created.")
    exit()

# Stylings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("muted")

# 1. Default rate by income
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='income', hue='default_status', multiple='stack', bins=10)
plt.title('Default Rate by Income', fontsize=16)
plt.xlabel('Annual Income', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Default Status', labels=['Yes', 'No'])
plt.savefig('../images/default_rate_by_income.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Loan amount vs default risk
plt.figure(figsize=(10, 6))
sns.boxplot(x='default_status', y='loan_amount', data=df)
plt.title('Loan Amount vs Default Risk', fontsize=16)
plt.xlabel('Default Status (0 = No, 1 = Yes)', fontsize=12)
plt.ylabel('Loan Amount', fontsize=12)
plt.savefig('../images/loan_amount_vs_default.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Credit history impact
plt.figure(figsize=(10, 6))
sns.violinplot(x='default_status', y='credit_history', data=df)
plt.title('Credit History Impact on Default Risk', fontsize=16)
plt.xlabel('Default Status (0 = No, 1 = Yes)', fontsize=12)
plt.ylabel('Credit History Score', fontsize=12)
plt.savefig('../images/credit_history_impact.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Risk segmentation charts
df['risk_segment'] = pd.cut(df['debt_ratio'], bins=[0, 0.2, 0.4, 1.0], labels=['Low', 'Medium', 'High'])
plt.figure(figsize=(10, 6))
sns.countplot(x='risk_segment', hue='default_status', data=df)
plt.title('Risk Segmentation by Debt Ratio', fontsize=16)
plt.xlabel('Risk Segment (Debt Ratio)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Default Status', labels=['Yes', 'No'])
plt.savefig('../images/risk_segmentation.png', dpi=300, bbox_inches='tight')
plt.close()

# Placeholder header
plt.figure(figsize=(12, 3))
plt.text(0.5, 0.5, 'Credit Risk Analysis', fontsize=30, ha='center', va='center', fontweight='bold', color='navy')
plt.axis('off')
plt.savefig('../images/header.png', dpi=300, bbox_inches='tight')
plt.close()

print("Images generated successfully in the 'images/' folder.")
