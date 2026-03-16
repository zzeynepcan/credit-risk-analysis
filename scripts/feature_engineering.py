import pandas as pd
import numpy as np

def calculate_dti(df):
    """Calculate Debt-to-Income (DTI). This is a huge indicator for banks."""
    print("Feature Extraction: Calculating DTI ratio...")
    # Just a quick check to avoid any weird division by zero errors
    df['dti'] = np.where(df['income'] == 0, 0, df['debt_ratio'] / df['income'])
    return df

def calculate_credit_utilization(df):
    """Simulate a Credit Utilization ratio based on available columns."""
    print("Calculating Credit Utilization...")
    # Assuming debt_ratio represents total debt and we have a pseudo credit_history scaling
    df['credit_utilization'] = df['debt_ratio'] / (df['credit_history'] + 1)
    return df

def encode_categorical(df):
    """One-hot encode categorical features like employment_length."""
    print("Encoding categorical variables...")
    categorical_cols = ['employment_length']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df

if __name__ == "__main__":
    print("Loading cleaned dataset...")
    df = pd.read_csv('../data/cleaned_dataset.csv')
    df = calculate_dti(df)
    df = calculate_credit_utilization(df)
    df = encode_categorical(df)
    
    df.to_csv('../data/engineered_dataset.csv', index=False)
    print("Feature engineering complete. Data saved to 'data/engineered_dataset.csv'")
