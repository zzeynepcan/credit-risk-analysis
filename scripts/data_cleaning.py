import pandas as pd
import numpy as np

def load_data(filepath='../data/sample_dataset.csv'):
    """Load the dataset from the specified file path."""
    try:
        df = pd.read_csv(filepath)
        print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except FileNotFoundError:
        print(f"Error: Dataset not found at {filepath}")
        return None

def handle_missing_values(df):
    """Fill or drop missing values appropriately depending on the column type."""
    print("Handling missing values...")
    # Numerical data imputation with median
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
            
    # Categorical data imputation with mode
    for col in df.select_dtypes(exclude=[np.number]).columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
            
    print("Missing values handled.")
    return df

def remove_anomalies(df):
    """Remove statistical anomalies (outliers) based on Interquartile Range (IQR)."""
    print("Removing anomalies...")
    numerical_cols = ['income', 'loan_amount', 'debt_ratio']
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        initial_shape = df.shape[0]
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
    print(f"Anomalies removed. Final dataset size: {df.shape[0]} rows.")
    return df

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        df = handle_missing_values(df)
        df = remove_anomalies(df)
        df.to_csv('../data/cleaned_dataset.csv', index=False)
        print("Cleaned data saved to 'data/cleaned_dataset.csv'")
