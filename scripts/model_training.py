import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib

def load_data(filepath='../data/engineered_dataset.csv'):
    """Load the fully engineered dataset."""
    df = pd.read_csv(filepath)
    return df

def train_and_evaluate(df):
    """Split the data, train the model, and evaluate accuracy."""
    print("Splitting datasets for training and testing...")
    X = df.drop('default_status', axis=1)
    y = df['default_status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training shapes: {X_train.shape}, limits testing shapes: {X_test.shape}")
    
    # We will use Random Forest as the primary model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    print("\nTraining Random Forest model...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("\nEvaluation Results:")
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Save model
    joblib.dump(model, 'credit_risk_model.pkl')
    print("Model saved to scripts/credit_risk_model.pkl")

if __name__ == "__main__":
    df = load_data()
    train_and_evaluate(df)
