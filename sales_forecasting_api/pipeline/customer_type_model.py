# customer_type_model.py
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import joblib
import os
from app.services.visualization_service import plot_confusion_matrix, plot_classification_metrics  # Importing visualization functions

def train_customer_type_model(df):
    # Target Variable and Features
    df['Target'] = np.where(df['Customer_Type'] == 'New Customer', 1, 0)
    features = ['quantity', 'month', 'year', 'TotalRevenue']
    X = df[features]
    y = df['Target']
    
    # Train and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Models
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5, metric='minkowski')
    }

    # Dictionary to store results
    results = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results[model_name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "classification_report": classification_report(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred)
        }
        
        # Confusion Matrix visualization
        cm = results[model_name]["confusion_matrix"]
        plot_confusion_matrix(cm, labels=["Old Customer", "New Customer"])
    
    # Visualizing performance metrics
    plot_classification_metrics(results)

    # Saving models
    for model_name, model in models.items():
        model_path = os.path.join("app", "models", f"{model_name.lower().replace(' ', '_')}_customer_type_model.pkl")
        joblib.dump(model, model_path)

    return results
