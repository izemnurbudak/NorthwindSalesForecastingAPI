# sales_category_model.py
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
from app.services.visualization_service import plot_classification_metrics, plot_feature_importance, plot_decision_tree  # Import visualization functions

def train_sales_category_model(df):
    # Features and Target Variable
    X = df[["product_id", "unit_price", "month", "year", "TotalRevenue"]]
    y = df['Sales_Category']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Models
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }

    results = {}

    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Save the model
        model_path = os.path.join("app", "models", f"{model_name.lower().replace(' ', '_')}_sales_category_model.pkl")
        joblib.dump(model, model_path)
        
        # Calculate performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Save results
        results[model_name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "classification_report": classification_report(y_test, y_pred)
        }

        # Visualize feature importance (only for RandomForest)
        if model_name == "Random Forest":
            plot_feature_importance(model, X.columns)  # Call the visualization externally
        
        # Visualize Decision Tree (only for DecisionTree)
        if model_name == "Decision Tree":
            plot_decision_tree(model, X.columns)
    
    # Visualize performance metrics
    plot_classification_metrics(results)
    return results
