import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import joblib
import matplotlib.pyplot as plt
from app.services.visualization_service import plot_feature_importance, plot_decision_tree

def train_quantity_model(df):
    # Features and Target Variable
    X = df[["product_id", "unit_price", "month", "year", "TotalRevenue"]]
    y = df["quantity"]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    # Test predictions and metrics
    y_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Training predictions and metrics
    y_train_pred = model.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    
    # Save the model
    model_path = os.path.join("app", "models", "quantity_model.pkl")  # Save the model file in the models folder
    joblib.dump(model, model_path)
    
    # Visualizations
    plot_feature_importance(model, X_train.columns)  # Visualize feature importance
    plot_decision_tree(model, X_train.columns)  # Visualize the decision tree

    return {
        "test_r2": test_r2,
        "test_rmse": test_rmse,
        "train_r2": train_r2,
        "train_rmse": train_rmse
    }
