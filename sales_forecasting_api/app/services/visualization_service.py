# visualization.py
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance(model, feature_names):
    feature_importances = model.feature_importances_

    # Convert feature names and importances into a DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })

    # Sort by importance
    importance_df = importance_df.sort_values(by="Importance", ascending=False)
    
    # Visualization
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis")
    plt.title("Feature Importance")
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show()

def plot_decision_tree(model, feature_names):
    plt.figure(figsize=(15, 10))
    plot_tree(model, filled=True, feature_names=feature_names, rounded=True)
    plt.title("Decision Tree Visualization")
    plt.tight_layout()
    plt.show()


def plot_top_selling_products(sales_summary):
    top_products = sales_summary.groupby("product_name")["quantity"].sum().sort_values(ascending=False).head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_products.values, y=top_products.index, palette="viridis")
    plt.title("Top 10 Selling Products")
    plt.xlabel("Total Quantity Sold")
    plt.ylabel("Product")
    plt.tight_layout()
    plt.show()

def plot_monthly_sales(df):
    df["order_month"] = df["order_date"].dt.to_period("M").dt.to_timestamp()
    monthly_sales = df.groupby("order_month")["quantity"].sum().reset_index()
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=monthly_sales, x="order_month", y="quantity", marker="o")
    plt.title("Monthly Sales Over Time")
    plt.xlabel("Date")
    plt.ylabel("Sales Quantity")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("Actual Labels")
    plt.tight_layout()
    plt.show()

def plot_classification_metrics(metrics):
    labels = list(metrics.keys())
    accuracies = [metrics[label]["accuracy"] for label in labels]
    precisions = [metrics[label]["precision"] for label in labels]
    recalls = [metrics[label]["recall"] for label in labels]
    f1_scores = [metrics[label]["f1_score"] for label in labels]
    
    x = np.arange(len(labels))
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - 0.2, accuracies, 0.4, label="Accuracy")
    plt.bar(x + 0.2, precisions, 0.4, label="Precision")
    plt.bar(x - 0.2, recalls, 0.4, label="Recall")
    plt.bar(x + 0.2, f1_scores, 0.4, label="F1 Score")
    
    plt.xticks(x, labels, rotation=45)
    plt.title("Model Performance Metrics")
    plt.xlabel("Models")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.show()
