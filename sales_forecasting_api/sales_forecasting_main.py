import warnings

from sklearn.tree import DecisionTreeClassifier
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
from pipeline.customer_type_model import train_customer_type_model
from pipeline.load_data import load_data
from pipeline.feature_engineering import feature_engineering
from pipeline.quantity_model import train_quantity_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pipeline.sales_category_model import train_sales_category_model

def main():
    # 1. Load the data
    df, orders, order_details, products, categories, customers = load_data()
    print(df.head())
    print(df.shape)
    print("CSV loaded")

    # 2. Feature engineering
    df = feature_engineering(df)
    print(df.head())

    # 3. Print target class distribution
    print(df["Sales_Category"].value_counts(normalize=True))

    # 4. Train the model for Sales Category prediction (classification example)
    X = df[["product_id", "unit_price", "month", "year", "TotalRevenue"]]
    y = df["Sales_Category"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred, digits=2)
    print(report)

    # 5. Train the Quantity prediction model (regression)
    print("\n[Quantity Model - Decision Tree Regressor]")
    train_quantity_model(df)
     
    # 5. Train the Sales Category prediction model (classification)
    print("\n[Sales Category Model - Random Forest and Decision Tree]")
    train_sales_category_model(df)

    # 6. Train the Customer Type prediction model (classification)
    print("\n[Customer Type Model - Random Forest, Decision Tree, KNN]")
    train_customer_type_model(df)

if __name__ == "__main__":
    main()
