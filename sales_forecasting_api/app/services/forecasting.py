import pandas as pd
import joblib
from sqlalchemy import create_engine
from app.models.schemas import PredictRequest, RetrainRequest
import os
from pipeline.quantity_model import train_quantity_model
from pipeline.sales_category_model import train_sales_category_model
from pipeline.customer_type_model import train_customer_type_model

# Get database connection from environment variable
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:yourpassword@localhost/yourdbname")
engine = create_engine(DATABASE_URL)

MODEL_PATH = "apppipelinequantity_model.pkl"

# Quantity Prediction

def get_all_products():
    query = """
    SELECT product_id, product_name, unit_price FROM products;
    """
    df_products = pd.read_sql(query, engine)
    return df_products.to_dict(orient="records")

def predict_quantity(request: PredictRequest):
    model = joblib.load(MODEL_PATH)

    input_data = pd.DataFrame([{
        "product_id": request.product_id,
        "unit_price": request.unit_price,
        "month": request.month,
        "year": request.year,
        "TotalRevenue": request.TotalRevenue
    }])

    prediction = model.predict(input_data)[0]

    return {
        "predicted_quantity": prediction,
        "input_details": input_data.to_dict(orient="records")[0]
    }

def retrain_sales_model(request: RetrainRequest):
    df = pd.DataFrame([{
        "product_id": item.product_id,
        "unit_price": item.unit_price,
        "month": item.month,
        "year": item.year,
        "TotalRevenue": item.TotalRevenue,
        "quantity": item.quantity
    } for item in request.data])

    if "quantity" not in df.columns:
        raise ValueError("The 'quantity' (target variable) is missing for training.")

    results = train_quantity_model(df)

    return {
        "message": "Quantity model successfully retrained.",
        "metrics": results
    }

# Sales Category Classification

def retrain_sales_category_model(df: pd.DataFrame):
    if "Sales_Category" not in df.columns:
        raise ValueError("The 'Sales_Category' target variable is missing for training.")

    results = train_sales_category_model(df)

    return {
        "message": "Sales category model successfully retrained.",
        "metrics": results
    }

# Customer Type Classification

def retrain_customer_type_model(df: pd.DataFrame):
    if "Customer_Type" not in df.columns:
        raise ValueError("The 'Customer_Type' target variable is missing for training.")

    results = train_customer_type_model(df)

    return {
        "message": "Customer type model successfully retrained.",
        "metrics": results
    }

def get_sales_summary_data():
    query = """
    SELECT
        COUNT(DISTINCT o.order_id) AS total_orders,
        SUM(od.quantity) AS total_units_sold,
        ROUND(CAST(SUM(od.quantity * od.unit_price * (1 - od.discount)) AS NUMERIC), 2) AS total_revenue
    FROM
        orders o
    JOIN
        order_details od ON o.order_id = od.order_id;
    """
    
    df_sales = pd.read_sql(query, engine)

    if df_sales.empty:
        raise ValueError("No sales data found.")

    return df_sales.to_dict(orient="records")
