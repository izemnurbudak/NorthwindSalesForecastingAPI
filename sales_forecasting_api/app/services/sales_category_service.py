import pandas as pd
from app.models.schemas import RetrainRequest

def retrain_sales_category_model(df: pd.DataFrame):
    if "Sales_Category" not in df.columns:
        raise ValueError("The target variable 'Sales_Category' is missing for training.")

    from pipeline.sales_category_model import train_sales_category_model
    results = train_sales_category_model(df)

    return {
        "message": "Sales category model successfully retrained.",
        "metrics": results
    }

