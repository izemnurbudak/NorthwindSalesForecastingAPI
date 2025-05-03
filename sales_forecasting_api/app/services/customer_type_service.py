import pandas as pd
from app.models.schemas import RetrainRequest

def retrain_customer_type_model(df: pd.DataFrame):
    if "Customer_Type" not in df.columns:
        raise ValueError("The 'Customer_Type' target variable is missing for training.")

    from pipeline.customer_type_model import train_customer_type_model
    results = train_customer_type_model(df)

    return {
        "message": "Customer type model successfully retrained.",
        "metrics": results
    }

