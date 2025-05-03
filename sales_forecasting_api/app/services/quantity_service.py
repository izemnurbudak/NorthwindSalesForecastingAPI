import pandas as pd
import joblib
from app.models.schemas import PredictRequest, RetrainRequest
import os

MODEL_PATH = "app/pipeline/quantity_model.pkl"

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

    from pipeline.quantity_model import train_quantity_model
    results = train_quantity_model(df)

    return {
        "message": "Quantity model successfully retrained.",
        "metrics": results
    }
