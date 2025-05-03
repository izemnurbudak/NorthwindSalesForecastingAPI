# app/api/routes.py
from fastapi import APIRouter, HTTPException
from app.models.schemas import PredictRequest, RetrainRequest, DataItem
from app.services.forecasting import (
    get_all_products,
    predict_quantity,
    retrain_sales_model,
    get_sales_summary_data
)

router = APIRouter()

@router.get("/products", tags=["Products"])
def get_products():
    try:
        return get_all_products()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch products: {str(e)}")

@router.post("/predict", tags=["Predicts"])
def predict_sales(request: PredictRequest):
    try:
        return predict_quantity(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/retrain", tags=["Retrain"])
def retrain_model(request: RetrainRequest):
    try:
        return retrain_sales_model(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model retraining failed: {str(e)}")

@router.get("/sales_summary", tags=["SalesSummary"])
def sales_summary():
    try:
        return get_sales_summary_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sales summary query failed: {str(e)}")
