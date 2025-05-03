# app/models/schemas.py
from pydantic import BaseModel, Field
from typing import List

class PredictRequest(BaseModel):
    product_id: int = Field(..., gt=0)
    unit_price: float = Field(..., gt=0)
    month: int = Field(..., ge=1, le=12)
    year: int = Field(..., ge=2000)
    TotalRevenue: float = Field(..., gt=0)

class DataItem(BaseModel):
    product_id: int
    unit_price: float
    month: int
    year: int
    TotalRevenue: float
    quantity: int

class RetrainRequest(BaseModel):
    data: List[DataItem]
