
from fastapi import FastAPI
from app.api.routes import router as api_router


app = FastAPI(title="Sales Prediction API", description="Sales Prediction API using Models")

app.include_router(api_router)
