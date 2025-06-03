# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Load the model pipeline
model = joblib.load("model_pipeline.pkl")

# Define FastAPI app
app = FastAPI(title="House Price Prediction API")

# Define input schema
class HouseFeatures(BaseModel):
    OverallQual: int
    TotalSF: float
    YearBuilt: int
    GrLivArea: float
    GarageCars: int
    GarageArea: int
    YearRemodAdd: int
    TotalBath: float
    LotArea: float
    OverallCond: int
    FirstFlrSF: float  # renamed to avoid invalid variable name starting with a digit
    BsmtFinSF1: float
    SecondFlrSF: float  # renamed to avoid invalid variable name starting with a digit
    TotalBsmtSF: float
    BsmtUnfSF: float
    LotFrontage: float
    GarageYrBlt: float
    Fireplaces: int
    TotalPorchSF: float
    MoSold: int
    OpenPorchSF: float
    MSSubClass: float
    MasVnrArea: float
    WoodDeckSF: float
    YrSold: int

@app.post("/predict")
def predict_price(data: HouseFeatures):
    input_data = pd.DataFrame([{col: getattr(data, col) for col in data.__annotations__}])
    prediction = model.predict(input_data)
    return {"predicted_price": prediction[0]}
