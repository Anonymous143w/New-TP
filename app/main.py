from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from src.features.preprocessor import DataPreprocessor  # Import the DataPreprocessor class

app = FastAPI()

class MedicalCostInput(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str

# Load model and preprocessor
model = joblib.load("models\medical_cost_model.pkl")

# Load your config (assuming it's in YAML format)
import yaml
with open('config/config.yaml') as f:
    config = yaml.safe_load(f)

preprocessor = DataPreprocessor(config)  # Initialize preprocessor with the loaded config

@app.post("/predict")
async def predict_cost(input_data: MedicalCostInput):
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data.dict()])
        
        # Preprocess the input
        X, _ = preprocessor.transform_features(input_df)
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        return {"predicted_cost": float(prediction)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
