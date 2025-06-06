from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model
model = joblib.load("heart_disease_model.pkl")

# Define input schema
class PatientData(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.get("/health")
def health_check():
    return {"status": "API is running"}

@app.post("/predict")
def predict(data: PatientData):
    try:
        input_array = np.array([[ 
            data.age, data.sex, data.cp, data.trestbps, data.chol,
            data.fbs, data.restecg, data.thalach, data.exang, data.oldpeak,
            data.slope, data.ca, data.thal
        ]])
        prediction = model.predict(input_array)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
