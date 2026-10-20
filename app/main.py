from fastapi import FastAPI
from pydantic import BaseModel
import joblib


# --------------------------------------------------
# 1. Create the FastAPI application
# --------------------------------------------------

app = FastAPI()

@app.get("/version")
def get_version():
    return {"version": "2.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}

# --------------------------------------------------
# 2. Load the trained ML model
# --------------------------------------------------

model = joblib.load("model/approval_model.joblib")


# --------------------------------------------------
# 3. Define the input structure
# --------------------------------------------------

class PredictionRequest(BaseModel):
    age: int
    salary: float


# --------------------------------------------------
# 4. Create the prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(request: PredictionRequest):

    prediction = model.predict([
        [request.age, request.salary]
    ])

    return {
        "prediction": int(prediction[0])
    }