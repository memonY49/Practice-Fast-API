from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="ML Model API")


# Load your trained model
model = joblib.load("model.pkl")


# Define input data
class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float


@app.get("/")
def home():
    return {
        "message": "ML Model API is running"
    }


@app.post("/predict")
def predict(data: InputData):

    input_data = [[
        data.feature1,
        data.feature2,
        data.feature3,
        data.feature4
    ]]

    prediction = model.predict(input_data)

    return {
        "prediction": prediction[0]
    }