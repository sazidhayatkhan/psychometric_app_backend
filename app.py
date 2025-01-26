from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
from utils.preprocess import preprocess_data, train_model
import os

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","http://192.168.0.110:3000"],  # Allow frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Load the trained model and encoders
try:
    # Ensure the models directory exists
    os.makedirs("models", exist_ok=True)

    # Load the model and encoders
    model = joblib.load("models/course_recommendation_model.pkl")
    encoder = joblib.load("models/one_hot_encoder.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
except FileNotFoundError:
    # Train the model if it doesn't exist
    print("Model not found. Training the model...")
    X, y, encoder, label_encoder = preprocess_data("data/sample_data.csv")
    train_model(X, y)
    model = joblib.load("models/course_recommendation_model.pkl")

# Define request body model
class UserResponses(BaseModel):
    responses: list[str]

# Explicitly handle OPTIONS requests for /predict
@app.options("/predict")
async def handle_options():
    return {"message": "OK"}

# Prediction endpoint
@app.post("/predict")
def predict(responses: UserResponses):
    try:
        # Preprocess user responses
        encoded_responses = encoder.transform([responses.responses])

        # Predict the course
        prediction = model.predict(encoded_responses)
        recommended_course = label_encoder.inverse_transform(prediction)

        return {"recommended_course": recommended_course[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Admin endpoint to retrain the model
@app.post("/retrain")
def retrain():
    try:
        # Load new data (replace with your logic to fetch new data)
        X, y, _, _ = preprocess_data("data/sample_data.csv")

        # Retrain the model
        train_model(X, y)

        return {"message": "Model retrained successfully!"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)