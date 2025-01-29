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
    allow_origins=["http://localhost:3000", "http://192.168.0.110:3000"],  # Allow frontend origin
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

@app.get("/model-metrics")
def get_model_metrics():
    try:
        metrics = joblib.load("models/model_metrics.pkl")
        return {
            "accuracy": metrics["accuracy"],
            "classification_report": metrics["classification_report"],
            "status": "Model metrics retrieved successfully"
        }
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, 
            detail="Model metrics not found. Please train the model first."
        )

# ------------------------------
# New Endpoint: Save User Interaction
# ------------------------------

# Path to store CSV
CSV_FILE = "data/sample_data.csv"

# Ensure CSV exists with proper headers
if not os.path.exists(CSV_FILE):
    df = pd.DataFrame(columns=["psychometric_inputs", "recommended_course", "feedback"])
    df.to_csv(CSV_FILE, index=False)

# Request model for saving user interactions
class Interaction(BaseModel):
    psychometric_inputs: list
    recommended_course: str
    feedback: str = None  # Optional

@app.post("/save_interaction")
async def save_interaction(interaction: Interaction):
    try:
        # Prepare new row
        new_data = {
            "psychometric_inputs": str(interaction.psychometric_inputs),  # Convert list to string for CSV
            "recommended_course": interaction.recommended_course,
            "feedback": interaction.feedback or "",  # Default to empty string if no feedback
        }
        
        # Read existing data and append new row
        df = pd.read_csv(CSV_FILE)
        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
        df.to_csv(CSV_FILE, index=False)

        return {"message": "User interaction saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save interaction: {e}")

# ------------------------------

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
