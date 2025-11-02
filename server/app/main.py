from fastapi import FastAPI
import pickle
import os
from fastapi import BackgroundTasks

app = FastAPI()

# Load model at startup
MODEL_PATH = "model.pkl"

def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def retrain_model():
    # Load old data + new data
    # Retrain model
    # Save updated model back to model.pkl
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(new_model, f)

# model = load_model()

@app.post("/retrain")
def trigger_retrain(background_tasks: BackgroundTasks):
    background_tasks.add_task(retrain_model)
    return {"status": "Retraining started"}

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
def predict(data: dict):
    # Mock prediction
    return {"prediction": 42}
