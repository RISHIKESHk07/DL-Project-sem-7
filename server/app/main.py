from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
import uvicorn
import io, base64
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import os

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(base_dir, "Model", "efficientnet_finetuned_unfreeze_b2_3_class.pth")
haar_path = os.path.join(base_dir, "Model", "haarcascade_frontalface_default.xml")

print(model_path,haar_path);

app = FastAPI()

# Allow your frontend origin(s)
origins = [
    "http://127.0.0.1:8001",  # if you serve HTML from here
    "http://localhost:8001",
    "http://127.0.0.1:5500",  # if using VS Code Live Server
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load face detector and model
face_classifier = cv2.CascadeClassifier(haar_path)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using:", device)

def load_efficientnet_model(model_path):
    checkpoint = torch.load(model_path, map_location=device)
    weights = EfficientNet_B2_Weights.DEFAULT
    model = models.efficientnet_b2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 7)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

classifier = load_efficientnet_model(model_path)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

@app.post("/predict_frame")
async def predict_frame(data: dict):
    try:
        img_data = base64.b64decode(data["frame"])
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        frame = np.array(img)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = face_classifier.detectMultiScale(gray)

        predictions = []
        log_records = []
        for (x, y, w, h) in faces:
            roi_color = frame[y:y+h, x:x+w]
            if roi_color.size == 0:
                continue
            roi_pil = Image.fromarray(cv2.resize(roi_color, (224, 224)))
            roi_tensor = transform(roi_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                prediction = classifier(roi_tensor)[0]
                probabilities = torch.nn.functional.softmax(prediction, dim=0)
                confidence, predicted = torch.max(probabilities, 0)
            
            predictions.append({
                "emotion": emotion_labels[predicted.item()],
                "confidence": float(confidence),
                "box": [int(x), int(y), int(w), int(h)]
            })
            log_records.append({
                "emotion": emotion,
                "confidence": conf_value,
                "frame_width": frame.shape[1],
                "frame_height": frame.shape[0],
                "num_faces": len(faces)
            })
        # --- WhyLogs Local Profiling ---
        if log_records:
            df = pd.DataFrame(log_records)
            profile = why.log(df, dataset_timestamp=datetime.now())

            # Save locally for whylabs docker to consume
            output_dir = "./whylogs_data"
            os.makedirs(output_dir, exist_ok=True)
            profile.writer("local").option("base_dir", output_dir).write()

        return {"faces": predictions}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

