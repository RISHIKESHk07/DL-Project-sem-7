import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# Load face detector (same as your original)
face_classifier = cv2.CascadeClassifier('/home/intelliagent-19/Desktop/DL-project-sem7/backend/Model/haarcascade_frontalface_default.xml')

# Load your trained EfficientNet model
classifier = None
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Initialize the model
def load_efficientnet_model(model_path):
    global classifier
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model (same architecture as training)
    model = models.efficientnet_b2(pretrained=False)
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
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("✅ EfficientNet model loaded successfully!")
    return model

# Load the model
classifier = load_efficientnet_model('/home/intelliagent-19/Desktop/DL-project-sem7/backend/Model/efficientnet_finetuned_unfreeze_b2_3_class.pth')

# Define transforms for EfficientNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        
        # Extract face ROI from COLOR frame (not grayscale) for EfficientNet
        roi_color = frame[y:y+h, x:x+w]
        
        if np.sum([roi_color]) != 0:
            try:
                # Convert BGR to RGB and resize for EfficientNet
                roi_rgb = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
                roi_resized = cv2.resize(roi_rgb, (224, 224), interpolation=cv2.INTER_AREA)
                
                # Convert to PIL Image and apply transforms
                roi_pil = Image.fromarray(roi_resized)
                roi_tensor = transform(roi_pil).unsqueeze(0)  # Add batch dimension
                
                # Move to appropriate device
                device = next(classifier.parameters()).device
                roi_tensor = roi_tensor.to(device)
                
                # Predict
                with torch.no_grad():
                    prediction = classifier(roi_tensor)[0]
                    probabilities = torch.nn.functional.softmax(prediction, dim=0)
                    confidence, predicted = torch.max(probabilities, 0)
                
                label = emotion_labels[predicted.item()]
                confidence_score = confidence.item()
                
                # Display label with confidence
                label_position = (x, y)
                display_text = f"{label} ({confidence_score:.2f})"
                cv2.putText(frame, display_text, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
            except Exception as e:
                print(f"Error in prediction: {e}")
                cv2.putText(frame, 'Error', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, 'No Faces', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('EfficientNet Emotion Detector', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
