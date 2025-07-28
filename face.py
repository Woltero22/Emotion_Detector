import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from transformers import pipeline
import torch.nn.functional as F

emotion_labels = ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Angry", "Normal"]

model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(emotion_labels))
model.load_state_dict(torch.load('trained_model.pth', map_location=torch.device('cpu')))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("Model ResNet18 wczytany i gotowy do użycia.")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

emotion_detector = pipeline(
    "image-classification",
    model="dima806/facial_emotions_image_detection"
)

print("Model Transformers wczytany i gotowy do użycia.")

emotion_mapping = {
    "joy": "Happy",
    "neutral": "Normal",
    "sad": "Sad",
    "anger": "Angry",
    "fear": "Fear",
    "surprise": "Surprise",
    "disgust": "Disgust"
}

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Nie można otworzyć kamery")
    exit()

print("Rozpoczęcie detekcji emocji z kamery...")
print("Naciśnij 'q', aby zakończyć.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]

        pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        preds_transformers = emotion_detector(pil_img)

        transformers_results = {}

        for emotion in emotion_labels:
            transformers_results[emotion] = 0.0
            
        for pred in preds_transformers:
            emotion_name = pred['label']
            if emotion_name in emotion_mapping:
                mapped_emotion = emotion_mapping[emotion_name]
                transformers_results[mapped_emotion] = pred['score']
        
        label_transformers = preds_transformers[0]['label']
        score_transformers = preds_transformers[0]['score']
        
        if label_transformers in emotion_mapping:
            label_transformers_mapped = emotion_mapping[label_transformers]
        else:
            label_transformers_mapped = label_transformers

        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_tensor = transform(face_rgb).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(face_tensor)
            probabilities = F.softmax(outputs, dim=1).squeeze().cpu().numpy()
            _, preds = torch.max(outputs, 1)
        
        label_custom = emotion_labels[preds.item()]
        score_custom = probabilities[preds.item()]
        
        custom_results = {}
        for i, emotion in enumerate(emotion_labels):
            custom_results[emotion] = float(probabilities[i])
        
        avg_results = {}
        
        for emotion in emotion_labels:
            avg_results[emotion] = (transformers_results[emotion] + custom_results[emotion]) / 2.0
        
        final_emotion = max(avg_results, key=avg_results.get)
        final_score = avg_results[final_emotion]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        text_transformers = f"Transformers: {label_transformers_mapped} ({score_transformers:.2f})"
        cv2.putText(
            frame,
            text_transformers,
            (x, y - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (36, 255, 12),
            2
        )
        
        text_custom = f"Wlasny model: {label_custom} ({score_custom:.2f})"
        cv2.putText(
            frame,
            text_custom,
            (x, y - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

        text_avg = f"Srednia: {final_emotion} ({final_score:.2f})"
        cv2.putText(
            frame,
            text_avg,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )
    try:
        print(f"Transformers: {label_transformers_mapped} {score_transformers:.2f}\tWlasny model: {label_custom} {score_custom:.2f}\tSrednia: {final_emotion} {final_score:.2f}")
    except NameError:
        pass
    cv2.imshow('Porownanie detekcji emocji', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()