# Emotion Detector 🎭

Real-time facial emotion detection from webcam feed.

## Description 📝

Application detecting emotions from webcam using two models:
- 🤖 Custom ResNet18 model
- 🤗 Hugging Face Transformers model

## Requirements 📋

- Python 3.8+
- 📷 Webcam

## Installation 🚀

```bash
git clone https://github.com/yourusername/emotion-detector.git
cd emotion-detector
pip install -r requirements.txt
```

## Usage 💻

```bash
python face.py
```

Press 'q' to quit the application.

## Detected Emotions 😊

- 😲 Surprise
- 😨 Fear
- 🤢 Disgust
- 😊 Happy
- 😢 Sad
- 😡 Angry
- 😐 Normal

## Files 📁

- `face.py` - main application
- `train_resnet.py` - model training
- `requirements.txt` - dependencies
- `trained_model.pth` - trained model
- `config.py` - configuration settings
