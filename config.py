
CAMERA_INDEX = 0  
CAMERA_WIDTH = 640  
CAMERA_HEIGHT = 480  


FACE_SCALE_FACTOR = 1.1
FACE_MIN_NEIGHBORS = 5
FACE_MIN_SIZE = (60, 60)

MODEL_PATH = 'trained_model.pth'
HF_MODEL_NAME = "dima806/facial_emotions_image_detection"
INPUT_SIZE = (224, 224)

SHOW_CONFIDENCE = True
FONT_SCALE = 0.7
FONT_THICKNESS = 2
BOX_COLOR = (0, 255, 0)
HF_TEXT_COLOR = (36, 255, 12)
CUSTOM_TEXT_COLOR = (255, 0, 0)
ENSEMBLE_TEXT_COLOR = (0, 0, 255)

USE_GPU = True
PROCESS_EVERY_N_FRAMES = 1

SAVE_RESULTS = False
OUTPUT_DIR = "output"
VIDEO_FPS = 30

EMOTION_LABELS = ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Angry", "Normal"]

EMOTION_MAPPING = {
    "joy": "Happy",
    "neutral": "Normal",
    "sad": "Sad", 
    "anger": "Angry",
    "fear": "Fear",
    "surprise": "Surprise",
    "disgust": "Disgust"
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
