from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "brain_tumor_model.h5"
CLASS_INDICES_PATH = "class_indices.txt"
IMG_WIDTH, IMG_HEIGHT = 224, 224

model = None
class_names = []

def load_model_and_classes():
    global model, class_names
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded.")
    else:
        print("Model file not found. Please train the model first.")

    if os.path.exists(CLASS_INDICES_PATH):
        with open(CLASS_INDICES_PATH, 'r') as f:
            indices = eval(f.read())
            # Invert the dictionary to get {index: name}
            class_names = {v: k for k, v in indices.items()}
    else:
        # Fallback if file doesn't exist (e.g. before training)
        class_names = {0: 'glioma', 1: 'meningioma', 2: 'notumor', 3: 'pituitary'}

@app.on_event("startup")
async def startup_event():
    load_model_and_classes()

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)).convert("RGB").resize((IMG_WIDTH, IMG_HEIGHT)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded. Please train the model first."}
    
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    # Preprocess (EfficientNet expects [0, 255], so no rescaling needed)
    # img_batch = img_batch / 255.0

    predictions = model.predict(img_batch)
    predicted_class_index = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))
    
    predicted_class = class_names.get(predicted_class_index, "Unknown")

    return {
        "class": predicted_class,
        "confidence": confidence
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
