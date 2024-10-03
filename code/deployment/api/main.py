from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import torch
from model_loader import load_model
from config import class_names
from image_utils import prepare_image

app = FastAPI()

# Load model
model = load_model('/models/animal_classifier.pth')

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    prepared_image = prepare_image(image)

    with torch.no_grad():
        predictions = model(prepared_image)
        predicted_class = class_names[torch.argmax(predictions, dim=1).item()]

    return {"prediction": predicted_class}