import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from model import load_model
from recognizer import ASLRecognizer
from gtts import gTTS
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model, labels = load_model("asl_bilstm.pth")
recognizer = ASLRecognizer(model, labels)

# Serve static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def serve_home():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()
    
@app.post("/predict")
async def predict(frame: UploadFile = File(...)):
    image_bytes = await frame.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    word = recognizer.process_frame(frame)

    return {
        "word": word, 
        "sentence": " ".join(recognizer.sentence)
    }

@app.get("/tts")
def tts(sentence: str):
    tts = gTTS(sentence)
    path = "speech.mp3"
    tts.save(path)
    return FileResponse(path, media_type="audio/mpeg")
