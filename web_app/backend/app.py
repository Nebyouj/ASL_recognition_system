import cv2
import numpy as np
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
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

@app.post("/predict")
async def predict(frame: UploadFile):
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
