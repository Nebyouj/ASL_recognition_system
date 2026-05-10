import time
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# import pyttsx3  # optional: for text-to-speech

SEQUENCE_LENGTH = 30
SMOOTH_WINDOW = 10      # smoothing predictions over last 10
CONF_THRESHOLD = 0.7    # minimum softmax probability to show word

# ==============================
# Load Model
# ==============================
checkpoint = torch.load("asl_bilstm.pth", map_location="cpu")
labels = checkpoint["labels"]

class ASLBiLSTM(nn.Module):
    def __init__(self, input_size=84, hidden_size=128, num_classes=len(labels), dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size,
            batch_first=True,
            bidirectional=True
        )
        self.layernorm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.layernorm(out)
        out = self.dropout(out)
        return self.fc(out)


# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ASLBiLSTM(num_classes=len(labels)).to(device)
model.load_state_dict(checkpoint["model"])
model.eval()

# ==============================
# MediaPipe Hand Detector
# ==============================
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2
)
detector = vision.HandLandmarker.create_from_options(options)

# ==============================
# TTS Engine (optional)
# ==============================
# tts = pyttsx3.init()
# tts.setProperty("rate", 150)  # speaking speed

# ==============================
# Feature extraction
# ==============================
def extract_two_hand_features(result):
    features = []
    for i in range(2):
        if i < len(result.hand_landmarks):
            for p in result.hand_landmarks[i]:
                features.extend([p.x, p.y])
        else:
            features.extend([0.0] * 42)
    return features

# ==============================
# Real-time loop
# ==============================
cap = cv.VideoCapture(0)
sequence = deque(maxlen=SEQUENCE_LENGTH)
predictions = deque(maxlen=SMOOTH_WINDOW)
prev_word = ""

fps_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.flip(frame, 1)
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect_for_video(mp_image, int(time.time() * 1000))

    word = ""
    if result.hand_landmarks:
        sequence.append(extract_two_hand_features(result))

        if len(sequence) == SEQUENCE_LENGTH:
            x = torch.tensor(np.array(sequence), dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                conf, pred_idx = torch.max(probs, dim=1)
                conf, pred_idx = conf.item(), pred_idx.item()

                if conf >= CONF_THRESHOLD:
                    predictions.append(labels[pred_idx])
                    # smoothing: most common prediction in the last SMOOTH_WINDOW
                    word = max(set(predictions), key=predictions.count)

                    # speak only when new word is recognized
                    # if word != prev_word:
                    #     tts.say(word)
                    #     tts.runAndWait()
                    #     prev_word = word

    # Overlay recognized word
    if word:
        cv.putText(frame, word, (20, 60), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    # Display FPS
    fps = 1.0 / (time.time() - fps_time)
    fps_time = time.time()
    cv.putText(frame, f"FPS: {int(fps)}", (20, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv.imshow("ASL Recognition", frame)
    if cv.waitKey(1) == 27:  # ESC to exit
        break

cap.release()
cv.destroyAllWindows()