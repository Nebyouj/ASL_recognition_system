import time
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

SEQUENCE_LENGTH = 30

# ==============================
# Load Model
# ==============================
checkpoint = torch.load("asl_bilstm.pth")
labels = checkpoint["labels"]

class ASLBiLSTM(nn.Module):
    def __init__(self, input_size=84, hidden_size=128, num_classes=len(labels)):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = ASLBiLSTM()
model.load_state_dict(checkpoint["model"])
model.eval()

# ==============================
# MediaPipe
# ==============================
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2
)
detector = vision.HandLandmarker.create_from_options(options)

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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.flip(frame, 1)
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect_for_video(mp_image, int(time.time() * 1000))

    if result.hand_landmarks:
        sequence.append(extract_two_hand_features(result))

        if len(sequence) == SEQUENCE_LENGTH:
            x = torch.tensor(np.array(sequence), dtype=torch.float32).unsqueeze(0)
            pred = torch.argmax(model(x), dim=1).item()
            word = labels[pred]

            cv.putText(frame, word, (20, 60),
                       cv.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)

    cv.imshow("ASL Recognition", frame)
    if cv.waitKey(1) == 27:
        break

cap.release()
cv.destroyAllWindows()
