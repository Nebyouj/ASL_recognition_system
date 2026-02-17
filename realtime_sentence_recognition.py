import time
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

torch.set_grad_enabled(False)

# ==============================
# CONFIG
# ==============================
SEQUENCE_LENGTH = 30
PAUSE_FRAMES = 8
MOTION_THRESHOLD = 0.0015
FRAME_SKIP = 2

# ==============================
# Load Model
# ==============================
checkpoint = torch.load("asl_bilstm.pth", map_location="cpu")
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

# ==============================
# Feature Extraction
# ==============================
def extract_features(result):
    features = []
    wrists = []

    for i in range(2):
        if i < len(result.hand_landmarks):
            lm = result.hand_landmarks[i]
            wrists.extend([lm[0].x, lm[0].y])  # wrist only
            for p in lm:
                features.extend([p.x, p.y])
        else:
            wrists.extend([0.0, 0.0])
            features.extend([0.0] * 42)

    return features, wrists

# ==============================
# Motion Detection (FAST)
# ==============================
def motion_detected(prev, curr):
    if prev is None:
        return True
    diff = np.mean(np.abs(np.array(curr) - np.array(prev)))
    return diff > MOTION_THRESHOLD

# ==============================
# Main Loop
# ==============================
cap = cv.VideoCapture(0)

sequence = deque(maxlen=SEQUENCE_LENGTH)
sentence = []
prev_wrists = None
still_frames = 0
frame_id = 0

print("🟢 FAST ASL Sentence Recognition")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.flip(frame, 1)
    frame_id += 1

    if frame_id % FRAME_SKIP == 0:
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect_for_video(mp_image, int(time.time() * 1000))

        if result.hand_landmarks:
            features, wrists = extract_features(result)

            if motion_detected(prev_wrists, wrists):
                sequence.append(features)
                still_frames = 0
            else:
                still_frames += 1

            prev_wrists = wrists

            # ---- Word Boundary ----
            if still_frames >= PAUSE_FRAMES and len(sequence) == SEQUENCE_LENGTH:
                x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
                pred = torch.argmax(model(x), dim=1).item()
                sentence.append(labels[pred])

                print("Recognized:", labels[pred])

                sequence.clear()
                still_frames = 0

    # ==============================
    # Display
    # ==============================
    cv.putText(
        frame,
        " ".join(sentence[-6:]),
        (20, 60),
        cv.FONT_HERSHEY_SIMPLEX,
        1.4,
        (0, 255, 0),
        3
    )

    cv.imshow("ASL Sentence Recognition (FAST)", frame)

    if cv.waitKey(1) == 27:
        break

cap.release()
cv.destroyAllWindows()
