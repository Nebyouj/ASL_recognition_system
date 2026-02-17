import os
import time
import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque

# ==============================
# CONFIG
# ==============================
SEQUENCE_LENGTH = 30
INPUT_SIZE = 84
DATASET_DIR = "asl_dataset"

ASL_LABELS = {
    ord('h'): "HELLO",
    ord('t'): "THANK_YOU",
    ord('y'): "YES",
    ord('n'): "NO",
    ord('s'): "SORRY"  # Space for blank
}

# ==============================
# MediaPipe Initialization
# ==============================
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

# ==============================
# Helper Functions
# ==============================
def extract_two_hand_features(result):
    features = []
    for hand_idx in range(2):
        if hand_idx < len(result.hand_landmarks):
            lm = result.hand_landmarks[hand_idx]
            for p in lm:
                features.extend([p.x, p.y])
        else:
            features.extend([0.0] * 42)
    if len(features) != INPUT_SIZE:
        raise ValueError(f"Expected {INPUT_SIZE} features, got {len(features)}")
    return features

# ==============================
# Main
# ==============================
cap = cv.VideoCapture(0)
sequence = []
current_label = None
sample_count = 0

print("Press h,t,y,n to record | ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.flip(frame, 1)
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect_for_video(mp_image, int(time.time() * 1000))

    key = cv.waitKey(1)
    if key == 27:
        break

    if key in ASL_LABELS:
        current_label = ASL_LABELS[key]
        sequence.clear()
        print(f"Recording: {current_label}")

    if current_label and result.hand_landmarks:
        features = extract_two_hand_features(result)
        sequence.append(features)

        cv.putText(frame, f"Recording {current_label}: {len(sequence)}/{SEQUENCE_LENGTH}",
                   (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        if len(sequence) == SEQUENCE_LENGTH:
            save_dir = os.path.join(DATASET_DIR, current_label)
            os.makedirs(save_dir, exist_ok=True)
            np.save(f"{save_dir}/{sample_count:03d}.npy", np.array(sequence))
            print(f"Saved {current_label} sample {sample_count}")
            sample_count += 1
            sequence.clear()

    cv.imshow("ASL Dataset Collection", frame)

cap.release()
cv.destroyAllWindows()
