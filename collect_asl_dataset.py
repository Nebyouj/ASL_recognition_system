# GUI for data collection

import os
import time
import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import threading
import json
import tkinter as tk
from tkinter import ttk, messagebox

# ==============================
# CONFIG
# ==============================
SEQUENCE_LENGTH = 30
DATASET_DIR = "asl_dataset"
FRAME_SKIP = 2
NORMALIZE_LANDMARKS = True

# Load gestures from JSON
with open("labels.json", "r") as f:
    data = json.load(f)

GESTURES = data["words"]

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

            if NORMALIZE_LANDMARKS:
                wrist_x, wrist_y = lm[0].x, lm[0].y
                for p in lm:
                    features.append(p.x - wrist_x)
                    features.append(p.y - wrist_y)
            else:
                for p in lm:
                    features.append(p.x)
                    features.append(p.y)
        else:
            features.extend([0.0] * 42)

    return np.array(features, dtype=np.float32)


def save_sequence(path, data):
    np.save(path, data)

# ==============================
# GUI & Recording
# ==============================
class ASLRecorder:
    def __init__(self, root):
        self.root = root
        self.root.title("ASL Dataset Recorder")

        self.current_label = None
        self.seq_idx = 0
        self.sample_count = 151
        self.frame_count = 0
        self.sequence = np.zeros((SEQUENCE_LENGTH, 84), dtype=np.float32)
        self.recording = False

        # GUI Components
        self.label_var = tk.StringVar()
        self.label_var.set("Select a gesture")

        self.label_dropdown = ttk.Combobox(
            root, values=GESTURES,
            textvariable=self.label_var,
            width=30
        )
        self.label_dropdown.pack(pady=10)

        # START BUTTON
        self.start_btn = tk.Button(
            root, text="Start Recording",
            command=self.start_recording,
            bg="#00c6ff"
        )
        self.start_btn.pack(pady=5)

        # ✅ PAUSE BUTTON (ADDED)
        self.pause_btn = tk.Button(
            root, text="Pause",
            command=self.pause_recording,
            bg="#ffaa00"
        )
        self.pause_btn.pack(pady=5)

        # CLEAR BUTTON
        self.clear_btn = tk.Button(
            root, text="Clear Samples",
            command=self.clear_samples,
            bg="#ff5c5c"
        )
        self.clear_btn.pack(pady=5)

        self.status_label = tk.Label(
            root, text="Status: Idle", fg="green"
        )
        self.status_label.pack(pady=10)

        # Camera thread
        self.cap = cv.VideoCapture(0)
        threading.Thread(target=self.video_loop, daemon=True).start()

    # ==============================
    # CONTROL FUNCTIONS
    # ==============================

    def start_recording(self):
        gesture = self.label_var.get()
        if gesture not in GESTURES:
            messagebox.showerror("Error", "Please select a valid gesture!")
            return

        # ✅ ONLY reset if new gesture selected
        if self.current_label != gesture:
            self.sample_count = 151

        self.current_label = gesture
        self.seq_idx = 0
        self.recording = True

        self.status_label.config(
            text=f"Recording: {gesture}", fg="blue"
        )

    # ✅ PAUSE FUNCTION (ADDED)
    def pause_recording(self):
        self.recording = False
        self.status_label.config(text="Paused", fg="orange")

    def clear_samples(self):
        self.sample_count = 50
        self.seq_idx = 0
        self.sequence.fill(0)
        self.status_label.config(
            text="Samples cleared", fg="orange"
        )

    # ==============================
    # VIDEO LOOP
    # ==============================

    def video_loop(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            self.frame_count += 1
            display_frame = cv.flip(frame, 1)

            if self.recording and self.frame_count % FRAME_SKIP == 0:
                rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                result = detector.detect_for_video(
                    mp_image,
                    int(time.time() * 1000)
                )

                if self.current_label and result.hand_landmarks:
                    features = extract_two_hand_features(result)
                    self.sequence[self.seq_idx] = features
                    self.seq_idx += 1

                    # Save when full
                    if self.seq_idx == SEQUENCE_LENGTH:
                        save_dir = os.path.join(
                            DATASET_DIR, self.current_label
                        )
                        os.makedirs(save_dir, exist_ok=True)

                        save_path = os.path.join(
                            save_dir,
                            f"{self.sample_count:03d}.npy"
                        )

                        threading.Thread(
                            target=save_sequence,
                            args=(save_path, self.sequence.copy())
                        ).start()

                        print(
                            f"Saved {self.current_label} sample {self.sample_count}"
                        )

                        self.sample_count += 1
                        self.seq_idx = 0

                        # ✅ AUTO PAUSE AFTER SAVE (ADDED)
                        self.recording = False
                        self.status_label.config(
                            text=f"Saved {self.current_label} - Paused",
                            fg="green"
                        )

            # Show video
            cv.imshow("ASL Recorder", display_frame)

            if cv.waitKey(1) == 27:  # ESC
                break

        self.cap.release()
        cv.destroyAllWindows()

# ==============================
# RUN
# ==============================
root = tk.Tk()
app = ASLRecorder(root)
root.mainloop()