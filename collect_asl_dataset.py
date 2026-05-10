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
SAMPLE_START_INDEX = 151

# Load gestures from JSON
with open("labels.json", "r") as f:
    data = json.load(f)

GESTURES = data["letters"] + data["words"]

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
                hand_features = [(p.x - wrist_x, p.y - wrist_y) for p in lm]
            else:
                hand_features = [(p.x, p.y) for p in lm]
            features.extend([coord for pair in hand_features for coord in pair])
        else:
            features.extend([0.0] * 42)
    return np.array(features, dtype=np.float32)

def save_sequence(path, data):
    np.save(path, data)

def get_next_sample_index(label):
    save_dir = os.path.join(DATASET_DIR, label)
    if not os.path.isdir(save_dir):
        return SAMPLE_START_INDEX

    used_indices = set()
    for filename in os.listdir(save_dir):
        if not filename.endswith(".npy"):
            continue
        stem = os.path.splitext(filename)[0]
        if stem.isdigit():
            used_indices.add(int(stem))

    idx = SAMPLE_START_INDEX
    while idx in used_indices:
        idx += 1
    return idx

# ==============================
# GUI & Recording
# ==============================
class ASLRecorder:
    def __init__(self, root):
        self.root = root
        self.root.title("ASL Dataset Recorder")
        self.current_label = None
        self.seq_idx = 0
        self.sample_count = 0
        self.frame_count = 0
        self.sequence = np.zeros((SEQUENCE_LENGTH, 84), dtype=np.float32)
        self.recording = False
        self.paused = False

        # GUI Components
        self.label_var = tk.StringVar()
        self.label_var.set("Select a gesture")
        self.label_dropdown = ttk.Combobox(root, values=GESTURES, textvariable=self.label_var, width=30)
        self.label_dropdown.pack(pady=10)

        self.start_btn = tk.Button(root, text="Start Recording", command=self.start_recording, bg="#00c6ff")
        self.start_btn.pack(pady=5)

        self.pause_btn = tk.Button(root, text="Pause Recording", command=self.toggle_pause, bg="#ffd166")
        self.pause_btn.pack(pady=5)

        self.cancel_btn = tk.Button(root, text="Cancel Recording", command=self.cancel_recording, bg="#ff5c5c")
        self.cancel_btn.pack(pady=5)

        self.status_label = tk.Label(root, text="Status: Idle", fg="green")
        self.status_label.pack(pady=10)

        # Start video loop in separate thread
        self.cap = cv.VideoCapture(0)
        threading.Thread(target=self.video_loop, daemon=True).start()

    def start_recording(self):
        gesture = self.label_var.get()
        if gesture not in GESTURES:
            messagebox.showerror("Error", "Please select a valid gesture!")
            return
        self.current_label = gesture
        self.sample_count = get_next_sample_index(gesture)
        self.seq_idx = 0
        self.sequence.fill(0)
        self.recording = True
        self.paused = False
        self.pause_btn.config(text="Pause Recording")
        self.status_label.config(
            text=f"Recording: {gesture} (start #{self.sample_count:03d})",
            fg="blue"
        )

    def toggle_pause(self):
        if not self.recording:
            return
        self.paused = not self.paused
        if self.paused:
            self.pause_btn.config(text="Resume Recording")
            self.status_label.config(text=f"Paused: {self.current_label}", fg="orange")
        else:
            self.pause_btn.config(text="Pause Recording")
            self.status_label.config(text=f"Recording: {self.current_label}", fg="blue")

    def cancel_recording(self):
        self.recording = False
        self.paused = False
        self.seq_idx = 0
        self.sequence.fill(0)
        self.pause_btn.config(text="Pause Recording")
        self.status_label.config(text="Status: Idle", fg="green")

    def video_loop(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            self.frame_count += 1
            display_frame = cv.flip(frame, 1)

            if self.recording and (not self.paused) and self.frame_count % FRAME_SKIP == 0:
                rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = detector.detect_for_video(mp_image, int(time.time() * 1000))

                if self.current_label and result.hand_landmarks:
                    features = extract_two_hand_features(result)
                    self.sequence[self.seq_idx] = features
                    self.seq_idx += 1

                    # Save when sequence is full
                    if self.seq_idx == SEQUENCE_LENGTH:
                        save_dir = os.path.join(DATASET_DIR, self.current_label)
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, f"{self.sample_count:03d}.npy")
                        threading.Thread(target=save_sequence, args=(save_path, self.sequence.copy())).start()
                        print(f"Saved {self.current_label} sample {self.sample_count}")
                        self.sample_count += 1
                        self.seq_idx = 0

            # Show video
            cv.imshow("ASL Recorder", display_frame)
            if cv.waitKey(1) == 27:  # ESC to exit
                break

        self.cap.release()
        cv.destroyAllWindows()

# ==============================
# Run GUI
# ==============================
root = tk.Tk()
app = ASLRecorder(root)
root.mainloop()
