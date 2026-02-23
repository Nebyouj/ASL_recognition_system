# import os
# import time
# import cv2 as cv
# import numpy as np
# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# from collections import deque

# # ==============================
# # CONFIG
# # ==============================
# SEQUENCE_LENGTH = 30
# DATASET_DIR = "asl_dataset"

# ASL_LABELS = {
#     ord('h'): "HELLO",
#     ord('t'): "THANK_YOU",
#     ord('y'): "YES",
#     ord('n'): "NO",
#     ord('s'): "SORRY"  # Space for blank
# }

# # ==============================
# # MediaPipe Initialization
# # ==============================
# base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
# options = vision.HandLandmarkerOptions(
#     base_options=base_options,
#     running_mode=vision.RunningMode.VIDEO,
#     num_hands=2,
#     min_hand_detection_confidence=0.7,
#     min_hand_presence_confidence=0.5,
#     min_tracking_confidence=0.5
# )
# detector = vision.HandLandmarker.create_from_options(options)

# # ==============================
# # Helper Functions
# # ==============================
# def extract_two_hand_features(result):
#     features = []
#     for hand_idx in range(2):
#         if hand_idx < len(result.hand_landmarks):
#             lm = result.hand_landmarks[hand_idx]
#             for p in lm:
#                 features.extend([p.x, p.y])
#         else:
#             features.extend([0.0] * 42)
#     return features

# # ==============================
# # Main
# # ==============================
# cap = cv.VideoCapture(0)
# sequence = []
# current_label = None
# sample_count = 0

# print("Press h,t,y,n to record | ESC to exit")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = cv.flip(frame, 1)
#     rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
#     mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

#     result = detector.detect_for_video(mp_image, int(time.time() * 1000))

#     key = cv.waitKey(1)
#     if key == 27:
#         break

#     if key in ASL_LABELS:
#         current_label = ASL_LABELS[key]
#         sequence.clear()
#         print(f"Recording: {current_label}")

#     if current_label and result.hand_landmarks:
#         features = extract_two_hand_features(result)
#         sequence.append(features)

#         cv.putText(frame, f"Recording {current_label}: {len(sequence)}/{SEQUENCE_LENGTH}",
#                    (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

#         if len(sequence) == SEQUENCE_LENGTH:
#             save_dir = os.path.join(DATASET_DIR, current_label)
#             os.makedirs(save_dir, exist_ok=True)
#             np.save(f"{save_dir}/{sample_count:03d}.npy", np.array(sequence))
#             print(f"Saved {current_label} sample {sample_count}")
#             sample_count += 1
#             sequence.clear()

#     cv.imshow("ASL Dataset Collection", frame)

# cap.release()
# cv.destroyAllWindows()



# import os
# import time
# import cv2 as cv
# import numpy as np
# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# import threading

# # ==============================
# # CONFIG
# # ==============================
# SEQUENCE_LENGTH = 30
# DATASET_DIR = "asl_dataset"
# FRAME_SKIP = 2  # process every 2nd frame
# NORMALIZE_LANDMARKS = True  # normalize landmarks relative to wrist

# ASL_LABELS = {
#     ord('h'): "HELLO",
#     ord('t'): "THANK_YOU",
#     ord('y'): "YES",
#     ord('n'): "NO",
#     ord('s'): "SORRY"
# }

# # ==============================
# # MediaPipe Initialization
# # ==============================
# base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
# options = vision.HandLandmarkerOptions(
#     base_options=base_options,
#     running_mode=vision.RunningMode.VIDEO,
#     num_hands=2,
#     min_hand_detection_confidence=0.7,
#     min_hand_presence_confidence=0.5,
#     min_tracking_confidence=0.5
# )
# detector = vision.HandLandmarker.create_from_options(options)

# # ==============================
# # Helper Functions
# # ==============================
# def extract_two_hand_features(result):
#     features = []
#     for hand_idx in range(2):
#         if hand_idx < len(result.hand_landmarks):
#             lm = result.hand_landmarks[hand_idx]
#             if NORMALIZE_LANDMARKS:
#                 wrist_x, wrist_y = lm[0].x, lm[0].y
#                 features.extend([(p.x - wrist_x, p.y - wrist_y) for p in lm])
#             else:
#                 features.extend([(p.x, p.y) for p in lm])
#             features = [coord for pair in features for coord in pair]  # flatten
#         else:
#             features.extend([0.0] * 42)
#     return np.array(features, dtype=np.float32)

# def save_sequence(path, data):
#     np.save(path, data)

# # ==============================
# # Main
# # ==============================
# cap = cv.VideoCapture(0)
# sequence = np.zeros((SEQUENCE_LENGTH, 84), dtype=np.float32)  # preallocate
# seq_idx = 0
# current_label = None
# sample_count = 0
# frame_count = 0

# print("Press h,t,y,n,s to record | ESC to exit")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame_count += 1
#     display_frame = cv.flip(frame, 1)  # flip only for display

#     key = cv.waitKey(1)
#     if key == 27:  # ESC
#         break

#     if key in ASL_LABELS:
#         current_label = ASL_LABELS[key]
#         seq_idx = 0
#         print(f"Recording: {current_label}")

#     # Process only every FRAME_SKIP frames
#     if frame_count % FRAME_SKIP == 0:
#         rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
#         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
#         result = detector.detect_for_video(mp_image, int(time.time() * 1000))

#         if current_label and result.hand_landmarks:
#             features = extract_two_hand_features(result)
#             sequence[seq_idx] = features
#             seq_idx += 1

#             cv.putText(display_frame, f"Recording {current_label}: {seq_idx}/{SEQUENCE_LENGTH}",
#                        (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

#             # Save when sequence is full
#             if seq_idx == SEQUENCE_LENGTH:
#                 save_dir = os.path.join(DATASET_DIR, current_label)
#                 os.makedirs(save_dir, exist_ok=True)
#                 save_path = os.path.join(save_dir, f"{sample_count:03d}.npy")
#                 threading.Thread(target=save_sequence, args=(save_path, sequence.copy())).start()
#                 print(f"Saved {current_label} sample {sample_count}")
#                 sample_count += 1
#                 seq_idx = 0  # reset sequence index

#     cv.imshow("ASL Dataset Collection", display_frame)

# cap.release()
# cv.destroyAllWindows()










# for more conversation 

# labels.json
# {
#     "letters": ["A","B","C","D","E","F","G","H","I","J","K","L","M",
#                 "N","O","P","Q","R","S","T","U","V","W","X","Y","Z","SPACE"],
#     "words": ["HELLO","THANK_YOU","YES","NO","SORRY","PLEASE","GOOD_MORNING","HELP"]
# }




# import os
# import time
# import cv2 as cv
# import numpy as np
# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# import threading
# import json

# # ==============================
# # CONFIG
# # ==============================
# SEQUENCE_LENGTH = 30
# DATASET_DIR = "asl_dataset"
# FRAME_SKIP = 2  # process every 2nd frame
# NORMALIZE_LANDMARKS = True  # normalize landmarks relative to wrist

# # Load gestures from JSON
# with open("labels.json", "r") as f:
#     data = json.load(f)

# GESTURES = data["letters"] + data["words"]  # all gestures to record
# NUM_GESTURES = len(GESTURES)

# print("Gestures available for recording:")
# for idx, g in enumerate(GESTURES):
#     print(f"{idx+1}: {g}")

# # ==============================
# # MediaPipe Initialization
# # ==============================
# base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
# options = vision.HandLandmarkerOptions(
#     base_options=base_options,
#     running_mode=vision.RunningMode.VIDEO,
#     num_hands=2,
#     min_hand_detection_confidence=0.7,
#     min_hand_presence_confidence=0.5,
#     min_tracking_confidence=0.5
# )
# detector = vision.HandLandmarker.create_from_options(options)

# # ==============================
# # Helper Functions
# # ==============================
# def extract_two_hand_features(result):
#     features = []
#     for hand_idx in range(2):
#         if hand_idx < len(result.hand_landmarks):
#             lm = result.hand_landmarks[hand_idx]
#             if NORMALIZE_LANDMARKS:
#                 wrist_x, wrist_y = lm[0].x, lm[0].y
#                 features.extend([(p.x - wrist_x, p.y - wrist_y) for p in lm])
#             else:
#                 features.extend([(p.x, p.y) for p in lm])
#             features = [coord for pair in features for coord in pair]  # flatten
#         else:
#             features.extend([0.0] * 42)
#     return np.array(features, dtype=np.float32)

# def save_sequence(path, data):
#     np.save(path, data)

# # ==============================
# # Main
# # ==============================
# cap = cv.VideoCapture(0)
# sequence = np.zeros((SEQUENCE_LENGTH, 84), dtype=np.float32)  # preallocate
# seq_idx = 0
# current_label = None
# sample_count = 0
# frame_count = 0

# print("\nPress number keys (1-{}) to start recording a gesture.".format(NUM_GESTURES))
# print("Press ESC to exit.\n")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame_count += 1
#     display_frame = cv.flip(frame, 1)  # flip only for display

#     key = cv.waitKey(1)
#     if key == 27:  # ESC
#         break

#     # Check if key is number key 1-9, 10-99 for large gesture lists
#     if 49 <= key <= 57:  # ASCII '1'-'9'
#         idx = key - 49  # 0-indexed
#         if idx < NUM_GESTURES:
#             current_label = GESTURES[idx]
#             seq_idx = 0
#             print(f"Recording: {current_label}")

#     # Optional: handle number keys >9 (for gesture lists >9)
#     # Can expand with more sophisticated key input (like arrow keys or GUI)

#     # Process only every FRAME_SKIP frames
#     if frame_count % FRAME_SKIP == 0:
#         rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
#         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
#         result = detector.detect_for_video(mp_image, int(time.time() * 1000))

#         if current_label and result.hand_landmarks:
#             features = extract_two_hand_features(result)
#             sequence[seq_idx] = features
#             seq_idx += 1

#             cv.putText(display_frame, f"Recording {current_label}: {seq_idx}/{SEQUENCE_LENGTH}",
#                        (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

#             # Save when sequence is full
#             if seq_idx == SEQUENCE_LENGTH:
#                 save_dir = os.path.join(DATASET_DIR, current_label)
#                 os.makedirs(save_dir, exist_ok=True)
#                 save_path = os.path.join(save_dir, f"{sample_count:03d}.npy")
#                 threading.Thread(target=save_sequence, args=(save_path, sequence.copy())).start()
#                 print(f"Saved {current_label} sample {sample_count}")
#                 sample_count += 1
#                 seq_idx = 0  # reset sequence index

#     cv.imshow("ASL Dataset Collection", display_frame)

# cap.release()
# cv.destroyAllWindows()




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
                for p in lm:
                    features.append(p.x - wrist_x)
                    features.append(p.y - wrist_y)
            else:
                for p in lm:
                    features.append(p.x)
                    features.append(p.y)
        else:
            # 21 landmarks × (x,y) = 42 values
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
        self.sample_count = 50
        self.frame_count = 0
        self.sequence = np.zeros((SEQUENCE_LENGTH, 84), dtype=np.float32)
        self.recording = False

        # GUI Components
        self.label_var = tk.StringVar()
        self.label_var.set("Select a gesture")
        self.label_dropdown = ttk.Combobox(root, values=GESTURES, textvariable=self.label_var, width=30)
        self.label_dropdown.pack(pady=10)

        self.start_btn = tk.Button(root, text="Start Recording", command=self.start_recording, bg="#00c6ff")
        self.start_btn.pack(pady=5)

        self.clear_btn = tk.Button(root, text="Clear Samples", command=self.clear_samples, bg="#ff5c5c")
        self.clear_btn.pack(pady=5)

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
        self.seq_idx = 0
        self.recording = True
        self.status_label.config(text=f"Recording: {gesture}", fg="blue")

    def clear_samples(self):
        self.sample_count = 50
        self.seq_idx = 0
        self.sequence.fill(0)
        self.status_label.config(text="Samples cleared", fg="orange")

    def video_loop(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            self.frame_count += 1
            display_frame = cv.flip(frame, 1)

            if self.recording and self.frame_count % FRAME_SKIP == 0:
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

