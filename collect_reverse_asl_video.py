# import cv2
# import mediapipe as mp
# import numpy as np
# import time
# import os

# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision

# # ==============================
# # CONFIG
# # ==============================
# SAVE_DIR = "reverse_dataset"
# SEQUENCE = []
# RECORDING = False

# # ==============================
# # INIT MEDIAPIPE TASKS
# # ==============================

# # Hand Landmarker
# hand_options = vision.HandLandmarkerOptions(
#     base_options=python.BaseOptions(model_asset_path="hand_landmarker.task"),
#     running_mode=vision.RunningMode.VIDEO,
#     num_hands=2,
#     min_hand_detection_confidence=0.7,
#     min_tracking_confidence=0.5
# )
# hand_detector = vision.HandLandmarker.create_from_options(hand_options)

# # Pose Landmarker
# pose_options = vision.PoseLandmarkerOptions(
#     base_options=python.BaseOptions(model_asset_path="pose_landmarker.task"),
#     running_mode=vision.RunningMode.VIDEO,
#     min_pose_detection_confidence=0.7,
#     min_tracking_confidence=0.5
# )
# pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

# # ==============================
# # CAMERA
# # ==============================
# cap = cv2.VideoCapture(0)

# print("Press SPACE to start/pause recording")
# print("Press ESC to stop and save")

# timestamp = 0  # IMPORTANT: must be increasing

# # ==============================
# # MAIN LOOP
# # ==============================
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = cv2.flip(frame, 1)
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

#     timestamp += 1

#     pose_result = pose_detector.detect_for_video(mp_image, timestamp)
#     hand_result = hand_detector.detect_for_video(mp_image, timestamp)

#     # ==============================
#     # POSE FEATURES (18 values)
#     # ==============================
#     pose_features = []

#     if pose_result.pose_landmarks:
#         lm = pose_result.pose_landmarks[0]

#         for idx in [11, 12, 13, 14, 15, 16]:  # shoulders → wrists
#             pose_features.extend([lm[idx].x, lm[idx].y, lm[idx].visibility])
#     else:
#         pose_features = [0.0] * 18

#     # ==============================
#     # HAND FEATURES (84 values)
#     # ==============================
#     hand_features = []

#     for i in range(2):
#         if hand_result.hand_landmarks and i < len(hand_result.hand_landmarks):
#             lm = hand_result.hand_landmarks[i]

#             wrist_x, wrist_y = lm[0].x, lm[0].y

#             for p in lm:
#                 hand_features.extend([p.x - wrist_x, p.y - wrist_y])
#         else:
#             hand_features.extend([0.0] * 42)

#     # ==============================
#     # FINAL FEATURE VECTOR (102)
#     # ==============================
#     frame_data = pose_features + hand_features

#     # ==============================
#     # KEY CONTROLS
#     # ==============================
#     key = cv2.waitKey(1)

#     if key == 32:  # SPACE
#         RECORDING = not RECORDING
#         print("Recording..." if RECORDING else "Paused")

#     if key == 27:  # ESC
#         break

#     # ==============================
#     # RECORD DATA
#     # ==============================
#     if RECORDING:
#         SEQUENCE.append(frame_data)

#         cv2.putText(frame, f"REC {len(SEQUENCE)}",
#                     (20, 50),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     1,
#                     (0, 0, 255),
#                     2)

#     # ==============================
#     # DISPLAY
#     # ==============================
#     cv2.imshow("ASL Recorder (Pose + Hands)", frame)

# # ==============================
# # CLEANUP
# # ==============================
# cap.release()
# cv2.destroyAllWindows()
# hand_detector.close()
# pose_detector.close()

# # ==============================
# # SAVE DATA
# # ==============================
# if len(SEQUENCE) > 0:
#     label = input("Enter sign label (e.g. HELLO): ").upper()

#     os.makedirs(SAVE_DIR, exist_ok=True)
#     save_path = os.path.join(SAVE_DIR, f"{label}.npy")

#     np.save(save_path, np.array(SEQUENCE))
#     print(f"Saved {len(SEQUENCE)} frames for {label}")
# else:
#     print("No data recorded.")

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os

# ── Pose Landmarker (gives full body + face points) ──────────────────────────
base_options = python.BaseOptions(model_asset_path="pose_landmarker.task")
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.7,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
pose_detector = vision.PoseLandmarker.create_from_options(options)

# ── Hand Landmarker ───────────────────────────────────────────────────────────
base_options_h = python.BaseOptions(model_asset_path="hand_landmarker.task")
options_h = vision.HandLandmarkerOptions(
    base_options=base_options_h,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
hand_detector = vision.HandLandmarker.create_from_options(options_h)

cap = cv2.VideoCapture(0)
frames = []
recording = False
frame_timestamp = 0

print("Press SPACE to start/stop recording, ESC to save and quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    frame_timestamp += 33

    pose_result = pose_detector.detect_for_video(mp_image, frame_timestamp)
    hand_result = hand_detector.detect_for_video(mp_image, frame_timestamp)

    frame_data = []

    if pose_result.pose_landmarks:
        lm = pose_result.pose_landmarks[0]

        # Face landmarks: nose(0), eyes(1-6), ears(7-8), mouth(9-10)
        face_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for idx in face_indices:
            frame_data.extend([lm[idx].x, lm[idx].y, lm[idx].z])  # 11×3 = 33

        # Upper body: shoulders(11,12), elbows(13,14), wrists(15,16)
        for idx in [11, 12, 13, 14, 15, 16]:
            frame_data.extend([lm[idx].x, lm[idx].y, lm[idx].visibility])  # 6×3 = 18
    else:
        frame_data.extend([0.0] * 51)  # 33 + 18

    # Hands: 2 × 21 × 2 (wrist-normalized x,y) = 84
    for i in range(2):
        if i < len(hand_result.hand_landmarks):
            lm = hand_result.hand_landmarks[i]
            wrist_x, wrist_y = lm[0].x, lm[0].y
            for p in lm:
                frame_data.extend([p.x - wrist_x, p.y - wrist_y])
        else:
            frame_data.extend([0.0] * 42)

    # Total: 33 (face) + 18 (body) + 84 (hands) = 135 features

    key = cv2.waitKey(1)
    if key == 32:
        recording = not recording
        print("● Recording..." if recording else "■ Paused")
    if key == 27:
        break

    if recording:
        frames.append(frame_data)
        cv2.putText(frame, f"REC {len(frames)}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Record Sign", frame)

cap.release()
cv2.destroyAllWindows()
hand_detector.close()
pose_detector.close()

label = input("Enter sign label (e.g. HELLO): ").upper()
os.makedirs("reverse_dataset", exist_ok=True)
np.save(f"reverse_dataset/{label}.npy", np.array(frames, dtype=np.float32))
print(f"Saved {len(frames)} frames → reverse_dataset/{label}.npy")