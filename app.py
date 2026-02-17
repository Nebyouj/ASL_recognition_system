# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# import csv
# import copy
# import argparse
# import itertools
# from collections import Counter
# from collections import deque

# import cv2 as cv
# import numpy as np
# import mediapipe as mp

# from utils import CvFpsCalc
# from model import KeyPointClassifier
# from model import PointHistoryClassifier


# def get_args():
#     parser = argparse.ArgumentParser()

#     parser.add_argument("--device", type=int, default=0)
#     parser.add_argument("--width", help='cap width', type=int, default=960)
#     parser.add_argument("--height", help='cap height', type=int, default=540)

#     parser.add_argument('--use_static_image_mode', action='store_true')
#     parser.add_argument("--min_detection_confidence",
#                         help='min_detection_confidence',
#                         type=float,
#                         default=0.7)
#     parser.add_argument("--min_tracking_confidence",
#                         help='min_tracking_confidence',
#                         type=int,
#                         default=0.5)

#     args = parser.parse_args()

#     return args


# def main():
#     # Argument parsing #################################################################
#     args = get_args()

#     cap_device = args.device
#     cap_width = args.width
#     cap_height = args.height

#     use_static_image_mode = args.use_static_image_mode
#     min_detection_confidence = args.min_detection_confidence
#     min_tracking_confidence = args.min_tracking_confidence

#     use_brect = True

#     # Camera preparation ###############################################################
#     cap = cv.VideoCapture(cap_device)
#     cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
#     cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

#     # Model load #############################################################
#     mp_hands = mp.solutions.hands
#     hands = mp_hands.Hands(
#         static_image_mode=use_static_image_mode,
#         max_num_hands=1,
#         min_detection_confidence=min_detection_confidence,
#         min_tracking_confidence=min_tracking_confidence,
#     )

#     keypoint_classifier = KeyPointClassifier()

#     point_history_classifier = PointHistoryClassifier()

#     # Read labels ###########################################################
#     with open('model/keypoint_classifier/keypoint_classifier_label.csv',
#               encoding='utf-8-sig') as f:
#         keypoint_classifier_labels = csv.reader(f)
#         keypoint_classifier_labels = [
#             row[0] for row in keypoint_classifier_labels
#         ]
#     with open(
#             'model/point_history_classifier/point_history_classifier_label.csv',
#             encoding='utf-8-sig') as f:
#         point_history_classifier_labels = csv.reader(f)
#         point_history_classifier_labels = [
#             row[0] for row in point_history_classifier_labels
#         ]

#     # FPS Measurement ########################################################
#     cvFpsCalc = CvFpsCalc(buffer_len=10)

#     # Coordinate history #################################################################
#     history_length = 16
#     point_history = deque(maxlen=history_length)

#     # Finger gesture history ################################################
#     finger_gesture_history = deque(maxlen=history_length)

#     #  ########################################################################
#     mode = 0

#     while True:
#         fps = cvFpsCalc.get()

#         # Process Key (ESC: end) #################################################
#         key = cv.waitKey(10)
#         if key == 27:  # ESC
#             break
#         number, mode = select_mode(key, mode)

#         # Camera capture #####################################################
#         ret, image = cap.read()
#         if not ret:
#             break
#         image = cv.flip(image, 1)  # Mirror display
#         debug_image = copy.deepcopy(image)

#         # Detection implementation #############################################################
#         image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

#         image.flags.writeable = False
#         results = hands.process(image)
#         image.flags.writeable = True

#         #  ####################################################################
#         if results.multi_hand_landmarks is not None:
#             for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
#                                                   results.multi_handedness):
#                 # Bounding box calculation
#                 brect = calc_bounding_rect(debug_image, hand_landmarks)
#                 # Landmark calculation
#                 landmark_list = calc_landmark_list(debug_image, hand_landmarks)

#                 # Conversion to relative coordinates / normalized coordinates
#                 pre_processed_landmark_list = pre_process_landmark(
#                     landmark_list)
#                 pre_processed_point_history_list = pre_process_point_history(
#                     debug_image, point_history)
#                 # Write to the dataset file
#                 logging_csv(number, mode, pre_processed_landmark_list,
#                             pre_processed_point_history_list)

#                 # Hand sign classification
#                 hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
#                 if hand_sign_id == 2:  # Point gesture
#                     point_history.append(landmark_list[8])
#                 else:
#                     point_history.append([0, 0])

#                 # Finger gesture classification
#                 finger_gesture_id = 0
#                 point_history_len = len(pre_processed_point_history_list)
#                 if point_history_len == (history_length * 2):
#                     finger_gesture_id = point_history_classifier(
#                         pre_processed_point_history_list)

#                 # Calculates the gesture IDs in the latest detection
#                 finger_gesture_history.append(finger_gesture_id)
#                 most_common_fg_id = Counter(
#                     finger_gesture_history).most_common()

#                 # Drawing part
#                 debug_image = draw_bounding_rect(use_brect, debug_image, brect)
#                 debug_image = draw_landmarks(debug_image, landmark_list)
#                 debug_image = draw_info_text(
#                     debug_image,
#                     brect,
#                     handedness,
#                     keypoint_classifier_labels[hand_sign_id],
#                     point_history_classifier_labels[most_common_fg_id[0][0]],
#                 )
#         else:
#             point_history.append([0, 0])

#         debug_image = draw_point_history(debug_image, point_history)
#         debug_image = draw_info(debug_image, fps, mode, number)

#         # Screen reflection #############################################################
#         cv.imshow('Hand Gesture Recognition', debug_image)

#     cap.release()
#     cv.destroyAllWindows()


# def select_mode(key, mode):
#     number = -1
#     if 48 <= key <= 57:  # 0 ~ 9
#         number = key - 48
#     if key == 110:  # n
#         mode = 0
#     if key == 107:  # k
#         mode = 1
#     if key == 104:  # h
#         mode = 2
#     return number, mode


# def calc_bounding_rect(image, landmarks):
#     image_width, image_height = image.shape[1], image.shape[0]

#     landmark_array = np.empty((0, 2), int)

#     for _, landmark in enumerate(landmarks.landmark):
#         landmark_x = min(int(landmark.x * image_width), image_width - 1)
#         landmark_y = min(int(landmark.y * image_height), image_height - 1)

#         landmark_point = [np.array((landmark_x, landmark_y))]

#         landmark_array = np.append(landmark_array, landmark_point, axis=0)

#     x, y, w, h = cv.boundingRect(landmark_array)

#     return [x, y, x + w, y + h]
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
import time
from collections import Counter, deque

import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--model", type=str, default="hand_landmarker.task")
    
    parser.add_argument("--min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_presence_confidence", type=float, default=0.5)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)

    return parser.parse_args()

def main():
    args = get_args()

    # --- MediaPipe Tasks Initialization ---
    base_options = python.BaseOptions(model_asset_path=args.model)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=args.min_detection_confidence,
        min_hand_presence_confidence=args.min_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # Classifiers and Utilities
    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history
    history_length = 16
    point_history = deque(maxlen=history_length)
    finger_gesture_history = deque(maxlen=history_length)

    # Labels
    try:
        with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
            keypoint_labels = [row[0] for row in csv.reader(f)]
        with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
            point_labels = [row[0] for row in csv.reader(f)]
    except FileNotFoundError:
        print("Label files not found. Please check your directory structure.")
        return

    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    mode = 0

    while True:
        fps = cvFpsCalc.get()
        key = cv.waitKey(10)
        if key == 27: break
        number, mode = select_mode(key, mode)

        ret, frame = cap.read()
        if not ret: break
        
        frame = cv.flip(frame, 1)
        debug_image = copy.deepcopy(frame)

        # Convert frame for MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        
        timestamp_ms = int(time.time() * 1000)
        result = detector.detect_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks:
            for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                
                # Conversion to relative coordinates
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                brect = calc_bounding_rect(debug_image, hand_landmarks)

                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)

                logging_csv(number, mode, pre_processed_landmark_list, pre_processed_point_history_list)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 2:  # Point gesture
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                # Finger gesture classification
                finger_gesture_id = 0
                if len(pre_processed_point_history_list) == (history_length * 2):
                    finger_gesture_id = point_history_classifier(pre_processed_point_history_list)

                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(finger_gesture_history).most_common()

                # Drawing
                debug_image = draw_bounding_rect(True, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                
                # In Tasks API, handedness is a list of Category objects. 
                # Use handedness[0].category_name to get the label.
                h_label = handedness[0].category_name
                if h_label == "Left":
                    h_label = "Right"
                else:
                    h_label = "Left"
                
                debug_image = draw_info_text(
                    debug_image, brect, h_label, 
                    keypoint_labels[hand_sign_id],
                    point_labels[most_common_fg_id[0][0]]
                )
        else:
            point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()

# --- Helper Functions ---

def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110: mode = 0  # n
    if key == 107: mode = 1  # k
    if key == 104: mode = 2  # h
    return number, mode

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for landmark in landmarks:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)
    for landmark in landmarks:
        lx = min(int(landmark.x * image_width), image_width - 1)
        ly = min(int(landmark.y * image_height), image_height - 1)
        landmark_array = np.append(landmark_array, [np.array((lx, ly))], axis=0)
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))
    def normalize_(n):
        return n / max_value if max_value != 0 else n
    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list

def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]
    temp_point_history = copy.deepcopy(point_history)
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]
        temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height
    return list(itertools.chain.from_iterable(temp_point_history))

def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])

def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Define connection pairs for landmarks
        connections = [
            (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12),
            (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20),
            (0, 1), (1, 2), (2, 5), (5, 9), (9, 13), (13, 17), (17, 0)
        ]
        for start, end in connections:
            cv.line(image, tuple(landmark_point[start]), tuple(landmark_point[end]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[start]), tuple(landmark_point[end]), (255, 255, 255), 2)

    for index, landmark in enumerate(landmark_point):
        size = 8 if index in [4, 8, 12, 16, 20] else 5
        cv.circle(image, (landmark[0], landmark[1]), size, (255, 255, 255), -1)
        cv.circle(image, (landmark[0], landmark[1]), size, (0, 0, 0), 1)
    return image

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
    return image

def draw_info_text(image, brect, handedness_label, hand_sign_text, finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)
    info_text = handedness_label
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)
    return image

def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2), (152, 251, 152), 2)
    return image

def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)
    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image

if __name__ == '__main__':
    main()
