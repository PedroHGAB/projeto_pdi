import cv2
import mediapipe as mp
from math import dist

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

zoom_factor = 1.0

min_zoom_factor = 1.0
max_zoom_factor = 3.0

initial_distance = None

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark

        thumb_tip = (int(landmarks[mp_hands.HandLandmark.THUMB_TIP].x * frame.shape[1]),
                     int(landmarks[mp_hands.HandLandmark.THUMB_TIP].y * frame.shape[0]))
        index_tip = (int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1]),
                     int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0]))

        current_distance = dist(thumb_tip, index_tip)

        if initial_distance is not None:
            zoom_factor *= current_distance / initial_distance

            zoom_factor = max(min_zoom_factor, min(max_zoom_factor, zoom_factor))

        initial_distance = current_distance

    zoomed_frame = cv2.resize(frame, None, fx=zoom_factor, fy=zoom_factor)

    cv2.imshow('Zoomed Frame', zoomed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
