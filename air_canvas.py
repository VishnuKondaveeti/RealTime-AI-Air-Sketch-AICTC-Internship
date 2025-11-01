import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

canvas = np.zeros((720, 1280, 3), dtype=np.uint8) + 255
drawing_color = (0, 0, 0)
brush_thickness = 10
prev_x, prev_y = 0, 0

def is_index_finger_up(landmarks):
    tip_y = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    pip_y = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    return tip_y < pip_y

while cap.isOpened():
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            tip_of_index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, c = img.shape
            tip_x = int(tip_of_index.x * w)
            tip_y = int(tip_of_index.y * h)
            
            if is_index_finger_up(hand_landmarks):
                cv2.circle(img, (tip_x, tip_y), 10, (0, 255, 0), cv2.FILLED)

                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = tip_x, tip_y
                else:
                    cv2.line(canvas, (prev_x, prev_y), (tip_x, tip_y), drawing_color, brush_thickness)
                    prev_x, prev_y = tip_x, tip_y
            else:
                prev_x, prev_y = 0, 0

    img_combined = cv2.addWeighted(img, 0.5, canvas, 0.5, 0)
    
    cv2.imshow("RealTime-AI-Air-Sketch", img_combined)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
