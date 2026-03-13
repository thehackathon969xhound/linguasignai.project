import cv2
import numpy as np
import os
import sys

# --- FORCE STABILITY ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

try:
    import tensorflow as tf
    # Using the standard import that works with the install above
    from tensorflow.keras.models import load_model
    import mediapipe as mp
    mp_holistic = mp.solutions.holistic
    mp_draw = mp.solutions.drawing_utils
except Exception as e:
    print(f"[ERROR] Dependency missing: {e}")
    sys.exit()

# --- 1. INITIALIZE ---
print("[SYSTEM] Waking up LinguaSign PRO Brain...")
try:
    model = load_model('action.h5')
    actions = np.array(os.listdir('Hackathon_Demo_Data'))
    actions.sort()
    print(f"[SYSTEM] Brain Online. Vocabulary: {len(actions)} words.")
except Exception as e:
    print(f"[ERROR] Brain load failed: {e}")
    sys.exit()

# --- 2. V2 EXTRACTOR ---
def extract_keypoints_v2(results, last_lh, last_rh):
    if results.pose_landmarks:
        ax, ay, az = results.pose_landmarks.landmark[0].x, results.pose_landmarks.landmark[0].y, results.pose_landmarks.landmark[0].z
    else: ax, ay, az = 0.0, 0.0, 0.0

    pose = np.array([[res.x-ax, res.y-ay, res.z-az, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)

    if results.left_hand_landmarks:
        lh = np.array([[res.x-ax, res.y-ay, res.z-az] for res in results.left_hand_landmarks.landmark]).flatten()
        last_lh = lh
    else: lh = last_lh if last_lh is not None else np.zeros(21*3)

    if results.right_hand_landmarks:
        rh = np.array([[res.x-ax, res.y-ay, res.z-az] for res in results.right_hand_landmarks.landmark]).flatten()
        last_rh = rh
    else: rh = last_rh if last_rh is not None else np.zeros(21*3)

    return np.concatenate([pose, lh, rh]), last_lh, last_rh

# --- 3. STYLING UTILS ---
def draw_styled_landmarks(image, results):
    if results.pose_landmarks:
        mp_draw.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_draw.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_draw.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

# --- 4. LIVE VARIABLES ---
sequence = []
last_lh, last_rh = None, None
display_word = "WAITING..."
confidence_threshold = 0.85 

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        image = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)
        
        draw_styled_landmarks(image, results)
        keypoints, last_lh, last_rh = extract_keypoints_v2(results, last_lh, last_rh)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            current_prob = res[np.argmax(res)]
            if current_prob > confidence_threshold:
                display_word = actions[np.argmax(res)].upper()

        cv2.rectangle(image, (0, 0), (1280, 80), (15, 15, 15), -1)
        cv2.putText(image, f'BIM: {display_word}', (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        cv2.imshow('LinguaSign AI', image)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()