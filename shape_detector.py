import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import math
import time
import mediapipe as mp
from pathlib import Path

# Setup Model paths
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker_float16.task"

# MediaPipe aliases
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)

def distance(p1, p2):
    """Calculate Euclidean distance between two normalized landmarks."""
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def detect_shapes(hand_landmarks_list):
    """
    Detect shapes based on 3D/2D hand landmarks.
    Return a shape name: 'Circle', 'Triangle', 'Square', 'Cross', or 'None'
    """
    # 1. Check for Circle (OK sign on any single hand)
    for hand_landmarks in hand_landmarks_list:
        thumb_tip = hand_landmarks[4]
        index_tip = hand_landmarks[8]
        if distance(thumb_tip, index_tip) < 0.04:
            # Optionally check if middle, ring, pinky are extended to be strict "OK", 
            # but thumb+index close is usually enough.
            return "Circle"

    # Need exactly two hands for the other shapes
    if len(hand_landmarks_list) == 2:
        h1 = hand_landmarks_list[0]
        h2 = hand_landmarks_list[1]
        
        # Triangle: Thumbs touching, Indexes touching
        thumb_dist = distance(h1[4], h2[4])
        index_dist = distance(h1[8], h2[8])
        if thumb_dist < 0.05 and index_dist < 0.05:
            # Ensure it's not a circle (where thumb and index of same hand touch)
            if distance(h1[4], h1[8]) > 0.08:
                return "Triangle"

        # Square: Hand 1 Index touches Hand 2 Thumb AND Hand 2 Index touches Hand 1 Thumb
        sq_dist1 = distance(h1[8], h2[4])
        sq_dist2 = distance(h2[8], h1[4])
        if sq_dist1 < 0.05 and sq_dist2 < 0.05:
            return "Square"

        # Cross: Index fingers crossed. (Index PIP joints close, but tips further apart)
        # Using PIP (6) or DIP (7) joints for crossing
        pip_dist = distance(h1[6], h2[6])
        tip_dist = distance(h1[8], h2[8])
        if pip_dist < 0.05 and tip_dist > 0.05:
            return "Cross"

    return "None"

def draw_landmarks(image, hand_landmarks_list):
    h, w, _ = image.shape
    for hand_landmarks in hand_landmarks_list:
        points = []
        for lm in hand_landmarks:
            px, py = int(lm.x * w), int(lm.y * h)
            points.append((px, py))
            cv2.circle(image, (px, py), 2, (0, 255, 255), cv2.FILLED)
        
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(image, points[start_idx], points[end_idx], (0, 200, 0), 1)

def main():
    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}")
        print("Please run main.py first to download it, or check your models directory.")
        return

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2, # Essential: we need up to 2 hands for Triangle, Square, Cross
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    landmarker = HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    # To get stable shape readings, we can require it to be held for a few frames
    last_timestamp_ms = 0
    current_shape = "None"
    
    print("Shape Detector Started!")
    print("- Circle: Touch your thumb and index finger (OK sign)")
    print("- Triangle: Touch both thumbs together, and both index fingers together")
    print("- Square: Left index to Right thumb, AND Right index to Left thumb")
    print("- Cross: Cross your index fingers to make an X")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1) # Mirror for natural interaction
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        ts = int(time.perf_counter() * 1000)
        if ts <= last_timestamp_ms:
            ts = last_timestamp_ms + 1
        last_timestamp_ms = ts

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = landmarker.detect_for_video(mp_image, ts)
        
        shape = "None"
        if result.hand_landmarks:
            draw_landmarks(frame, result.hand_landmarks)
            shape = detect_shapes(result.hand_landmarks)

        if shape != "None":
            current_shape = shape
            
        # Draw the detected shape on the screen
        cv2.putText(frame, f"Shape: {shape}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0) if shape != "None" else (0, 0, 255), 3)

        cv2.imshow("Rhythm Game Shape Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
