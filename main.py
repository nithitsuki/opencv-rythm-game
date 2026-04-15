import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import math
import time
import random
import numpy as np
import mediapipe as mp
from pathlib import Path

# MediaPipe Setup
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker_float16.task"
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

import wave
import subprocess

# Secure pure-python Audio Synthesizer (No volatile C-extensions)
sample_rate = 44100
sound_dir = PROJECT_ROOT / "sounds"
sound_dir.mkdir(exist_ok=True)

class AsyncWavPlayer:
    def __init__(self, seq, filename, wave_type='square'):
        self.filepath = str(sound_dir / filename)
        
        # 1) Synthesize Sound Math into Numpy Arrays
        parts = []
        for freq, duration, pvol in seq:
            if freq == 0:
                parts.append(np.zeros(int(sample_rate * duration)))
            else:
                t = np.linspace(0, duration, int(sample_rate * duration), False)
                if wave_type == 'sine':
                    wave_data = np.sin(freq * t * 2 * np.pi)
                else:
                    wave_data = np.sign(np.sin(freq * t * 2 * np.pi))
                
                # Envelope to prevent clicking
                ramp = int(sample_rate * 0.01)
                env = np.ones_like(t)
                if len(env) > ramp * 2:
                    env[:ramp] = np.linspace(0, 1, ramp)
                    env[-ramp:] = np.linspace(1, 0, ramp)
                
                parts.append(wave_data * env * pvol)
        
        # Convert to 16-bit PCM Audio
        audio = (np.concatenate(parts) * 32767).astype(np.int16)
        
        # 2) Save to pure .wav file
        with wave.open(self.filepath, 'w') as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(sample_rate)
            f.writeframes(audio.tobytes())

    def play(self):
        # 3) Safely pass local file to robust Linux audio server subprocess
        try:
            # `aplay` natively hooks ALSA. Can swap to `paplay` (PulseAudio) if you want.
            subprocess.Popen(['aplay', '-q', self.filepath], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

print("Pre-rendering synthesized sounds...")
sfx_hit = AsyncWavPlayer([(880, 0.05, 0.4), (1760, 0.08, 0.4)], 'hit.wav', 'sine')
sfx_miss = AsyncWavPlayer([(150, 0.25, 0.5)], 'miss.wav', 'square')
sfx_levelup = AsyncWavPlayer([(440, 0.1, 0.3), (554, 0.1, 0.3), (659, 0.1, 0.3), (880, 0.4, 0.4)], 'levelup.wav', 'square')

music_freqs = [261.63, 311.13, 349.23, 392.00, 349.23, 311.13]
music_notes = [AsyncWavPlayer([(f, 0.15, 0.15)], f'note_{i}.wav', 'square') for i, f in enumerate(music_freqs)]

# -------------------- Game Logic --------------------

def distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def detect_shapes(hand_landmarks_list):
    for hand_landmarks in hand_landmarks_list:
        if distance(hand_landmarks[4], hand_landmarks[8]) < 0.04:
            return "Circle"
    if len(hand_landmarks_list) == 2:
        h1, h2 = hand_landmarks_list[0], hand_landmarks_list[1]
        
        if distance(h1[4], h2[4]) < 0.05 and distance(h1[8], h2[8]) < 0.05:
            if distance(h1[4], h1[8]) > 0.08:
                return "Triangle"
                
        if distance(h1[6], h2[6]) < 0.05 and distance(h1[8], h2[8]) > 0.05:
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

def draw_shape_icon(frame, shape_type, x, y, size):
    thickness = 4
    if shape_type == "Circle":
        cv2.circle(frame, (x, y), size, (0, 165, 255), thickness)
    elif shape_type == "Triangle":
        pts = np.array([
            [x, y - size],
            [x - int(size*0.866), y + int(size*0.5)],
            [x + int(size*0.866), y + int(size*0.5)]
        ], np.int32)
        cv2.polylines(frame, [pts], True, (0, 255, 0), thickness)
    elif shape_type == "Cross":
        offset = int(size*0.7)
        cv2.line(frame, (x - offset, y - offset), (x + offset, y + offset), (0, 0, 255), thickness)
        cv2.line(frame, (x + offset, y - offset), (x - offset, y + offset), (0, 0, 255), thickness)

def main():
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    W, H = 640, 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    # Gameplay Variables
    notes = [] 
    speed = 6
    spawn_timer = 0
    spawn_interval = 45 
    score = 0
    combo = 0
    
    # Levels and Timing
    level = 1
    level_start = time.time()
    LEVEL_DURATION = 30 # seconds

    # Strict Hit System
    previous_detected = "None"
    shape_formed_time = time.time()
    ALLOWED_HOLD_DURATION = 1.0 # Max time to hold a shape before it becomes "Early"

    hit_msg = ""
    hit_timer = 0
    too_early_timer = 0
    music_idx = 0
    
    ALLOWED_SHAPES = ["Circle", "Triangle", "Cross"]
    TARGET_Y = H - 100
    HIT_TOLERANCE = 50

    last_ts = 0
    print("Welcome to Symbol Rhythm Game!")
    
    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            current_time = time.time()
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            fH, fW, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            ts = int(time.perf_counter() * 1000)
            if ts <= last_ts: ts = last_ts + 1
            last_ts = ts
            
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_img, ts)
            
            detected = "None"
            if result.hand_landmarks:
                draw_landmarks(frame, result.hand_landmarks)
                detected = detect_shapes(result.hand_landmarks)
            
            # Check edge trigger for shapes
            if detected != previous_detected:
                shape_formed_time = current_time
                previous_detected = detected

            # Level Progression
            if current_time - level_start > LEVEL_DURATION:
                level += 1
                level_start = current_time
                speed += 2
                spawn_interval = max(15, spawn_interval - 5)
                hit_msg = f"LEVEL {level}!"
                hit_timer = 45
                sfx_levelup.play()

            # Draw target hit zone
            cv2.line(frame, (0, TARGET_Y), (fW, TARGET_Y), (255, 255, 255), 2)
            
            # Spawn notes & play beat
            spawn_timer += 1
            if spawn_timer >= spawn_interval:
                spawn_timer = 0
                shape = random.choice(ALLOWED_SHAPES)
                x_pos = random.randint(100, fW - 100)
                notes.append({'type': shape, 'x': x_pos, 'y': -50, 'hit': False, 'scored': False})
                
                # Play Synth background music note matched to spawn beat
                music_notes[music_idx].play()
                music_idx = (music_idx + 1) % len(music_notes)

            # Update and Draw notes
            for note in notes[:]:
                note['y'] += speed
                if note['hit']: continue
                
                draw_shape_icon(frame, note['type'], note['x'], note['y'], 30)

                # Check Hit
                if abs(note['y'] - TARGET_Y) <= HIT_TOLERANCE:
                    if detected == note['type']:
                        time_held = current_time - shape_formed_time
                        if time_held < ALLOWED_HOLD_DURATION:
                            # VALID HIT
                            note['hit'] = True
                            note['scored'] = True
                            score += (10 * (1 + combo))
                            combo += 1
                            hit_msg = "PERFECT!"
                            hit_timer = 15
                            sfx_hit.play()
                        else:
                            # POORLY TIMED (Held too long before arriving)
                            too_early_timer = 5
                            cv2.putText(frame, "HELD TOO LONG!", (fW // 2 - 200, TARGET_Y - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
                # Check Miss
                if note['y'] - TARGET_Y > HIT_TOLERANCE and not note['hit']:
                    note['hit'] = True
                    combo = 0
                    hit_msg = "MISS!"
                    hit_timer = 15
                    sfx_miss.play()

            # Remove off-screen notes
            notes = [n for n in notes if n['y'] < fH + 100 or (n.get('scored') and n['y'] > TARGET_Y + HIT_TOLERANCE + 20)]

            # Draw UI
            cv2.putText(frame, f"Score: {score}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame, f"Combo: {combo}x", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"Level: {level}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            cv2.putText(frame, f"Detecting: {detected}", (fW - 350, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

            time_held = current_time - shape_formed_time
            if detected != "None":
                color = (0, 255, 0) if time_held < ALLOWED_HOLD_DURATION else (0, 0, 255)
                cv2.putText(frame, f"Hold Time: {time_held:.1f}s", (fW - 350, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            if hit_timer > 0:
                if "LEVEL" in hit_msg:
                    color = (255, 0, 255)
                else:
                    color = (0, 255, 0) if hit_msg == "PERFECT!" else (0, 0, 255)
                cv2.putText(frame, hit_msg, (fW // 2 - 120, fH // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)
                hit_timer -= 1

            cv2.imshow("Hand Shape Rhythm Game", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
