# Hand Shape Rhythm Game (OpenCV + MediaPipe)

A rhythm game played using hand gestures! Detects shapes like Circle, Triangle, and Cross using MediaPipe hand landmarks, scoring based on timing as falling notes drop down the screen.

### Attribution and Event Submission

This repository was originally forked from [Chirayu Chaudhari's repo](https://github.com/Champion2049/Hill_Climb_GestureBased) (see `git log` for details). It was used as a base for the submission for the event **Edge-Core: Bridging the Gap Between AI Models and Hardware** conducted by IEEE CIS on Wednesday, 15th April 2026.

## Gameplay Features

- **Gesture Detection:** Form shapes with your hands (Circle, Triangle, Cross) to hit falling notes.
- **Rhythm Mechanics:** Notes drop down the screen; hit them exactly when they cross the target line!
- **Dynamic Progression:** The game speeds up and notes spawn faster as you progress through levels.
- **Synthesized Audio:** Pure Python audio synthesizer (`numpy` + `wave`) played via Linux `aplay`.

## Setup & Run

Install the required dependencies:

```bash
pip install mediapipe opencv-python numpy
```

Start the game:

```bash
python main.py
```

## Quantized TFLite Pipeline

This project uses the MediaPipe Tasks `HandLandmarker` pipeline backed by a TFLite model bundle (`models/hand_landmarker_float16.task`). 

Press `q` to quit the game window.


P.S: There isn't really music or rythm just yet, so please imagine up cool music playing in the background! thx