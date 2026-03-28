# Gesture Recognition Controller

Control basic desktop actions using hand gestures and your webcam.

This project uses OpenCV + MediaPipe Hands for real-time hand landmark detection, then maps recognized gestures to system actions like mouse movement, click, scrolling, and volume control.

## Features

- Real-time hand tracking via webcam
- Gesture classification for common actions
- Smoothed cursor movement (reduced jitter)
- Gesture debouncing to avoid accidental repeats
- Visual overlay with FPS and active gesture
- Detection zone to reduce false triggers

## Supported Gestures

- Point (index finger): Move mouse cursor
- Peace (index + middle): Left click
- Index + middle spread: Right click
- Pinch (thumb + index close): Scroll
- Open hand: Alt+Tab
- Thumbs up: Volume up
- Four fingers: Volume down
- Rock (thumb + pinky): Mute toggle
- Fist: Safe/no action

## Tech Stack

- Python 3.10 or 3.11
- mediapipe==0.10.14
- opencv-python==4.9.0.80
- numpy==1.26.4
- pyautogui==0.9.54
- pynput==1.7.6

Windows extras:

- pycaw==20181226
- comtypes==1.4.1

macOS extras:

- pyobjc-framework-Quartz==10.3.1

## Why Version Pinning Matters

MediaPipe 0.10.14 is not compatible with NumPy 2.x and does not provide wheels for Python 3.12+ in this setup. Use the pinned requirements and Python 3.10/3.11.

## Installation

### 1) Verify Python version

```bash
python --version
```

Use Python 3.10 or 3.11.

### 2) Create virtual environment

```bash
python -m venv gesture-env
```

### 3) Activate virtual environment

Windows PowerShell:

```powershell
gesture-env\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source gesture-env/bin/activate
```

### 4) Install dependencies

Core dependencies:

```bash
pip install -r requirements.txt
```

Windows extras:

```bash
pip install -r requirements-windows.txt
```

macOS extras:

```bash
pip install -r requirements-mac.txt
```

### 5) Verify imports

```bash
python -c "import cv2, mediapipe, numpy, pyautogui, pynput; print('All good')"
```

## Run

```bash
python main.py
```

## Runtime Controls

- q: Quit
- d: Toggle debug overlay
- s: Print gesture statistics

## Project Structure

- main.py: App loop and module wiring
- hand_detector.py: MediaPipe hands wrapper
- gesture_engine.py: Gesture recognition logic
- action_mapper.py: Gesture-to-action execution
- utils.py: Shared config and overlay helpers
- requirements.txt: Core pinned dependencies
- requirements-windows.txt: Windows-only extras
- requirements-mac.txt: macOS-only extras

## Tips for Better Accuracy

- Keep your hand inside the on-screen detection zone
- Use good lighting and a simple background
- Keep hand roughly centered and 1-2 feet from the camera
- If gestures trigger too easily, increase cooldown and thresholds in utils.py

## Troubleshooting

- Camera not opening:
  - Close other apps using the webcam
  - Try a different camera index in main.py
- Gestures lagging:
  - Lower camera resolution in utils.py
  - Reduce background CPU load
- MediaPipe install errors:
  - Confirm Python is 3.10/3.11
  - Recreate venv and reinstall from pinned requirements

## Safety Note

This app can move your cursor and trigger inputs globally. Keep one hand off camera while testing and press q to exit immediately if needed.
