# VisionBasedRoboticArmControls
# **Color Blob Arm Control**

This project connects computer vision and robotics. The program tracks color blobs using OpenCV and controls a robot arm to move toward or interact with the detected object. It’s designed to run with or without an actual arm connected, depending on the mode you choose.

---

## **How It Works**

The system uses your webcam feed to detect color blobs in real time. Once a blob is detected, it calculates the position of the object in the frame and translates that into movement commands for the robotic arm.

The tracking logic is adapted from the same method used in `colorBlob.py` — it detects colors using HSV masks, applies contour detection, and identifies the largest blob that matches the target color.

---

## **Files Overview**

* **`colorBlob.py`** – Handles the core color tracking logic. Detects color blobs and returns their coordinates.
* **`colorBlobArmControl.py`** – Uses the blob detection system to move a robotic arm toward the detected object.
* **`colorPickerTracking.py`** – Optional tool for calibrating color ranges. You can use it to find the best HSV values for your object.

---

## **How to Run**

### 1. Install Dependencies

Make sure Python and pip are set up, then install OpenCV and other required packages:

```bash
pip install opencv-python numpy
```

### 2. Calibrate Color Range

Run the color picker script to find HSV values for your target color:

```bash
python colorPickerTracking.py
```

Click on the color you want to track, then note the printed HSV range.

### 3. Update the Main Script

In `colorBlobArmControl.py`, replace the HSV values under:

```python
COLOR_LOWER = (H_low, S_low, V_low)
COLOR_UPPER = (H_high, S_high, V_high)
```

Use the range from your calibration step.

### 4. Run the Program

If you’re testing tracking only (no arm connection):

```bash
python colorBlobArmControl.py --mode tracking
```

If you’re running it with your robot arm connected:

```bash
python colorBlobArmControl.py --mode arm
```

---

## **How It Tracks Objects**

1. Captures frames from your webcam.
2. Converts each frame from BGR to HSV.
3. Masks out everything except pixels within your color range.
4. Finds contours in the mask and selects the largest one.
5. Draws a circle around the detected blob and computes its center position.
6. Sends movement commands to the arm (if connected) based on that position.

---

## **Notes**

* If tracking is unstable, tweak the HSV range slightly.
* Use good lighting to avoid color noise.
* The arm won’t move if it’s not connected — the program will just display tracking output.
* You can easily expand this to track multiple colors or integrate with other sensors.

---

Would you like me to include a short section at the end explaining how to modify it for multiple color tracking (e.g., red + yellow objects)?
