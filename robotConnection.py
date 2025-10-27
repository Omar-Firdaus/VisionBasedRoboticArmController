import cv2
import numpy as np
from xarm.wrapper import XArmAPI
import time

# -------------------- CONFIG --------------------
ARM_IP = '192.168.1.XXX'   # Replace with your robot's IP
CAMERA_HEIGHT = 30          # cm above robot base
SCALE = 0.1                 # conversion from pixels to real cm (tune for your setup)

# Color range in HSV
COLOR_LOWER = np.array([0, 100, 100])
COLOR_UPPER = np.array([10, 255, 255])
MIN_AREA = 500

# Modes: "click" = click to move once, "track" = follow blob continuously
MODE = "click"

# -------------------- CONNECT TO ARM --------------------
arm = XArmAPI(ARM_IP)
arm.connect()
arm.motion_enable(enable=True)
arm.set_mode(0)    # position control
arm.set_state(0)   # ready

# -------------------- CAMERA SETUP --------------------
cap = cv2.VideoCapture(0)

selected_centroid = None

# Mouse callback for click-to-move mode
def click_event(event, x, y, flags, param):
    global selected_centroid
    if MODE != "click":
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        min_dist = float('inf')
        nearest = None
        for cx, cy in param:
            dist = np.hypot(cx - x, cy - y)
            if dist < min_dist:
                min_dist = dist
                nearest = (cx, cy)
        selected_centroid = nearest
        print(f"Selected blob at: {selected_centroid}")

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", click_event)

# -------------------- MAIN LOOP --------------------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, COLOR_LOWER, COLOR_UPPER)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centroids = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > MIN_AREA:
                x, y, w, h = cv2.boundingRect(cnt)
                cx, cy = x + w // 2, y + h // 2
                centroids.append((cx, cy))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"({cx},{cy})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        cv2.imshow("Frame", frame)
        cv2.waitKey(1)

        # Decide which centroid to use
        target = None
        if MODE == "click" and selected_centroid:
            target = selected_centroid
        elif MODE == "track" and centroids:
            # Use largest blob for tracking
            target = max(centroids, key=lambda c: cv2.contourArea(np.array([[c]])))

        if target:
            cx, cy = target
            rx = (cx - 320) * SCALE
            ry = (cy - 240) * SCALE
            rz = CAMERA_HEIGHT

            # Move arm
            arm.set_position(rx, ry, rz, speed=50, wait=True)

            if MODE == "click":
                selected_centroid = None  # reset after one move

except KeyboardInterrupt:
    print("Exiting...")
finally:
    cap.release()
    cv2.destroyAllWindows()
    arm.disconnect()
