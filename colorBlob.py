import cv2
import numpy as np
from collections import deque

# --- Camera setup ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera.")

# --- Parameters ---
MAX_HISTORY = 10
centroid_history = deque(maxlen=MAX_HISTORY)
selected_hsv = None
tolerance = np.array([10, 50, 50])  # H, S, V tolerance

# --- Mouse callback to pick a color ---
def pick_color(event, x, y, flags, param):
    global selected_hsv
    if event == cv2.EVENT_LBUTTONDOWN:
        ret, frame = cap.read()
        if not ret:
            return
        blurred = cv2.GaussianBlur(frame, (7,7), 0)
        hsv_frame = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        selected_hsv = hsv_frame[y, x]
        print(f"Selected HSV: {selected_hsv}")

cv2.namedWindow("Color Picker")
cv2.setMouseCallback("Color Picker", pick_color)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    blurred = cv2.GaussianBlur(frame, (7,7), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    if selected_hsv is not None:
        lower = np.clip(selected_hsv - tolerance, [0,0,0], [179,255,255])
        upper = np.clip(selected_hsv + tolerance, [0,0,0], [179,255,255])

        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.medianBlur(mask, 5)
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) > 500:
                hull = cv2.convexHull(cnt)
                cv2.drawContours(frame, [hull], 0, (0,255,0), 2)

                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"]/M["m00"])
                    cy = int(M["m01"]/M["m00"])
                    centroid_history.append((cx,cy))
                    avg_cx = int(np.mean([c[0] for c in centroid_history]))
                    avg_cy = int(np.mean([c[1] for c in centroid_history]))

                    cv2.circle(frame, (avg_cx, avg_cy), 5, (255,255,255), -1)
                    cv2.putText(frame, f"({avg_cx},{avg_cy})", (avg_cx+10, avg_cy-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    cv2.imshow("Color Picker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
