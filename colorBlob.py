import cv2
import numpy as np
from collections import deque
from scipy import ndimage
from scipy.spatial.distance import cdist

# --- Parameters ---
MAX_HISTORY = 15
tolerance = np.array([20, 80, 80])
MIN_AREA = 300
GAUSSIAN_SIGMA = 1.5
KALMAN_Q = 0.1  # Process noise
KALMAN_R = 0.5  # Measurement noise

class KalmanFilter:
    """Simple 2D Kalman filter for centroid tracking"""
    def __init__(self, x=0, y=0):
        # State: [x, y, vx, vy]
        self.state = np.array([x, y, 0, 0], dtype=np.float32)
        
        # State transition matrix
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Process noise covariance
        self.Q = np.eye(4, dtype=np.float32) * KALMAN_Q
        
        # Measurement noise covariance
        self.R = np.eye(2, dtype=np.float32) * KALMAN_R
        
        # Error covariance
        self.P = np.eye(4, dtype=np.float32) * 1000
        
    def predict(self):
        """Predict next state"""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[:2]
    
    def update(self, measurement):
        """Update with measurement"""
        if measurement is None:
            return self.predict()
            
        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        y = measurement - self.H @ self.state
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        
        return self.state[:2]

class AdvancedColorBlobDetector:
    def __init__(self, max_history=MAX_HISTORY):
        self.centroid_history = deque(maxlen=max_history)
        self.selected_hsv = None
        self.tolerance = tolerance
        self.kalman_filter = None
        self.last_valid_centroid = None
        self.stability_counter = 0
        self.confidence_threshold = 0.7
        
    def set_color(self, hsv_value):
        """Set the target color for detection"""
        self.selected_hsv = hsv_value
        self.kalman_filter = None  # Reset filter
        self.last_valid_centroid = None
        self.stability_counter = 0
        print(f"Selected HSV: {self.selected_hsv}")
    
    def _preprocess_frame(self, frame):
        """Advanced preprocessing with bilateral filtering and edge preservation"""
        # Bilateral filter preserves edges while smoothing
        filtered = cv2.bilateralFilter(frame, 9, 75, 75)
        
        # Gaussian blur for noise reduction
        blurred = cv2.GaussianBlur(filtered, (9, 9), GAUSSIAN_SIGMA)
        
        return cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    def _create_robust_mask(self, hsv):
        """Create mask with advanced morphological operations"""
        # Create color mask
        lower = np.clip(self.selected_hsv - self.tolerance, [0,0,0], [179,255,255])
        upper = np.clip(self.selected_hsv + self.tolerance, [0,0,0], [179,255,255])
        mask = cv2.inRange(hsv, lower, upper)
        
        # Advanced morphological operations
        # Use different kernel sizes for different operations
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # Remove small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        # Fill gaps and holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)
        
        # Final smoothing
        mask = cv2.medianBlur(mask, 5)
        
        return mask
    
    def _find_dominant_blob(self, contours):
        """Find the most dominant blob using multiple criteria"""
        if not contours:
            return None, 0.0
        
        best_contour = None
        best_score = -1
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA:
                continue
            
            # Calculate multiple quality metrics
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            
            # Circularity (closer to 1 = more circular)
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Aspect ratio (closer to 1 = more square)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0
            
            # Solidity (area / convex hull area)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Combined score (weighted combination)
            score = (0.4 * circularity + 
                    0.3 * aspect_ratio + 
                    0.2 * solidity + 
                    0.1 * min(area / 10000, 1))  # Normalized area
            
            if score > best_score:
                best_score = score
                best_contour = cnt
        
        return best_contour, best_score if best_contour is not None else 0.0
    
    def _calculate_centroid_stability(self, new_centroid):
        """Calculate stability score for centroid"""
        if not self.centroid_history:
            return 1.0
        
        # Calculate distance to recent centroids
        recent_centroids = list(self.centroid_history)[-5:]  # Last 5
        distances = [np.linalg.norm(np.array(new_centroid) - np.array(c)) for c in recent_centroids]
        
        # Stability based on consistency
        avg_distance = np.mean(distances)
        stability = max(0, 1 - avg_distance / 50)  # Normalize by expected movement
        
        return stability
    
    def detect_blobs(self, frame):
        """Detect color blobs with advanced processing"""
        if self.selected_hsv is None:
            return []
        
        hsv = self._preprocess_frame(frame)
        mask = self._create_robust_mask(hsv)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find dominant blob
        best_contour, score = self._find_dominant_blob(contours)
        
        if best_contour is not None and score > 0.3:  # Quality threshold
            M = cv2.moments(best_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return [(cx, cy)]
        
        return []
    
    def get_smoothed_centroid(self, frame):
        """Get smoothed centroid using Kalman filtering and stability analysis"""
        if self.selected_hsv is None:
            return None
        
        # Detect current blob
        centroids = self.detect_blobs(frame)
        
        if not centroids:
            # No detection - use prediction
            if self.kalman_filter is not None:
                predicted = self.kalman_filter.predict()
                self.stability_counter = max(0, self.stability_counter - 1)
                
                # If we've been stable, use prediction
                if self.stability_counter > 5:
                    return tuple(map(int, predicted))
            return None
        
        # Use the first (and only) centroid
        current_centroid = centroids[0]
        
        # Initialize Kalman filter if needed
        if self.kalman_filter is None:
            self.kalman_filter = KalmanFilter(current_centroid[0], current_centroid[1])
            self.last_valid_centroid = current_centroid
            self.stability_counter = 1
            return current_centroid
        
        # Calculate stability
        stability = self._calculate_centroid_stability(current_centroid)
        
        # Update Kalman filter
        predicted = self.kalman_filter.predict()
        updated = self.kalman_filter.update(np.array(current_centroid))
        
        # Use Kalman filter output if stable, otherwise use raw detection
        if stability > self.confidence_threshold:
            self.stability_counter += 1
            self.last_valid_centroid = tuple(map(int, updated))
            return self.last_valid_centroid
        else:
            # Low stability - use prediction to avoid jitter
            self.stability_counter = max(0, self.stability_counter - 1)
            return tuple(map(int, predicted))
    
    def draw_debug_info(self, frame):
        """Draw debug information with advanced visualization"""
        if self.selected_hsv is None:
            return frame
        
        hsv = self._preprocess_frame(frame)
        mask = self._create_robust_mask(hsv)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find dominant blob
        best_contour, score = self._find_dominant_blob(contours)
        
        if best_contour is not None:
            # Draw convex hull with quality-based color
            hull = cv2.convexHull(best_contour)
            
            # Color based on quality score
            if score > 0.7:
                color = (0, 255, 0)  # Green - high quality
            elif score > 0.4:
                color = (0, 255, 255)  # Yellow - medium quality
            else:
                color = (0, 165, 255)  # Orange - low quality
            
            cv2.drawContours(frame, [hull], 0, color, 3)
            
            # Draw centroid
            M = cv2.moments(best_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Draw centroid with stability indicator
                stability = self._calculate_centroid_stability((cx, cy))
                radius = int(8 + stability * 4)  # Size based on stability
                
                cv2.circle(frame, (cx, cy), radius, (255, 255, 255), -1)
                cv2.circle(frame, (cx, cy), radius + 2, (0, 0, 0), 2)
                
                # Draw stability text
                cv2.putText(frame, f"Q:{score:.2f} S:{stability:.2f}", 
                           (cx + 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw Kalman prediction if available
        if self.kalman_filter is not None:
            predicted = self.kalman_filter.predict()
            pred_x, pred_y = int(predicted[0]), int(predicted[1])
            cv2.circle(frame, (pred_x, pred_y), 3, (255, 0, 255), -1)  # Magenta for prediction
        
        return frame

# Backward compatibility
ColorBlobDetector = AdvancedColorBlobDetector

# --- Standalone demo functionality ---
def run_color_picker_demo():
    """Run the original color picker demo"""
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera.")

    detector = AdvancedColorBlobDetector()
    
    def pick_color(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            ret, frame = cap.read()
            if not ret:
                return
            blurred = cv2.GaussianBlur(frame, (7,7), 0)
            hsv_frame = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            detector.set_color(hsv_frame[y, x])

    cv2.namedWindow("Advanced Color Picker")
    cv2.setMouseCallback("Advanced Color Picker", pick_color)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Draw debug info
        frame = detector.draw_debug_info(frame)
        
        cv2.imshow("Advanced Color Picker", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_color_picker_demo()
