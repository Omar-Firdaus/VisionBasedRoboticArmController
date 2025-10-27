import cv2
import numpy as np
from xarm.wrapper import XArmAPI
import time
from colorBlob import ColorBlobDetector

ARM_IP = '192.168.1.XXX'   
CAMERA_HEIGHT = 30          
SCALE = 0.1                 

# Mode selection
MODE_VISUALIZATION = "visualization"
MODE_ROBOT = "robot"

def initialize_robot():
    """Initialize robot connection"""
    try:
        arm = XArmAPI(ARM_IP)
        arm.connect()
        arm.motion_enable(enable=True)
        arm.set_mode(0)    # position control
        arm.set_state(0)   # ready
        print("Robot connected successfully!")
        return arm
    except Exception as e:
        print(f"Failed to connect to robot: {e}")
        return None

def move_robot_to_position(arm, cx, cy):
    """Move robot to the calculated position"""
    if arm is None:
        return False
    
    try:
        rx = (cx - 320) * SCALE
        ry = (cy - 240) * SCALE
        rz = CAMERA_HEIGHT
        
        print(f"Moving robot to: ({rx:.2f}, {ry:.2f}, {rz})")
        arm.set_position(rx, ry, rz, speed=50, wait=True)
        return True
    except Exception as e:
        print(f"Failed to move robot: {e}")
        return False

def run_visualization_mode():
    """Run in visualization-only mode (no robot connection required)"""
    print("=== VISUALIZATION MODE ===")
    print("This mode shows detected blobs and target positions without moving the robot.")
    print("Controls:")
    print("  'c' - Click on color to set detection target")
    print("  's' - Click mode: Click on blobs to select them")
    print("  't' - Track mode: Follow largest blob")
    print("  'q' - Quit")
    print("  'r' - Switch to robot mode (requires robot connection)")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Initialize color blob detector
    detector = ColorBlobDetector()
    
    selected_centroid = None
    current_mode = "click"
    
    def click_event(event, x, y, flags, param):
        nonlocal selected_centroid
        if current_mode != "click":
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            # Get current centroids from detector
            ret, frame = cap.read()
            if not ret:
                return
            centroids = detector.detect_blobs(frame)
            
            min_dist = float('inf')
            nearest = None
            for cx, cy in centroids:
                dist = np.hypot(cx - x, cy - y)
                if dist < min_dist:
                    min_dist = dist
                    nearest = (cx, cy)
            selected_centroid = nearest
            print(f"Selected blob at: {selected_centroid}")
    
    def pick_color_for_detection(event, x, y, flags, param):
        """Mouse callback to pick color for detection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            ret, frame = cap.read()
            if not ret:
                return
            blurred = cv2.GaussianBlur(frame, (7,7), 0)
            hsv_frame = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            detector.set_color(hsv_frame[y, x])
    
    cv2.namedWindow("Visualization Mode")
    cv2.setMouseCallback("Visualization Mode", lambda event, x, y, flags, param: 
                         click_event(event, x, y, flags, param) or pick_color_for_detection(event, x, y, flags, param))
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Use detector to find blobs and draw debug info
            frame = detector.draw_debug_info(frame)
            
            # Draw target position indicator
            target = None
            if current_mode == "click" and selected_centroid:
                target = selected_centroid
            elif current_mode == "track":
                target = detector.get_smoothed_centroid(frame)
            
            if target:
                cx, cy = target
                # Draw robot target position
                rx = (cx - 320) * SCALE
                ry = (cy - 240) * SCALE
                rz = CAMERA_HEIGHT
                
                # Draw crosshair at target
                cv2.drawMarker(frame, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
                cv2.putText(frame, f"Target: ({rx:.2f}, {ry:.2f}, {rz})", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("Visualization Mode", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                print("Click on a color to set new detection target")
                current_mode = "color_pick"
            elif key == ord('s'):
                current_mode = "click"
                print("Click mode: Click on blobs to select them")
            elif key == ord('t'):
                current_mode = "track"
                print("Track mode: Following largest blob")
            elif key == ord('r'):
                print("Switching to robot mode...")
                cap.release()
                cv2.destroyAllWindows()
                run_robot_mode()
                return
            
            if current_mode == "click":
                selected_centroid = None  # reset after each frame
    
    except KeyboardInterrupt:
        print("Exiting visualization mode...")
    finally:
        cap.release()
        cv2.destroyAllWindows()

def run_robot_mode():
    """Run in robot mode (requires robot connection)"""
    print("=== ROBOT MODE ===")
    print("This mode controls the robot arm based on detected blobs.")
    print("Controls:")
    print("  'c' - Click on color to set detection target")
    print("  's' - Click mode: Click on blobs to select them")
    print("  't' - Track mode: Follow largest blob")
    print("  'q' - Quit")
    print("  'v' - Switch to visualization mode")
    
    # Initialize robot
    arm = initialize_robot()
    if arm is None:
        print("Cannot run robot mode without robot connection!")
        print("Switching to visualization mode...")
        run_visualization_mode()
        return
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        arm.disconnect()
        return
    
    # Initialize color blob detector
    detector = ColorBlobDetector()
    
    selected_centroid = None
    current_mode = "click"
    
    def click_event(event, x, y, flags, param):
        nonlocal selected_centroid
        if current_mode != "click":
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            # Get current centroids from detector
            ret, frame = cap.read()
            if not ret:
                return
            centroids = detector.detect_blobs(frame)
            
            min_dist = float('inf')
            nearest = None
            for cx, cy in centroids:
                dist = np.hypot(cx - x, cy - y)
                if dist < min_dist:
                    min_dist = dist
                    nearest = (cx, cy)
            selected_centroid = nearest
            print(f"Selected blob at: {selected_centroid}")
    
    def pick_color_for_detection(event, x, y, flags, param):
        """Mouse callback to pick color for detection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            ret, frame = cap.read()
            if not ret:
                return
            blurred = cv2.GaussianBlur(frame, (7,7), 0)
            hsv_frame = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            detector.set_color(hsv_frame[y, x])
    
    cv2.namedWindow("Robot Mode")
    cv2.setMouseCallback("Robot Mode", lambda event, x, y, flags, param: 
                         click_event(event, x, y, flags, param) or pick_color_for_detection(event, x, y, flags, param))
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Use detector to find blobs and draw debug info
            frame = detector.draw_debug_info(frame)
            
            cv2.imshow("Robot Mode", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                print("Click on a color to set new detection target")
                current_mode = "color_pick"
            elif key == ord('s'):
                current_mode = "click"
                print("Click mode: Click on blobs to select them")
            elif key == ord('t'):
                current_mode = "track"
                print("Track mode: Following largest blob")
            elif key == ord('v'):
                print("Switching to visualization mode...")
                cap.release()
                cv2.destroyAllWindows()
                run_visualization_mode()
                return
            
            # Decide which centroid to use for robot movement
            target = None
            if current_mode == "click" and selected_centroid:
                target = selected_centroid
            elif current_mode == "track":
                # Use smoothed centroid for tracking
                target = detector.get_smoothed_centroid(frame)
            
            if target:
                cx, cy = target
                success = move_robot_to_position(arm, cx, cy)
                if not success:
                    print("Robot movement failed!")
                
                if current_mode == "click":
                    selected_centroid = None  # reset after one move
    
    except KeyboardInterrupt:
        print("Exiting robot mode...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if arm:
            arm.disconnect()

def main():
    """Main function to select mode"""
    print("=== Vision-Based Robotic Arm Control ===")
    print("Choose a mode:")
    print("1. Visualization Mode (no robot connection required)")
    print("2. Robot Mode (requires robot connection)")
    print("3. Exit")
    
    while True:
        choice = input("Enter your choice (1/2/3): ").strip()
        
        if choice == "1":
            run_visualization_mode()
            break
        elif choice == "2":
            run_robot_mode()
            break
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
