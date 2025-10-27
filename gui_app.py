import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
from colorBlob import AdvancedColorBlobDetector
from xarm.wrapper import XArmAPI

class VisionBasedRobotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Vision-Based Robotic Arm Control System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')
        
        # Initialize variables
        self.camera = None
        self.detector = None  # We'll create a simple detector
        self.robot_arm = None
        self.is_running = False
        self.current_mode = "visualization"
        self.selected_color = None
        self.tolerance = np.array([20, 80, 80])  # HSV tolerance
        
        # Robot parameters
        self.ARM_IP = '192.168.1.XXX'
        self.CAMERA_HEIGHT = 30
        self.SCALE = 0.1
        
        # IP validation
        self.ip_var = tk.StringVar(value=self.ARM_IP)
        self.ip_var.trace_add('write', self.validate_ip)
        
        # Create GUI elements
        self.create_widgets()
        self.setup_camera()
        
    def create_widgets(self):
        """Create all GUI widgets"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Vision-Based Robotic Arm Control", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Left panel - Controls
        self.create_control_panel(main_frame)
        
        # Center panel - Camera view
        self.create_camera_panel(main_frame)
        
        # Right panel - Status and info
        self.create_status_panel(main_frame)
        
    def create_control_panel(self, parent):
        """Create the control panel"""
        control_frame = ttk.LabelFrame(parent, text="Control Panel", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Camera controls
        camera_frame = ttk.LabelFrame(control_frame, text="Camera Controls", padding="5")
        camera_frame.pack(fill='x', pady=(0, 10))
        
        self.start_btn = ttk.Button(camera_frame, text="Start Camera", command=self.start_camera)
        self.start_btn.pack(fill='x', pady=2)
        
        self.stop_btn = ttk.Button(camera_frame, text="Stop Camera", command=self.stop_camera, state='disabled')
        self.stop_btn.pack(fill='x', pady=2)
        
        # Mode selection
        mode_frame = ttk.LabelFrame(control_frame, text="Operation Mode", padding="5")
        mode_frame.pack(fill='x', pady=(0, 10))
        
        self.mode_var = tk.StringVar(value="visualization")
        ttk.Radiobutton(mode_frame, text="Visualization Only", variable=self.mode_var, 
                       value="visualization", command=self.change_mode).pack(anchor='w')
        ttk.Radiobutton(mode_frame, text="Robot Control", variable=self.mode_var, 
                       value="robot", command=self.change_mode).pack(anchor='w')
        
        
        # Color selection
        color_frame = ttk.LabelFrame(control_frame, text="Color Selection", padding="5")
        color_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Button(color_frame, text="Pick Color from Camera", 
                  command=self.pick_color_from_camera).pack(fill='x', pady=2)
        
        # Robot controls
        robot_frame = ttk.LabelFrame(control_frame, text="Robot Controls", padding="5")
        robot_frame.pack(fill='x', pady=(0, 10))
        
        # IP Address input
        ip_frame = ttk.Frame(robot_frame)
        ip_frame.pack(fill='x', pady=2)
        ttk.Label(ip_frame, text="Robot IP:").pack(side='left')
        ip_entry = ttk.Entry(ip_frame, textvariable=self.ip_var)
        ip_entry.pack(side='left', fill='x', expand=True, padx=(5, 0))
        
        self.connect_robot_btn = ttk.Button(robot_frame, text="Connect Robot", 
                                           command=self.connect_robot)
        self.connect_robot_btn.pack(fill='x', pady=2)
        
        self.disconnect_robot_btn = ttk.Button(robot_frame, text="Disconnect Robot", 
                                              command=self.disconnect_robot, state='disabled')
        self.disconnect_robot_btn.pack(fill='x', pady=2)
        
        # Manual controls
        manual_frame = ttk.LabelFrame(control_frame, text="Manual Controls", padding="5")
        manual_frame.pack(fill='x')
        
        ttk.Button(manual_frame, text="Move to Selected Object", 
                  command=self.move_to_selected).pack(fill='x', pady=2)
        
        ttk.Button(manual_frame, text="Home Position", 
                  command=self.home_robot).pack(fill='x', pady=2)
        
    def create_camera_panel(self, parent):
        """Create the camera preview panel"""
        camera_frame = ttk.LabelFrame(parent, text="Camera Preview", padding="10")
        camera_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10)
        
        # Camera display
        self.camera_label = ttk.Label(camera_frame, text="Camera not started", 
                                     background='black', foreground='white')
        self.camera_label.pack(fill='both', expand=True)
        
        # Detection info
        info_frame = ttk.Frame(camera_frame)
        info_frame.pack(fill='x', pady=(10, 0))
        
        self.detection_label = ttk.Label(info_frame, text="No detection active")
        self.detection_label.pack(anchor='w')
        
        self.object_count_label = ttk.Label(info_frame, text="Objects detected: 0")
        self.object_count_label.pack(anchor='w')
        
    def create_status_panel(self, parent):
        """Create the status panel"""
        status_frame = ttk.LabelFrame(parent, text="Status & Information", padding="10")
        status_frame.grid(row=1, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        
        # Status display
        self.status_text = tk.Text(status_frame, height=15, width=30, wrap='word')
        status_scrollbar = ttk.Scrollbar(status_frame, orient='vertical', command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=status_scrollbar.set)
        
        self.status_text.pack(side='left', fill='both', expand=True)
        status_scrollbar.pack(side='right', fill='y')
        
        # Log initial status
        self.log_status("System initialized")
        self.log_status("Ready to start camera")
        
        # Detection parameters
        params_frame = ttk.LabelFrame(status_frame, text="Detection Parameters", padding="5")
        params_frame.pack(fill='x', pady=(10, 0))
        
        # Tolerance controls
        ttk.Label(params_frame, text="Color Tolerance:").pack(anchor='w')
        self.tolerance_frame = ttk.Frame(params_frame)
        self.tolerance_frame.pack(fill='x', pady=2)
        
        ttk.Label(self.tolerance_frame, text="H:").grid(row=0, column=0, padx=(0, 5))
        self.h_tolerance = tk.IntVar(value=20)
        ttk.Scale(self.tolerance_frame, from_=5, to=50, variable=self.h_tolerance, 
                 orient='horizontal', length=100).grid(row=0, column=1)
        
        ttk.Label(self.tolerance_frame, text="S:").grid(row=0, column=2, padx=(10, 5))
        self.s_tolerance = tk.IntVar(value=80)
        ttk.Scale(self.tolerance_frame, from_=20, to=150, variable=self.s_tolerance, 
                 orient='horizontal', length=100).grid(row=0, column=3)
        
        ttk.Label(self.tolerance_frame, text="V:").grid(row=1, column=0, padx=(0, 5))
        self.v_tolerance = tk.IntVar(value=80)
        ttk.Scale(self.tolerance_frame, from_=20, to=150, variable=self.v_tolerance, 
                 orient='horizontal', length=100).grid(row=1, column=1)
        
        # Apply tolerance button
        ttk.Button(params_frame, text="Apply Tolerance", 
                  command=self.apply_tolerance).pack(fill='x', pady=5)
    
        
    def setup_camera(self):
        """Initialize camera"""
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                self.log_status("Error: Could not open camera")
                return False
            self.log_status("Camera initialized successfully")
            return True
        except Exception as e:
            self.log_status(f"Camera initialization error: {str(e)}")
            return False
    
    def start_camera(self):
        """Start camera feed"""
        if not self.camera or not self.camera.isOpened():
            if not self.setup_camera():
                return
        
        self.is_running = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        
        self.log_status("Camera started")
        
        # Start camera thread
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()
    
    def stop_camera(self):
        """Stop camera feed"""
        self.is_running = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        
        self.log_status("Camera stopped")
        
        # Clear camera display
        self.camera_label.config(image='', text="Camera stopped")
    
    def camera_loop(self):
        """Main camera processing loop"""
        while self.is_running:
            ret, frame = self.camera.read()
            if not ret:
                continue
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Update GUI
            self.update_camera_display(processed_frame)
            
            time.sleep(0.03)  # ~30 FPS
    
    def process_frame(self, frame):
        """Process camera frame for detection"""
        display_frame = frame.copy()
        
        # Update tolerance if changed
        if hasattr(self, 'h_tolerance'):
            self.tolerance = np.array([self.h_tolerance.get(), self.s_tolerance.get(), self.v_tolerance.get()])
        
        # Simple color-based detection
        if self.selected_color is not None:
            display_frame, centroids = self.simple_color_detection(display_frame)
            
            # Update detection info
            if centroids:
                # Use the largest centroid
                largest_centroid = max(centroids, key=lambda c: c[2])  # c[2] is area
                cx, cy, area = largest_centroid
                self.detection_label.config(text=f"Target: ({cx}, {cy})")
                self.object_count_label.config(text=f"Objects detected: {len(centroids)}")
            else:
                self.detection_label.config(text="No objects detected")
                self.object_count_label.config(text="Objects detected: 0")
        else:
            self.detection_label.config(text="Click 'Pick Color from Camera' to start detection")
            self.object_count_label.config(text="Objects detected: 0")
        
        return display_frame
    
    def simple_color_detection(self, frame):
        """Simple, reliable color detection"""
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create color mask
        lower = np.clip(self.selected_color - self.tolerance, [0,0,0], [179,255,255])
        upper = np.clip(self.selected_color + self.tolerance, [0,0,0], [179,255,255])
        
        mask = cv2.inRange(hsv, lower, upper)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        centroids = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:  # Minimum area threshold
                # Calculate centroid
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Draw convex hull outline
                    hull = cv2.convexHull(cnt)
                    cv2.drawContours(frame, [hull], 0, (0, 255, 0), 2)
                    
                    # Draw centroid
                    cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)
                    cv2.circle(frame, (cx, cy), 7, (0, 0, 0), 2)
                    
                    # Draw coordinates
                    cv2.putText(frame, f"({cx},{cy})", (cx + 10, cy - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    centroids.append((cx, cy, area))
        
        return frame, centroids
    
    
    
    
    
    
    def update_camera_display(self, frame):
        """Update camera display in GUI"""
        # Resize frame to fit display
        height, width = frame.shape[:2]
        max_width, max_height = 640, 480
        
        if width > max_width or height > max_height:
            scale = min(max_width/width, max_height/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Convert to PhotoImage
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(image)
        
        # Update label
        self.camera_label.config(image=photo, text="")
        self.camera_label.image = photo  # Keep a reference
    
    def pick_color_from_camera(self):
        """Allow user to pick color from camera"""
        if not self.is_running:
            messagebox.showwarning("Warning", "Please start the camera first")
            return
        
        self.log_status("Click on camera to pick color")
        self.camera_label.bind("<Button-1>", self.on_camera_click)
    
    def on_camera_click(self, event):
        """Handle camera click for color picking"""
        if not self.is_running:
            return
        
        # Get click coordinates
        x, y = event.x, event.y
        
        # Capture current frame
        ret, frame = self.camera.read()
        if not ret:
            return
        
        # Convert to HSV and get color
        blurred = cv2.GaussianBlur(frame, (7, 7), 0)
        hsv_frame = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # Scale coordinates to actual frame size
        label_width = self.camera_label.winfo_width()
        label_height = self.camera_label.winfo_height()
        frame_height, frame_width = frame.shape[:2]
        
        actual_x = int(x * frame_width / label_width)
        actual_y = int(y * frame_height / label_height)
        
        # Ensure coordinates are within bounds
        actual_x = max(0, min(actual_x, frame_width - 1))
        actual_y = max(0, min(actual_y, frame_height - 1))
        
        # Get HSV color
        hsv_color = hsv_frame[actual_y, actual_x]
        self.selected_color = hsv_color
        
        self.log_status(f"Color selected: HSV({hsv_color[0]}, {hsv_color[1]}, {hsv_color[2]})")
        
        # Unbind click event
        self.camera_label.unbind("<Button-1>")
    
    
    
    def change_mode(self):
        """Change operation mode"""
        self.current_mode = self.mode_var.get()
        self.log_status(f"Mode changed to: {self.current_mode}")
    
    def validate_ip(self, *args):
        """Validate the IP address format"""
        ip = self.ip_var.get()
        if ip:
            # Simple IP format validation
            parts = ip.split('.')
            if len(parts) == 4 and all(part.isdigit() and 0 <= int(part) <= 255 for part in parts):
                self.ARM_IP = ip
                return True
        return False

    def connect_robot(self):
        """Connect to the robot arm"""
        if not self.validate_ip():
            messagebox.showerror("Invalid IP", "Please enter a valid IP address (format: xxx.xxx.xxx.xxx)")
            return
        try:
            self.robot_arm = XArmAPI(self.ARM_IP)
            self.robot_arm.connect()
            self.robot_arm.motion_enable(enable=True)
            self.robot_arm.set_mode(0)  # position control
            self.robot_arm.set_state(0)  # ready
            
            self.connect_robot_btn.config(state='disabled')
            self.disconnect_robot_btn.config(state='normal')
            
            self.log_status("Robot connected successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to connect to robot: {str(e)}")
            self.log_status(f"Robot connection failed: {str(e)}")
    
    def disconnect_robot(self):
        """Disconnect from robot arm"""
        if self.robot_arm:
            try:
                self.robot_arm.disconnect()
                self.robot_arm = None
                
                self.connect_robot_btn.config(state='normal')
                self.disconnect_robot_btn.config(state='disabled')
                
                self.log_status("Robot disconnected")
            except Exception as e:
                self.log_status(f"Robot disconnection error: {str(e)}")
    
    def move_to_selected(self):
        """Move robot to selected object"""
        if not self.robot_arm:
            messagebox.showwarning("Warning", "Please connect to robot first")
            return
        
        if not self.is_running:
            messagebox.showwarning("Warning", "Please start the camera first")
            return
        
        if self.selected_color is None:
            messagebox.showwarning("Warning", "Please select a color first")
            return
        
        # Get current frame and detect object
        ret, frame = self.camera.read()
        if not ret:
            return
        
        # Use simple detection
        _, centroids = self.simple_color_detection(frame)
        if centroids:
            # Use the largest centroid
            largest_centroid = max(centroids, key=lambda c: c[2])  # c[2] is area
            cx, cy, area = largest_centroid
            
            # Calculate robot position
            rx = (cx - 320) * self.SCALE
            ry = (cy - 240) * self.SCALE
            rz = self.CAMERA_HEIGHT
            
            try:
                self.robot_arm.set_position(rx, ry, rz, speed=50, wait=True)
                self.log_status(f"Robot moved to: ({rx:.2f}, {ry:.2f}, {rz})")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to move robot: {str(e)}")
                self.log_status(f"Robot movement failed: {str(e)}")
        else:
            messagebox.showwarning("Warning", "No object detected to move to")
    
    def home_robot(self):
        """Move robot to home position"""
        if not self.robot_arm:
            messagebox.showwarning("Warning", "Please connect to robot first")
            return
        
        try:
            self.robot_arm.set_position(0, 0, self.CAMERA_HEIGHT, speed=50, wait=True)
            self.log_status("Robot moved to home position")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to move robot home: {str(e)}")
    
    def apply_tolerance(self):
        """Apply tolerance settings"""
        self.tolerance = np.array([self.h_tolerance.get(), self.s_tolerance.get(), self.v_tolerance.get()])
        self.log_status(f"Tolerance updated: H={self.tolerance[0]}, S={self.tolerance[1]}, V={self.tolerance[2]}")
    
    def log_status(self, message):
        """Log status message"""
        timestamp = time.strftime("%H:%M:%S")
        self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.status_text.see(tk.END)
        self.root.update_idletasks()
    
    def on_closing(self):
        """Handle application closing"""
        self.is_running = False
        if self.camera:
            self.camera.release()
        if self.robot_arm:
            self.robot_arm.disconnect()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = VisionBasedRobotGUI(root)
    
    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    root.mainloop()

if __name__ == "__main__":
    main()
