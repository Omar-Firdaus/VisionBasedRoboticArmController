import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading

# Preset HSV ranges for common colors
COLOR_PRESETS = {
    "Red":   ([0, 120, 70], [10, 255, 255]),
    "Green": ([35, 100, 100], [85, 255, 255]),
    "Blue":  ([100, 150, 0], [140, 255, 255]),
    "Yellow":([20, 100, 100], [30, 255, 255]),
    "Custom":([0, 0, 0], [179, 255, 255])
}

class ColorBlobTracker:
    def __init__(self, root):
        self.root = root
        self.root.title("Color Blob Tracker")
        self.root.geometry("900x700")
        self.root.configure(bg="#1e1e1e")

        # Camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot access camera.")

        # --- UI Layout ---
        self.video_label = tk.Label(self.root, bg="#1e1e1e")
        self.video_label.pack(pady=10)

        # Color selector
        frame_controls = tk.Frame(self.root, bg="#1e1e1e")
        frame_controls.pack()

        tk.Label(frame_controls, text="Select Color:", fg="white", bg="#1e1e1e").grid(row=0, column=0, padx=5)
        self.color_var = tk.StringVar(value="Red")
        color_menu = ttk.Combobox(frame_controls, textvariable=self.color_var, values=list(COLOR_PRESETS.keys()), width=10)
        color_menu.grid(row=0, column=1, padx=5)
        color_menu.bind("<<ComboboxSelected>>", self.set_preset)

        # Sliders for fine-tuning HSV range
        self.sliders = {}
        slider_names = ["LH", "LS", "LV", "UH", "US", "UV"]
        for i, name in enumerate(slider_names):
            tk.Label(frame_controls, text=name, fg="white", bg="#1e1e1e").grid(row=1 + i // 3, column=(i % 3) * 2, padx=5)
            self.sliders[name] = tk.Scale(frame_controls, from_=0, to=255 if 'S' in name or 'V' in name else 179,
                                          orient="horizontal", length=150, bg="#2b2b2b", fg="white",
                                          highlightthickness=0, troughcolor="#3a3a3a")
            self.sliders[name].grid(row=1 + i // 3, column=(i % 3) * 2 + 1, padx=5)
        
        self.set_preset()  # Initialize sliders with default

        # Exit button
        tk.Button(self.root, text="Exit", command=self.close, bg="#d9534f", fg="white", width=15).pack(pady=10)

        # Threaded video loop
        self.running = True
        self.thread = threading.Thread(target=self.video_loop, daemon=True)
        self.thread.start()

    def set_preset(self, event=None):
        color = self.color_var.get()
        lower, upper = COLOR_PRESETS[color]
        for (name, value) in zip(["LH", "LS", "LV"], lower):
            self.sliders[name].set(value)
        for (name, value) in zip(["UH", "US", "UV"], upper):
            self.sliders[name].set(value)

    def video_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            lower = np.array([self.sliders["LH"].get(), self.sliders["LS"].get(), self.sliders["LV"].get()])
            upper = np.array([self.sliders["UH"].get(), self.sliders["US"].get(), self.sliders["UV"].get()])

            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:
                    x, y, w, h = cv2.boundingRect(contour)
                    cx, cy = x + w // 2, y + h // 2
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"({cx}, {cy})", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Convert for Tkinter
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(frame))
            self.video_label.imgtk = img
            self.video_label.config(image=img)

        self.cap.release()

    def close(self):
        self.running = False
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ColorBlobTracker(root)
    root.mainloop()
