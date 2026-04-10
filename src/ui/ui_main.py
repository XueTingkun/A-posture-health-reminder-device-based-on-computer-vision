import time
import tkinter as tk
from tkinter import ttk
from queue import Queue
from PIL import Image, ImageTk
import cv2
import numpy as np


class PostureApp:
    def __init__(self, root, q_t2f: Queue):
        self.root = root
        self.q_t2f = q_t2f

        self.root.title("Posture Guard AI")
        self.root.geometry("1000x700")
        self.root.configure(bg="#1e272e")

        self.colors = {
            "bg": "#1e272e",
            "card": "#2f3640",
            "text": "#f5f6fa",
            "accent": "#00d2d3",
            "warning": "#ff9f43",
            "error": "#ee5253",
            "success": "#1dd1a1",
        }

        self.setup_styles()
        self.create_widgets()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("Card.TFrame", background=self.colors["card"], borderwidth=0)
        style.configure(
            "Title.TLabel",
            background=self.colors["card"],
            foreground=self.colors["text"],
            font=("Segoe UI", 18, "bold"),
        )
        style.configure(
            "Status.TLabel",
            background=self.colors["card"],
            foreground=self.colors["success"],
            font=("Segoe UI", 24, "bold"),
        )
        style.configure(
            "Metric.TLabel",
            background=self.colors["card"],
            foreground=self.colors["text"],
            font=("Segoe UI", 12),
        )
        style.configure("Modern.TButton", font=("Segoe UI", 10, "bold"))
        style.configure(
            "TCombobox",
            background=self.colors["card"],
            foreground=self.colors["text"],
            fieldbackground=self.colors["card"],
            selectbackground=self.colors["accent"],
            selectforeground="white",
            borderwidth=0,
            arrowcolor=self.colors["text"],
            font=("Segoe UI", 10),
        )
        style.configure(
            "Calibrate.TButton",
            background=self.colors["card"],
            foreground="white",
            borderwidth=0,
            focuscolor="none",
            padding=(20, 10),
            font=("Segoe UI", 12, "bold"),
        )

    def create_widgets(self):
        # Main Container
        self.main_container = tk.Frame(self.root, bg=self.colors["bg"])
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Left: Video
        self.video_container = tk.Frame(
            self.main_container,
            bg=self.colors["card"],
            highlightbackground=self.colors["accent"],
            highlightthickness=1,
        )
        self.video_container.place(relx=0, rely=0, relwidth=0.65, relheight=0.95)

        self.video_canvas = tk.Canvas(
            self.video_container, bg="black", highlightthickness=0
        )
        self.video_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.video_canvas.create_text(
            320,
            240,
            text="Waiting For Backend",
            fill=self.colors["text"],
            font=("Segoe UI", 14),
        )

        # Right: Dashboard
        self.dashboard = tk.Frame(self.main_container, bg=self.colors["bg"])
        self.dashboard.place(relx=0.67, rely=0, relwidth=0.33, relheight=0.95)

        tk.Label(
            self.dashboard,
            text="POSTURE GUARD AI",
            bg=self.colors["bg"],
            foreground=self.colors["accent"],
            font=("Segoe UI", 24, "bold"),
        ).pack(anchor=tk.W)

        # Status Card
        self.status_card = ttk.Frame(self.dashboard, style="Card.TFrame", padding=20)
        self.status_card.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(self.status_card, text="Status", style="Title.TLabel").pack(
            anchor=tk.W
        )
        self.status_var = tk.StringVar(value="Waiting Backend")
        self.status_display = ttk.Label(
            self.status_card, textvariable=self.status_var, style="Status.TLabel"
        )
        self.status_display.pack(pady=10)

        # Data Card
        self.metrics_card = ttk.Frame(self.dashboard, style="Card.TFrame", padding=20)
        self.metrics_card.pack(fill=tk.X, pady=10)

        ttk.Label(self.metrics_card, text="Realtime Data", style="Title.TLabel").pack(
            anchor=tk.W
        )
        self.metrics_text = tk.StringVar(
            value="Pitch: 0.0°\nRoll: 0.0°\nRefresh rate: 0.0 FPS"
        )
        ttk.Label(
            self.metrics_card,
            textvariable=self.metrics_text,
            style="Metric.TLabel",
            justify=tk.LEFT,
        ).pack(pady=10, anchor=tk.W)

        def _on_selected(event=None):
            selected_value = self.camera_combobox.get()
            if selected_value != self.camera_combobox.placeholder:
                self.combobox_cam_changed(selected_value)

        self.camera_combobox = ttk.Combobox(
            self.dashboard, values=[], state="readonly", style="TCombobox"
        )
        self.camera_combobox.bind("<<ComboboxSelected>>", _on_selected)
        self.camera_combobox.set("--Please select camera--")
        self.camera_combobox.placeholder = "--Please select camera--"
        self.camera_combobox.original_values = []
        self.camera_combobox.pack(in_=self.dashboard, fill=tk.X, pady=10)

        self.calibrate_btn = ttk.Button(
            self.dashboard,
            text="Calibrate Posture",
            command=self.calibrate_posture,
            style="Calibrate.TButton",
        )
        self.calibrate_btn.pack(in_=self.dashboard, fill=tk.X, pady=10)

    def update_time(self):
        curr_time = time.strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.config(text=curr_time)
        self.root.after(1000, self.update_time)

    def display_image(self, image_array):
        """Display image on video Canvas, maintaining original aspect ratio"""
        try:
            # If image_array is numpy array, convert to PIL Image
            if isinstance(image_array, np.ndarray):
                # Convert BGR to RGB (if in OpenCV format)
                if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

                # Convert to PIL Image
                pil_image = Image.fromarray(image_array)

                # Get Canvas dimensions
                canvas_width = self.video_canvas.winfo_width()
                canvas_height = self.video_canvas.winfo_height()

                # Get original image dimensions
                img_width, img_height = pil_image.size

                # Calculate scale ratio, maintain aspect ratio
                scale = min(canvas_width / img_width, canvas_height / img_height)

                # Calculate scaled dimensions
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)

                # Resize image, maintain aspect ratio
                pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)

                # Calculate image centered position in Canvas
                x_offset = (canvas_width - new_width) // 2
                y_offset = (canvas_height - new_height) // 2

                # Convert to Tkinter compatible image
                tk_image = ImageTk.PhotoImage(pil_image)

                # Clear Canvas and display new image, centered
                self.video_canvas.delete("all")
                self.video_canvas.create_image(
                    x_offset, y_offset, anchor=tk.NW, image=tk_image
                )

                # Keep reference to prevent garbage collection
                self.video_canvas.image = tk_image
        except Exception as e:
            print(f"Display image error: {e}")

    def update_status(self, metadata):
        """Update status card and dashboard data"""
        try:
            # Update status text
            status_text = metadata.get("status_text", "Unknown")
            self.status_var.set(status_text)

            # Set color based on status
            if status_text == "Normal":
                self.status_display.configure(foreground=self.colors["success"])
            elif status_text == "Warning":
                self.status_display.configure(foreground=self.colors["warning"])
            elif status_text == "Error":
                self.status_display.configure(foreground=self.colors["error"])

            # Update dashboard metrics
            metrics = metadata.get("metrics", {})
            metrics_text = f"Pitch: {metrics.get('pitch', 0.0):.1f}° dPitch: {metrics.get('d_pitch', 0.0):.1f}°\n"
            metrics_text += f"Roll: {metrics.get('roll', 0.0):.1f}° dRoll: {metrics.get('d_roll', 0.0):.1f}°\n"
            metrics_text += f"Refresh rate: {metrics.get('fps', 0.0):.1f} FPS"
            self.metrics_text.set(metrics_text)

            # Add log entry
            if metadata.get("alert", False):
                alert_message = metadata.get("alert_message", "Poor posture detected")
        except Exception as e:
            print(f"Update status error: {e}")

    def combobox_cam_changed(self, selected_value):
        """Handle camera selection change"""
        self.q_t2f.put_nowait({"cmd": "change_camera", "camera": selected_value})

    def set_camera_combobox_values(self, values):
        if self.camera_combobox:
            # Store the current selection
            current_selection = self.camera_combobox.get()

            # Update the values
            self.camera_combobox["values"] = values
            self.camera_combobox.original_values = values

            # If the current selection is not in the new values, reset to placeholder
            if (
                current_selection not in values
                and current_selection != self.camera_combobox.placeholder
            ):
                self.camera_combobox.set(self.camera_combobox.placeholder)

    def calibrate_posture(self):
        """Handle posture calibration"""
        self.q_t2f.put_nowait({"cmd": "calibrate_posture"})
        self.status_var.set("校准中...")


if __name__ == "__main__":
    root = tk.Tk()
    app = PostureApp(root)
    root.mainloop()
