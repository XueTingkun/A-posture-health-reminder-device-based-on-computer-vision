import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import time
import cv2
import numpy as np


class PostureApp:
    def __init__(self, root):
        self.root = root

        self.root.title("Posture Guard AI - 坐姿健康提神器")
        self.root.geometry("1000x700")
        self.root.configure(bg="#1e272e")  # 深色背景

        # 定义现代配色方案
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

        # 自定义卡片样式
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

    def create_widgets(self):
        # 顶部标题栏
        self.header = tk.Frame(self.root, bg=self.colors["bg"], height=60)
        self.header.pack(fill=tk.X, padx=20, pady=10)

        tk.Label(
            self.header,
            text="POSTURE GUARD AI",
            bg=self.colors["bg"],
            foreground=self.colors["accent"],
            font=("Segoe UI", 24, "bold"),
        ).pack(side=tk.LEFT)
        self.time_label = tk.Label(
            self.header,
            text="",
            bg=self.colors["bg"],
            foreground=self.colors["text"],
            font=("Segoe UI", 12),
        )
        self.time_label.pack(side=tk.RIGHT)
        self.update_time()

        # 主容器
        self.main_container = tk.Frame(self.root, bg=self.colors["bg"])
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # 左侧：视频区域 (Canvas 模拟)
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
            text="等待摄像头接入...",
            fill=self.colors["text"],
            font=("Segoe UI", 14),
        )

        # 右侧：仪表盘
        self.dashboard = tk.Frame(self.main_container, bg=self.colors["bg"])
        self.dashboard.place(relx=0.67, rely=0, relwidth=0.33, relheight=0.95)

        # 状态卡片
        self.status_card = ttk.Frame(self.dashboard, style="Card.TFrame", padding=20)
        self.status_card.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(self.status_card, text="当前状态", style="Title.TLabel").pack(
            anchor=tk.W
        )
        self.status_var = tk.StringVar(value="准备就绪")
        self.status_display = ttk.Label(
            self.status_card, textvariable=self.status_var, style="Status.TLabel"
        )
        self.status_display.pack(pady=10)

        # 数据卡片
        self.metrics_card = ttk.Frame(self.dashboard, style="Card.TFrame", padding=20)
        self.metrics_card.pack(fill=tk.X, pady=10)

        ttk.Label(self.metrics_card, text="实时数据", style="Title.TLabel").pack(
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

        # 日志卡片
        self.log_card = ttk.Frame(self.dashboard, style="Card.TFrame", padding=20)
        self.log_card.pack(fill=tk.BOTH, expand=True, pady=10)

        ttk.Label(self.log_card, text="警报记录", style="Title.TLabel").pack(
            anchor=tk.W
        )
        self.history_list = tk.Listbox(
            self.log_card,
            bg=self.colors["bg"],
            fg=self.colors["text"],
            borderwidth=0,
            highlightthickness=0,
            font=("Consolas", 10),
        )
        self.history_list.pack(fill=tk.BOTH, expand=True, pady=10)

        # 按钮栏
        self.controls = tk.Frame(self.dashboard, bg=self.colors["bg"])
        self.controls.pack(fill=tk.X, pady=10)

        self.start_btn = tk.Button(
            self.controls,
            text="开始监测",
            command=self.start_monitoring,
            bg=self.colors["accent"],
            fg="white",
            font=("Segoe UI", 12, "bold"),
            relief=tk.FLAT,
            padx=20,
            pady=10,
        )
        self.start_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        self.stop_btn = tk.Button(
            self.controls,
            text="停止",
            command=self.stop_monitoring,
            bg=self.colors["error"],
            fg="white",
            font=("Segoe UI", 12, "bold"),
            relief=tk.FLAT,
            padx=20,
            pady=10,
            state=tk.DISABLED,
        )
        self.stop_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))

    def update_time(self):
        curr_time = time.strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.config(text=curr_time)
        self.root.after(1000, self.update_time)

    def start_monitoring(self):
        self.status_var.set("正在监测")
        self.status_display.configure(foreground=self.colors["success"])
        self.start_btn.config(state=tk.DISABLED, bg="#576574")
        self.stop_btn.config(state=tk.NORMAL, bg=self.colors["error"])
        self.add_log("系统已启动...")

    def stop_monitoring(self):
        self.status_var.set("已停止")
        self.status_display.configure(foreground="#8395a7")
        self.start_btn.config(state=tk.NORMAL, bg=self.colors["accent"])
        self.stop_btn.config(state=tk.DISABLED, bg="#576574")
        self.add_log("系统已停止。")

    def add_log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.history_list.insert(0, f"[{timestamp}] {message}")
        if self.history_list.size() > 50:
            self.history_list.delete(50, tk.END)

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
            metrics_text = f"Pitch: {metrics.get('pitch', 0.0):.1f}°\n"
            metrics_text += f"Roll: {metrics.get('roll', 0.0):.1f}°\n"
            metrics_text += f"Refresh rate: {metrics.get('fps', 0.0):.1f} FPS"
            self.metrics_text.set(metrics_text)

            # Add log entry
            if metadata.get("alert", False):
                alert_message = metadata.get("alert_message", "Poor posture detected")
                self.add_log(alert_message)
        except Exception as e:
            print(f"Update status error: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = PostureApp(root)
    root.mainloop()
