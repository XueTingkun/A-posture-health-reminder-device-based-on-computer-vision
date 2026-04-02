import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import time

class PostureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("坐姿健康提神器 - Prototype")
        self.root.geometry("800x600")
        
        # 整体布局
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 左侧：视频显示区域 (目前用占位符)
        self.video_label = ttk.Label(self.main_frame, text="[视频流预览区域]", background="black", width=60, anchor="center")
        self.video_label.grid(row=0, column=0, rowspan=3, padx=10, pady=10, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        # 右侧：状态和控制
        self.status_frame = ttk.LabelFrame(self.main_frame, text="实时状态", padding="10")
        self.status_frame.grid(row=0, column=1, padx=10, pady=10, sticky=(tk.N, tk.E, tk.W))
        
        self.status_var = tk.StringVar(value="正常")
        self.status_display = ttk.Label(self.status_frame, textvariable=self.status_var, font=("Arial", 24), foreground="green")
        self.status_display.pack(pady=10)
        
        self.metrics_label = ttk.Label(self.status_frame, text="检测指标：\n- 歪头角度: 0.0°\n- 乌龟颈距离: 0.0px")
        self.metrics_label.pack(pady=10)
        
        # 警报记录
        self.history_frame = ttk.LabelFrame(self.main_frame, text="警报记录", padding="10")
        self.history_frame.grid(row=1, column=1, padx=10, pady=10, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        self.history_list = tk.Listbox(self.history_frame, height=10)
        self.history_list.pack(fill=tk.BOTH, expand=True)
        
        # 控制按钮
        self.btn_frame = ttk.Frame(self.main_frame, padding="10")
        self.btn_frame.grid(row=2, column=1, sticky=(tk.E, tk.W))
        
        self.start_btn = ttk.Button(self.btn_frame, text="开始监测", command=self.start_monitoring)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(self.btn_frame, text="停止", command=self.stop_monitoring, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

    def start_monitoring(self):
        self.status_var.set("正在监测...")
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.add_log("系统已启动...")
        
    def stop_monitoring(self):
        self.status_var.set("已停止")
        self.status_display.config(foreground="gray")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.add_log("系统已停止。")

    def add_log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.history_list.insert(0, f"[{timestamp}] {message}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PostureApp(root)
    root.mainloop()
