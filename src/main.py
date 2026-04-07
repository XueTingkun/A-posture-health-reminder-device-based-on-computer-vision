import threading
import tkinter as tk

from utils.DoubleBuffer import DoubleBuffer
from utils.face import FaceLandmarkerApp
from ui.ui_main import PostureApp


def start_face(buffer: DoubleBuffer, stop_event: threading.Event):
    app = FaceLandmarkerApp()
    app.run(buffer, stop_event)


def start_tk(buffer: DoubleBuffer, stop_event: threading.Event, ui_app=None):
    """Image processing thread for UI updates"""
    while not stop_event.is_set():
        try:
            # Receive buffer_id and pass to release
            meta, img_view, ts, buf_id = buffer.read(timeout=0.1)

            # Update UI if app instance is provided
            if ui_app is not None:
                # Update UI in a thread-safe manner
                ui_app.root.after(0, lambda: ui_app.display_image(img_view))
                ui_app.root.after(0, lambda: ui_app.update_status(meta))

            # Release the correct buffer using buf_id
            buffer.release(buf_id)

        except TimeoutError:
            continue
        except Exception as e:
            print(f"[Consumer] Error: {e}")


if __name__ == "__main__":
    # Create UI application
    root = tk.Tk()
    app = PostureApp(root)

    # Initialize double buffer and stop flag
    db = DoubleBuffer(drop_frames=True)
    stop_flag = threading.Event()

    # Create producer and consumer threads
    t_face = threading.Thread(target=start_face, args=(db, stop_flag))
    t_tk = threading.Thread(target=start_tk, args=(db, stop_flag, app))

    # Start threads
    t_face.start()
    t_tk.start()

    # Launch UI main loop
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        # Stop all threads
        stop_flag.set()
        t_face.join()
        t_tk.join()
