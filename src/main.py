import threading
import tkinter as tk
from queue import Queue, Empty

from utils.DoubleBuffer import DoubleBuffer
from utils.face import FaceLandmarkerApp
from ui.ui_main import PostureApp


def start_face(
    db_f2t: DoubleBuffer, q_f2t: Queue, q_t2f: Queue, stop_event: threading.Event
):
    app = FaceLandmarkerApp()
    app.run(db_f2t, q_f2t, q_t2f, stop_event)


def start_tk(
    db_f2t: DoubleBuffer,
    q_f2t: Queue,
    q_t2f: Queue,
    stop_event: threading.Event,
    ui_app=None,
):
    """Image processing thread for UI updates"""
    while not stop_event.is_set():
        try:
            # Receive buffer_id and pass to release
            meta, img_view, ts, buf_id = db_f2t.read(timeout=0.1)

            # Update UI if app instance is provided
            if ui_app is not None:
                # Update UI in a thread-safe manner
                ui_app.root.after(0, lambda: ui_app.display_image(img_view))
                ui_app.root.after(0, lambda: ui_app.update_status(meta))

            # Release the correct buffer using buf_id
            db_f2t.release(buf_id)

        except TimeoutError:
            continue
        except Exception as e:
            print(f"[Consumer] Error: {e}")

        try:
            d = q_f2t.get(timeout=0.1)

            match d["cmd"]:
                case "camera_selection":
                    if ui_app is not None:
                        ui_app.root.after(
                            0, lambda: ui_app.set_camera_combobox_values(d["data"])
                        )
        except Empty:
            continue
        except Exception as e:
            print(f"[Consumer-q_t2f] Error: {e}")


if __name__ == "__main__":
    # Initialize double buffer and stop flag
    db_f2t = DoubleBuffer(drop_frames=True)
    q_f2t = Queue(maxsize=16)
    q_t2f = Queue(maxsize=16)
    stop_flag = threading.Event()

    # Create UI application
    root = tk.Tk()
    app = PostureApp(root, q_t2f)

    # Create producer and consumer threads
    t_face = threading.Thread(target=start_face, args=(db_f2t, q_f2t, q_t2f, stop_flag))
    t_tk = threading.Thread(
        target=start_tk, args=(db_f2t, q_f2t, q_t2f, stop_flag, app)
    )

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
