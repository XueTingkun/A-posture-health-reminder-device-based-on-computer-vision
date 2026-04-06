import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import threading
import time
import os
import sys
from utils.DoubleBuffer import DoubleBuffer


class FaceLandmarkerApp:
    def __init__(self):
        # Initialize parameters
        self.cap = None
        self.landmarker = None
        self.running = False
        self.frame_timestamp_ms = 0

        # Store detection results
        self.detection_result = None
        self.latest_image = None
        self.lock = threading.Lock()

        # Model path
        self.model_path = "./models/face_landmarker.task"

        # Window settings
        self.window_name = "MediaPipe Face Landmarker"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        # Drawing color configuration
        self.MESH_COLOR = (0, 255, 0)  # Green mesh
        self.CONTOUR_COLOR = (255, 255, 255)  # White contour
        self.IRIS_COLOR = (0, 255, 255)  # Cyan iris
        self.POINT_COLOR = (0, 0, 255)  # Red key points

    def result_callback(self, result, output_image: mp.Image, timestamp_ms: int):
        """Asynchronous detection result callback function (executed in a separate thread)"""
        with self.lock:
            self.detection_result = result
            # Convert MediaPipe Image to numpy array
            self.latest_image = np.array(output_image.numpy_view())

    def create_landmarker(self):
        """Create Face Landmarker instance"""
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,  # Live stream mode
            num_faces=1,  # Maximum number of faces to detect
            min_face_detection_confidence=0.5,  # Face detection confidence threshold
            min_face_presence_confidence=0.5,  # Face presence confidence threshold
            min_tracking_confidence=0.5,  # Tracking confidence threshold
            output_face_blendshapes=True,  # Output expression blendshapes
            output_facial_transformation_matrixes=True,  # Output facial transformation matrix
            result_callback=self.result_callback,  # Result callback function
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        print("[INFO] Face Landmarker initialized")

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        """Draw face landmarks and mesh on image - manually implemented drawing functionality"""
        if not detection_result or not detection_result.face_landmarks:
            return rgb_image

        annotated_image = np.copy(rgb_image)
        height, width = annotated_image.shape[:2]
        face_landmarks_list = detection_result.face_landmarks

        for face_landmarks in face_landmarks_list:
            # Convert to pixel coordinates
            points = []
            for landmark in face_landmarks:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                points.append((x, y))

            # Draw triangular mesh (Tesselation)
            self.draw_mesh(annotated_image, points)

            # Draw facial contours
            self.draw_contours(annotated_image, points)

            # Draw iris
            self.draw_iris(annotated_image, points)

            # Draw key points
            for idx, (x, y) in enumerate(points):
                cv2.circle(annotated_image, (x, y), 1, self.POINT_COLOR, -1)

            # Draw key feature point indices
            self.draw_key_indices(annotated_image, points)

        return annotated_image

    def draw_mesh(self, image, points):
        """Draw face triangular mesh"""
        # FACEMESH_TESSELATION connection definitions (partial main connections)
        mesh_connections = [
            # Forehead region
            [10, 338],
            [338, 297],
            [297, 332],
            [332, 284],
            [284, 251],
            [251, 389],
            [389, 356],
            [356, 454],
            [454, 323],
            [323, 361],
            [361, 288],
            [288, 397],
            [397, 365],
            [365, 379],
            [379, 378],
            [378, 400],
            [400, 377],
            [377, 152],
            [152, 148],
            [148, 176],
            [176, 149],
            [149, 150],
            [150, 136],
            [136, 172],
            [172, 58],
            [58, 132],
            [132, 93],
            [93, 234],
            [234, 127],
            [127, 162],
            [162, 21],
            [21, 54],
            [54, 103],
            [103, 67],
            [67, 109],
            [109, 10],
            # Eye region
            [33, 7],
            [7, 163],
            [163, 144],
            [144, 145],
            [145, 153],
            [153, 154],
            [154, 155],
            [155, 133],
            [133, 246],
            [246, 161],
            [161, 160],
            [160, 159],
            [159, 158],
            [158, 157],
            [157, 173],
            [173, 133],
            [362, 382],
            [382, 381],
            [381, 380],
            [380, 374],
            [374, 373],
            [373, 390],
            [390, 249],
            [249, 263],
            [263, 466],
            [466, 388],
            [388, 387],
            [387, 386],
            [386, 385],
            [385, 384],
            [384, 398],
            [398, 362],
            # Nose region
            [168, 6],
            [6, 197],
            [197, 195],
            [195, 5],
            [5, 4],
            [4, 1],
            [1, 19],
            [19, 94],
            [94, 2],
            [2, 164],
            [164, 0],
            [0, 11],
            [11, 12],
            [12, 13],
            [13, 14],
            [14, 15],
            [15, 16],
            [16, 17],
            [17, 18],
            [18, 200],
            [200, 199],
            [199, 175],
            [175, 152],
            # Mouth region
            [61, 185],
            [185, 40],
            [40, 39],
            [39, 37],
            [37, 0],
            [0, 267],
            [267, 269],
            [269, 270],
            [270, 409],
            [409, 291],
            [291, 375],
            [375, 321],
            [321, 405],
            [405, 314],
            [314, 17],
            [17, 84],
            [84, 181],
            [181, 91],
            [91, 146],
            [146, 61],
            [78, 191],
            [191, 80],
            [80, 81],
            [81, 82],
            [82, 13],
            [13, 312],
            [312, 311],
            [311, 310],
            [310, 415],
            [415, 308],
            [308, 324],
            [324, 318],
            [318, 402],
            [402, 317],
            [317, 14],
            [14, 87],
            [87, 178],
            [178, 88],
            [88, 95],
            [95, 78],
            # Jaw contour
            [78, 95],
            [95, 88],
            [88, 178],
            [178, 87],
            [87, 14],
            [14, 317],
            [317, 402],
            [402, 318],
            [318, 324],
            [324, 308],
            [308, 415],
            [415, 310],
            [310, 311],
            [311, 312],
            [312, 13],
            [13, 82],
            [82, 81],
            [81, 80],
            [80, 191],
            [191, 78],
        ]

        for connection in mesh_connections:
            start_idx, end_idx = connection
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(image, points[start_idx], points[end_idx], self.MESH_COLOR, 1)

    def draw_contours(self, image, points):
        """Draw facial contours"""
        # Face outer contour
        face_outline = [
            10,
            338,
            297,
            332,
            284,
            251,
            389,
            356,
            454,
            323,
            361,
            288,
            397,
            365,
            379,
            378,
            400,
            377,
            152,
            148,
            176,
            149,
            150,
            136,
            172,
            58,
            132,
            93,
            234,
            127,
            162,
            21,
            54,
            103,
            67,
            109,
            10,
        ]

        for i in range(len(face_outline) - 1):
            idx1, idx2 = face_outline[i], face_outline[i + 1]
            if idx1 < len(points) and idx2 < len(points):
                cv2.line(image, points[idx1], points[idx2], self.CONTOUR_COLOR, 2)

        # Left eyebrow
        left_brow = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
        for i in range(len(left_brow) - 1):
            idx1, idx2 = left_brow[i], left_brow[i + 1]
            if idx1 < len(points) and idx2 < len(points):
                cv2.line(image, points[idx1], points[idx2], self.CONTOUR_COLOR, 2)

        # Right eyebrow
        right_brow = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
        for i in range(len(right_brow) - 1):
            idx1, idx2 = right_brow[i], right_brow[i + 1]
            if idx1 < len(points) and idx2 < len(points):
                cv2.line(image, points[idx1], points[idx2], self.CONTOUR_COLOR, 2)

        # Nose
        nose = [
            168,
            6,
            197,
            195,
            5,
            4,
            1,
            19,
            94,
            2,
            164,
            0,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            200,
            199,
            175,
            152,
        ]
        for i in range(len(nose) - 1):
            idx1, idx2 = nose[i], nose[i + 1]
            if idx1 < len(points) and idx2 < len(points):
                cv2.line(image, points[idx1], points[idx2], self.CONTOUR_COLOR, 1)

    def draw_iris(self, image, points):
        """Draw iris"""
        # Left eye iris
        left_iris = [468, 469, 470, 471, 472]
        for i in range(len(left_iris)):
            idx1 = left_iris[i]
            idx2 = left_iris[(i + 1) % len(left_iris)]
            if idx1 < len(points) and idx2 < len(points):
                cv2.line(image, points[idx1], points[idx2], self.IRIS_COLOR, 1)
                cv2.circle(image, points[idx1], 2, self.IRIS_COLOR, -1)

        # Right eye iris
        right_iris = [473, 474, 475, 476, 477]
        for i in range(len(right_iris)):
            idx1 = right_iris[i]
            idx2 = right_iris[(i + 1) % len(right_iris)]
            if idx1 < len(points) and idx2 < len(points):
                cv2.line(image, points[idx1], points[idx2], self.IRIS_COLOR, 1)
                cv2.circle(image, points[idx1], 2, self.IRIS_COLOR, -1)

    def draw_key_indices(self, image, points):
        """Draw indices on key feature points"""
        key_indices = {
            1: "Tip of the nose",
            33: "Inner corner of the left eye",
            133: "Outer corner of the left eye",
            362: "Inner corner of the right eye",
            263: "Outer corner of the right eye",
            61: "Left corner of the mouth",
            291: "Right corner of the mouth",
            0: "Middle of the upper lip",
            17: "Middle of the lower lip",
            152: "Chin",
        }

        for idx, label in key_indices.items():
            if idx < len(points):
                x, y = points[idx]
                cv2.circle(image, (x, y), 3, (0, 255, 255), -1)
                cv2.putText(
                    image,
                    str(idx),
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 255),
                    1,
                )

    def draw_blendshapes_info(self, image, blendshapes):
        """Display main expression coefficients on image"""
        height, width = image.shape[:2]

        # Select important expression coefficients to display
        important_categories = {
            "eyeBlinkLeft": "Left eye blink",
            "eyeBlinkRight": "Right eye blink",
            "mouthSmileLeft": "Left smile",
            "mouthSmileRight": "Right smile",
            "mouthOpen": "Open mouth",
            "jawOpen": "Jaw open",
            "browInnerUp": "Brow Inner Up",
            "browOuterUpLeft": "Left eyebrow up",
            "browOuterUpRight": "Right eyebrow up",
        }

        y_offset = 30
        cv2.putText(
            image,
            "Blendshapes:",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        y_offset += 25

        # Modified part: correctly handle blendshapes data structure
        if blendshapes and isinstance(blendshapes, list):
            for blendshape in blendshapes:
                category_name = (
                    blendshape.category_name
                    if hasattr(blendshape, "category_name")
                    else blendshape
                )
                score = blendshape.score if hasattr(blendshape, "score") else 0

                if category_name in important_categories:
                    cn_name = important_categories[category_name]
                    if score > 0.05:  # Only display meaningful values
                        text = f"{cn_name}: {score:.2f}"
                        # Display different colors based on value size
                        color = (
                            0,
                            int(255 * min(score * 2, 1)),
                            int(255 * (1 - min(score * 2, 1))),
                        )
                        cv2.putText(
                            image,
                            text,
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            1,
                        )
                        y_offset += 20

    def draw_transformation_matrix_info(self, image, detection_result):
        """Display facial transformation matrix information"""
        if not detection_result.facial_transformation_matrixes:
            return

        matrix = detection_result.facial_transformation_matrixes[0]
        height, width = image.shape[:2]

        # Display matrix information at bottom right
        start_x = width - 280
        start_y = height - 100

        cv2.putText(
            image,
            "Transform Matrix:",
            (start_x, start_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )

        # Display first few rows of matrix as reference
        for i in range(min(3, len(matrix))):
            row_text = f"[{matrix[i][0]:.2f}, {matrix[i][1]:.2f}, {matrix[i][2]:.2f}, {matrix[i][3]:.2f}]"
            cv2.putText(
                image,
                row_text,
                (start_x, start_y + 20 + i * 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1,
            )

    def process_frame(self, frame):
        """Process single frame image"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Asynchronous detection (non-blocking, returns immediately)
        self.landmarker.detect_async(mp_image, self.frame_timestamp_ms)
        self.frame_timestamp_ms += 1

    def display_loop(self, buffer: DoubleBuffer, stop_event: threading.Event):
        """Main display loop"""
        fps_history = []
        last_time = time.time()

        while not stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                print("[ERROR] Cannot read camera frame")
                break

            # Horizontal flip (mirror effect, like selfie)
            frame = cv2.flip(frame, 1)

            # Process current frame (asynchronously send to detector)
            self.process_frame(frame)

            # Get latest detection result and draw
            display_image = frame.copy()
            with self.lock:
                if self.detection_result and self.latest_image is not None:
                    # Draw landmarks on original frame
                    display_image = self.draw_landmarks_on_image(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), self.detection_result
                    )
                    display_image = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)

                    # Display Blendshapes information
                    if self.detection_result.face_blendshapes:
                        self.draw_blendshapes_info(
                            display_image, self.detection_result.face_blendshapes[0]
                        )

                    # Display transformation matrix information
                    self.draw_transformation_matrix_info(
                        display_image, self.detection_result
                    )

            # Calculate and display FPS
            current_time = time.time()
            fps = 1.0 / (current_time - last_time)
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)
            last_time = current_time

            cv2.putText(
                display_image,
                f"FPS: {avg_fps:.1f}",
                (10, 630),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # Display status information
            status = (
                "Face Detected"
                if (self.detection_result and self.detection_result.face_landmarks)
                else "No Face"
            )
            color = (0, 255, 0) if status == "Face Detected" else (0, 0, 255)
            cv2.putText(
                display_image, status, (10, 670), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
            )

            # Display number of key points
            if self.detection_result and self.detection_result.face_landmarks:
                num_points = len(self.detection_result.face_landmarks[0])
                cv2.putText(
                    display_image,
                    f"Points: {num_points}",
                    (10, 610),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    1,
                )

            # Display image
            # cv2.imshow(self.window_name, display_image)
            buffer.write({}, display_image)

            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # Q or ESC
                break
            elif key == ord("s"):  # S key to save screenshot
                screenshot_path = f"screenshot_{int(time.time())}.png"
                cv2.imwrite(screenshot_path, display_image)
                print(f"[INFO] Screenshot saved: {screenshot_path}")
            elif key == ord("f"):  # F key to toggle fullscreen
                is_fullscreen = cv2.getWindowProperty(
                    self.window_name, cv2.WND_PROP_FULLSCREEN
                )
                cv2.setWindowProperty(
                    self.window_name,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if is_fullscreen != 1 else cv2.WINDOW_NORMAL,
                )

    def enumerate_cameras(self, max_cameras=4):
        """
        Enumerate available cameras in the system
        :param max_cameras: Maximum number of cameras to check
        :return: List of available cameras [(index, name), ...]
        """
        available_cameras = []
        for i in range(max_cameras):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Try to get camera name
                backend_name = cap.getBackendName()
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                available_cameras.append(
                    (i, f"Camera {i} ({backend_name}, {width}x{height})")
                )
                cap.release()
        return available_cameras

    def select_camera(self):
        """
        Select camera to use
        :return: Selected camera index
        """
        available_cameras = self.enumerate_cameras()

        if not available_cameras:
            print("[ERROR] No camera detected")
            return -1

        # If only one camera, use it directly
        if len(available_cameras) == 1:
            print(f"[INFO] Only one camera detected: {available_cameras[0][1]}")
            return available_cameras[0][0]

        # Multiple cameras, show selection menu
        print("\nAvailable camera list:")
        for idx, (cam_idx, cam_name) in enumerate(available_cameras):
            print(f"  {idx}. {cam_name}")

        while True:
            try:
                choice = input("\nPlease select camera (enter number): ")
                choice_idx = int(choice)
                if 0 <= choice_idx < len(available_cameras):
                    selected = available_cameras[choice_idx]
                    print(f"[INFO] Selected: {selected[1]}")
                    return selected[0]
                else:
                    print("[ERROR] Invalid selection, please re-enter")
            except ValueError:
                print("[ERROR] Please enter a valid number")

    def run(self, buffer: DoubleBuffer, stop_event: threading.Event):
        """Main run function"""
        print("=" * 60)
        print("MediaPipe Face Landmarker Real Time")
        print("=" * 60)
        print("Controls:")
        print("  Q/ESC - Exit program")
        print("  S     - Save screenshot")
        print("  F     - Toggle fullscreen")
        print("=" * 60)

        # Create detector
        try:
            self.create_landmarker()
        except Exception as e:
            print(f"[ERROR] Failed to create Face Landmarker: {e}")
            return

        # Open camera
        camera_index = self.select_camera()
        if camera_index == -1:
            print("[ERROR] Cannot select camera")
            return

        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print(f"[ERROR] Cannot open camera {camera_index}")
            return

        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Print actual resolution
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[INFO] Camera resolution: {actual_width}x{actual_height}")

        print("[INFO] Camera started, press Q to exit")
        self.running = True

        try:
            self.display_loop(buffer, stop_event)
        except Exception as e:
            print(f"[ERROR] Runtime error: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        if self.landmarker:
            self.landmarker.close()
        print("[INFO] Program exited")


def main():
    """Program entry point"""
    # Run application
    app = FaceLandmarkerApp()
    app.run()


if __name__ == "__main__":
    main()
