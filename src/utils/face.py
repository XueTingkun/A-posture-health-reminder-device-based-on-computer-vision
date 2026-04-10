import json
import threading
import time
from queue import Queue, Empty

import cv2
from cv2_enumerate_cameras import enumerate_cameras as cv2_enum_cameras
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

from utils.DoubleBuffer import DoubleBuffer
from utils.config import PITCH_THRESHOLD, ROLL_THRESHOLD


class FaceLandmarkerApp:
    def __init__(self, test=False):
        self.test = test
        # Initialize parameters
        self.cap = None
        self.landmarker = None
        self.running = False
        self.frame_timestamp_ms = 0
        self.camera_arr = []
        self.current_camera_id = -1

        self.baseline_pitch = 0.0
        self.baseline_roll = 0.0

        # Store detection results
        self.detection_result = None
        self.latest_image = None
        self.lock = threading.Lock()

        # face configs
        with open("./src/config/mp_face.json") as f:
            self.face_configs = json.load(f)

        # Model path
        self.model_path = "./models/face_landmarker.task"

        # Drawing color configuration
        self.MESH_COLOR = (0, 255, 0)  # Green mesh
        self.CONTOUR_COLOR = (255, 255, 255)  # White contour
        self.IRIS_COLOR = (0, 255, 255)  # Cyan iris
        self.POINT_COLOR = (0, 0, 255)  # Red key points

        self.model_points = np.array(
            [
                [0.0, 0.0, 0.0],  # Nose tip (Landmark 1)
                [0.0, -330.0, -65.0],  # Chin (Landmark 199)
                [-225.0, 170.0, -135.0],  # Left eye corner (Landmark 33)
                [225.0, 170.0, -135.0],  # Right eye corner (Landmark 263)
                [-150.0, -150.0, -125.0],  # Left mouth corner (Landmark 61)
                [150.0, -150.0, -125.0],  # Right mouth corner (Landmark 291)
            ],
            dtype=np.float64,
        )
        # Corresponding 2D key point indices (MediaPipe 478-point model)
        self.image_points_idx = [1, 199, 33, 263, 61, 291]

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

    def get_camera_matrix(self, frame_w, frame_h):
        """Construct approximate camera intrinsic matrix"""
        focal_length = frame_w
        center = (frame_w / 2, frame_h / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype="double",
        )
        return camera_matrix

    def get_head_pose(self, landmarks, frame_w, frame_h):
        """
        Calculate head rotation Euler angles via solvePnP
        Args:
            landmarks: NormalizedLandmarkList returned by MediaPipe (each element has x, y, z)
            frame_w, frame_h: Image width and height
        Returns:
            (roll, pitch, yaw) in degrees; returns (None, None, None) on failure
        """
        # Extract 2D image coordinates (pixels)
        image_points = []
        for idx in self.image_points_idx:
            # Check index validity (MediaPipe Face Landmarker returns 478 points)
            if idx >= len(landmarks):
                return None, None, None
            x = landmarks[idx].x * frame_w
            y = landmarks[idx].y * frame_h
            image_points.append([x, y])
        image_points = np.array(image_points, dtype=np.float64)

        # Camera parameters
        camera_matrix = self.get_camera_matrix(frame_w, frame_h)
        dist_coeffs = np.zeros((4, 1))

        # Solve for rotation vector and translation vector
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None, None, None

        # Rotation vector -> Rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # Decompose Euler angles from rotation matrix (cv2.RQDecomp3x3 returns: rotation matrix, Euler angles(degrees), ...)
        # Output order: pitch, yaw, roll
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
        pitch, yaw, roll = angles[0], angles[1], angles[2]

        if pitch > 90:
            pitch = -pitch + 180
        elif pitch < -90:
            pitch = -pitch - 180
        if roll > 90:
            roll = roll - 180
        elif roll < -90:
            roll = roll + 180

        return roll, pitch, yaw

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
        for connection in self.face_configs["mesh_connections"]:
            start_idx, end_idx = connection
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(image, points[start_idx], points[end_idx], self.MESH_COLOR, 1)

    def draw_contours(self, image, points):
        """Draw facial contours"""
        # Face outer contour
        for i in range(len(self.face_configs["face_outline"]) - 1):
            idx1, idx2 = (
                self.face_configs["face_outline"][i],
                self.face_configs["face_outline"][i + 1],
            )
            if idx1 < len(points) and idx2 < len(points):
                cv2.line(image, points[idx1], points[idx2], self.CONTOUR_COLOR, 2)

        # Left eyebrow
        for i in range(len(self.face_configs["left_brow"]) - 1):
            idx1, idx2 = (
                self.face_configs["left_brow"][i],
                self.face_configs["left_brow"][i + 1],
            )
            if idx1 < len(points) and idx2 < len(points):
                cv2.line(image, points[idx1], points[idx2], self.CONTOUR_COLOR, 2)

        # Right eyebrow
        for i in range(len(self.face_configs["right_brow"]) - 1):
            idx1, idx2 = (
                self.face_configs["right_brow"][i],
                self.face_configs["right_brow"][i + 1],
            )
            if idx1 < len(points) and idx2 < len(points):
                cv2.line(image, points[idx1], points[idx2], self.CONTOUR_COLOR, 2)

        # Nose
        for i in range(len(self.face_configs["nose"]) - 1):
            idx1, idx2 = self.face_configs["nose"][i], self.face_configs["nose"][i + 1]
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
        for idx_str, label in self.face_configs["key_indices"].items():
            idx = int(idx_str)
            if idx < len(points):
                x, y = points[idx]
                cv2.circle(image, (x, y), 3, (0, 255, 255), -1)
                cv2.putText(
                    image,
                    str(idx_str),
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

                if category_name in self.face_configs["important_categories"]:
                    cn_name = self.face_configs["important_categories"][category_name]
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

    def display_loop(
        self,
        db_f2t: DoubleBuffer,
        q_f2t: Queue,
        q_t2f: Queue,
        stop_event: threading.Event,
    ):
        """Main display loop"""
        fps_history = []
        last_time = time.time()

        while not stop_event.is_set():
            try:
                d = q_t2f.get_nowait()
                match d["cmd"]:
                    case "change_camera":
                        if self.current_camera_id == d["camera"]:
                            continue
                        if self.cap:
                            self.cap.release()
                        self.current_camera_id = d["camera"]
                        self.create_cap()
                        try:
                            self.create_landmarker()
                        except Exception as e:
                            print(f"[ERROR] Failed to create Face Landmarker: {e}")
                    case "calibrate_posture":
                        if pitch is not None and roll is not None:
                            self.baseline_pitch = pitch
                            self.baseline_roll = roll
                            print(
                                f"[INFO] Posture calibrated: Pitch={self.baseline_pitch:.2f}, Roll={self.baseline_roll:.2f}"
                            )
            except Empty:
                pass
            if self.cap == None:
                db_f2t.write({"status_text": "Waiting Camera"}, [])
                time.sleep(0.1)
                continue
            if self.landmarker == None:
                db_f2t.write({"status_text": "Waiting Landmarker"}, [])
                time.sleep(0.1)
                continue

            ret, frame = self.cap.read()
            if not ret:
                print("[ERROR] Cannot read camera frame")
                db_f2t.write({"status_text": "Error"}, [])
                break

            # Horizontal flip (mirror effect, like selfie)
            frame = cv2.flip(frame, 1)

            # Process current frame (asynchronously send to detector)
            self.process_frame(frame)

            roll = None
            pitch = None
            yaw = None
            d_pitch = None
            d_roll = None
            status_text = "Normal"

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

                    # ------------------------------
                    # Calculate and display head pose angles (Pitch, Roll) and corresponding warnings
                    # ------------------------------
                    if self.detection_result.face_landmarks:
                        face_landmarks = self.detection_result.face_landmarks[0]
                        frame_h, frame_w = display_image.shape[:2]
                        roll, pitch, yaw = self.get_head_pose(
                            face_landmarks, frame_w, frame_h
                        )
                        if pitch is not None:
                            d_pitch = pitch - self.baseline_pitch
                            d_roll = roll - self.baseline_roll

                            # Head down warning
                            if d_pitch < PITCH_THRESHOLD:
                                cv2.putText(
                                    display_image,
                                    f"WARNING: Head down! d({int(d_pitch)} deg)",
                                    (10, 430),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (0, 0, 255),
                                    2,
                                )
                                status_text = "Warning"

                            # Head tilt warning
                            if abs(d_roll) > ROLL_THRESHOLD:
                                cv2.putText(
                                    display_image,
                                    f"WARNING: Head Tilted! d({int(d_roll)} deg)",
                                    (10, 470),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (0, 0, 255),
                                    2,
                                )
                                status_text = "Warning"

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
                display_image,
                status,
                (10, 670),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
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

            db_f2t.write(
                {
                    "metrics": {
                        "pitch": pitch or 0.0,
                        "roll": roll or 0.0,
                        "yaw": yaw or 0.0,
                        "fps": fps or 0.0,
                        "d_pitch": d_pitch or 0.0,
                        "d_roll": d_roll or 0.0,
                    },
                    "status_text": status_text,
                },
                display_image,
            )

    def enumerate_cameras(self, max_cameras=4):
        """
        Enumerate available cameras in the system using cv2-enumerate-cameras.
        Note: max_cameras parameter is kept for API compatibility but is ignored
              because the library enumerates all connected cameras.
        :return: List of available cameras [(index, name), ...]
        """
        cameras = cv2_enum_cameras()

        for cam in cameras:
            cap = cv2.VideoCapture(cam.index, cam.backend)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                backend_name = cap.getBackendName()
                if cam.name:
                    display_name = f"{cam.name} (Index {cam.index}, {backend_name}, {width}x{height})"
                else:
                    display_name = (
                        f"Camera {cam.index} ({backend_name}, {width}x{height})"
                    )
                self.camera_arr.append((cam.index, display_name))
                cap.release()

        return self.camera_arr

    def push_camera_choice(self, q_f2t: Queue, q_t2f: Queue):
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

        q_f2t.put({"cmd": "camera_selection", "data": available_cameras})

    def run(
        self,
        db_f2t: DoubleBuffer,
        q_f2t: Queue,
        q_t2f: Queue,
        stop_event: threading.Event,
    ):
        if self.test:
            raise RuntimeError("run_test should be called instead of run in test mode")

        # Open camera
        self.push_camera_choice(q_f2t, q_t2f)

        self.running = True

        try:
            self.display_loop(db_f2t, q_f2t, q_t2f, stop_event)
        except Exception as e:
            print(f"[ERROR] Runtime error: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self.cleanup()

    def create_cap(self):
        if self.test:
            return

        index = -1
        for i, (idx, desc) in enumerate(self.camera_arr):
            if f"{idx} " + "{" + desc + "}" == self.current_camera_id:
                index = i
                break

        if index == -1:
            print("[ERROR] Cannot select camera")
            return

        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            print(f"[ERROR] Cannot open camera {index}")
            return

        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Print actual resolution
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[INFO] Camera resolution: {actual_width}x{actual_height}")

        print("[INFO] Camera started, press Q to exit")

    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.cap:
            self.cap.release()
        if self.landmarker:
            self.landmarker.close()
        print("[INFO] Program exited")

    def prepare_run_test(self):
        if not self.test:
            raise RuntimeError("prepare_run_test can only be called in test mode")

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,  # Image mode
            num_faces=1,  # Maximum number of faces to detect
            min_face_detection_confidence=0.5,  # Face detection confidence threshold
            min_face_presence_confidence=0.5,  # Face presence confidence threshold
            min_tracking_confidence=0.5,  # Tracking confidence threshold
            output_face_blendshapes=True,  # Output expression blendshapes
            output_facial_transformation_matrixes=True,  # Output facial transformation matrix
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        # self.cap =

    def run_test(self, image_path, is_calibration=False):
        if not self.test:
            print("[ERROR] run_test can only be called in test mode")
            return None

        frame = cv2.imread(image_path)
        if frame is None:
            print(f"[ERROR] Cannot read image from {image_path}")
            return None

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Synchronous detection for IMAGE mode
        detection_result = self.landmarker.detect(mp_image)

        if not detection_result or not detection_result.face_landmarks:
            print("[WARNING] No face detected in the image")
            return None

        face_landmarks = detection_result.face_landmarks[0]
        frame_h, frame_w = frame.shape[:2]
        roll, pitch, yaw = self.get_head_pose(face_landmarks, frame_w, frame_h)

        if is_calibration:
            if pitch is not None and roll is not None:
                self.baseline_pitch = pitch
                self.baseline_roll = roll
                print(
                    f"[INFO] Posture calibrated: Pitch={self.baseline_pitch:.2f}, Roll={self.baseline_roll:.2f}"
                )
                return {
                    "pitch": pitch,
                    "roll": roll,
                    "yaw": yaw,
                    "d_pitch": 0,
                    "d_roll": 0,
                    "head_down": False,
                    "head_tilted": False,
                }
        else:
            if pitch is not None:
                d_pitch = pitch - self.baseline_pitch
                d_roll = roll - self.baseline_roll
                return {
                    "pitch": pitch,
                    "roll": roll,
                    "yaw": yaw,
                    "d_pitch": d_pitch,
                    "d_roll": d_roll,
                    "head_down": d_pitch < PITCH_THRESHOLD,
                    "head_tilted": abs(d_roll) > ROLL_THRESHOLD,
                }

    def draw_landmarks_only(self, image_path, output_path=None):
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"[ERROR] Cannot read image from {image_path}")
            return None

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.landmarker.detect(mp_image)

        if not detection_result or not detection_result.face_landmarks:
            print("[WARNING] No face detected in the image")
            return None

        display_image = self.draw_landmarks_on_image(rgb_frame, detection_result)
        display_image = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)

        if output_path:
            cv2.imwrite(output_path, display_image)
            print(f"[INFO] Image saved to {output_path}")

        return display_image


def main():
    """Program entry point"""
    # Run application
    app = FaceLandmarkerApp()
    app.run()


if __name__ == "__main__":
    main()
