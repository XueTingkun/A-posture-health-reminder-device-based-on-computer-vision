import os
import base64
import csv
import hashlib

import cv2
import numpy as np

VIDEO_DIR = "./dataset/raw"
OUTPUT_DIR = "./dataset/processed"
CSV_FILE_PATH = "./dataset/dataset.csv"


def file_hash_for_filename(filepath):
    hasher = hashlib.blake2b(digest_size=8)

    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)

    return base64.urlsafe_b64encode(hasher.digest()).rstrip(b"=").decode("ascii")


def process_videos(directory, output_dir, csv_path):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Check if CSV exists to write header, otherwise append
    file_exists = os.path.isfile(csv_path)

    # List all mp4 files in the directory
    videos = [f for f in os.listdir(directory) if f.lower().endswith(".mp4")]

    if not videos:
        print(f"No MP4 files found in {directory}")
        return

    print(f"Found {len(videos)} videos. Starting playback...")
    print(
        "Press 'c' to Capture frame, 'q' to Quit current video, 'ESC' to Quit script."
    )

    group_id = -1
    for video_name in videos:
        group_id += 1
        video_path = os.path.join(directory, video_name)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Could not open video {video_name}")
            continue

        print(f"Playing: {video_name}")

        while True:
            ret, frame = cap.read()

            # If video finished or error reading frame
            if not ret:
                print(f"Finished or error reading {video_name}")
                break

            # Display the current frame
            cv2.imshow("Video Player", frame)

            # Wait for user input (delay 1ms)
            key = cv2.waitKey(50) & 0xFF

            # Capture frame on 'c' key
            if key == ord("c"):
                # Temporarily save frame to image, then compute file hash (avoid repeated encoding)
                temp_path = os.path.join(output_dir, "_temp.jpg")
                cv2.imwrite(temp_path, frame)

                # Compute hash from file (faster and more stable)
                frame_hash = file_hash_for_filename(temp_path)
                filename = f"{frame_hash}.jpg"
                final_path = os.path.join(output_dir, filename)

                # Rename to hash-based filename
                os.rename(temp_path, final_path)
                print(f"Captured: {filename}")

                # Write to CSV
                try:
                    with open(csv_path, mode="a", newline="", encoding="utf-8") as file:
                        writer = csv.writer(file)
                        # Write header if file is new
                        if not file_exists:
                            writer.writerow(["filename", "group_id", "type"])
                            file_exists = True
                        # Write data row
                        writer.writerow([filename, group_id, -1])
                except Exception as e:
                    print(f"Error writing to CSV: {e}")

            # Quit current video on 'q' key
            elif key == ord("q"):
                print(f"Skipping {video_name}")
                break

            # Quit entire script on 'ESC' key
            elif key == 27:
                print("Exiting script...")
                cap.release()
                cv2.destroyAllWindows()
                return

        cap.release()

    cv2.destroyAllWindows()
    print("All videos processed.")


if __name__ == "__main__":
    process_videos(VIDEO_DIR, OUTPUT_DIR, CSV_FILE_PATH)
