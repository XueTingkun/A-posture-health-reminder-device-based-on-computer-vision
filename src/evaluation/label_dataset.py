import cv2
import os
import csv
import numpy as np


CSV_FILE_PATH = "./dataset/dataset.csv"
IMAGE_ROOT_DIR = "./dataset/processed/"
WINDOW_NAME = "Posture Annotation Tool"
INSTRUCTIONS = """
Instructions:
- Input numbers (e.g., 1, 32) or sums (e.g., 1+16) to update label.
- Input 'n' for Next image.
- Input 'p' for Previous image.
- Input 'q' to Quit.

Key Map (Values):
0: Neutral pose
1: Left head turn
2: Right head turn
4: Turtle neck
8: Left tilt
16: Right tilt
32: Head up
64: Normal
"""


class DatasetAnnotator:
    def __init__(self, csv_path, root_dir):
        self.csv_path = csv_path
        self.root_dir = root_dir
        self.fieldnames = ["filename", "group_id", "type"]
        self.data = []
        self.current_index = 0
        self.running = True  # Flag to control main loop

        # Read CSV data into memory
        self.load_csv()

        # Initialize window
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 800, 600)

        print(INSTRUCTIONS)
        self.load_current_image()

    def load_csv(self):
        """Read CSV file into self.data list"""
        if not os.path.exists(self.csv_path):
            print(f"Error: CSV file not found at {self.csv_path}")
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
            return

        try:
            with open(self.csv_path, mode="r", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                if reader.fieldnames != self.fieldnames:
                    print(
                        f"Warning: CSV headers {reader.fieldnames} do not match expected {self.fieldnames}"
                    )
                for row in reader:
                    row["type"] = int(row["type"])
                    self.data.append(row)
            print(f"Loaded {len(self.data)} rows from CSV.")
        except Exception as e:
            print(f"Error reading CSV: {e}")

    def update_csv(self):
        """Write self.data list back to CSV file"""
        try:
            with open(self.csv_path, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=self.fieldnames)
                writer.writeheader()
                writer.writerows(self.data)
            print(f"Saved CSV successfully ({len(self.data)} rows).")
        except Exception as e:
            print(f"Error saving CSV: {e}")

    def get_full_path(self, filename):
        return os.path.join(self.root_dir, filename)

    def parse_input(self, input_str):
        """Parse user input string, return integer bit flag value"""
        try:
            parts = input_str.replace(" ", "").split("+")
            total = 0
            for part in parts:
                if not part:
                    continue
                total += int(part)
            return total
        except ValueError:
            return None

    def load_current_image(self):
        if self.current_index < 0 or self.current_index >= len(self.data):
            print("End of dataset.")
            return

        row = self.data[self.current_index]
        filename = row["filename"]
        current_type = row["type"]

        path = self.get_full_path(filename)
        img = cv2.imread(path)

        if img is None:
            print(f"Error: Could not read image {path}")
            img = np.zeros((100, 200, 3), dtype=np.uint8)
            cv2.putText(
                img,
                "Image Not Found",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

        status_text = f"[{self.current_index + 1}/{len(self.data)}] File: {filename} | Current Type: {current_type}"
        cv2.setWindowTitle(WINDOW_NAME, status_text)
        cv2.imshow(WINDOW_NAME, img)

    def run(self):
        # Start a separate thread to keep OpenCV window responsive and prevent freezing
        # But since input() blocks the main thread, we use a simple polling approach here
        # In Python, input() and cv2.waitKey() are hard to coexist in the same thread
        # Therefore, the design here is: main thread handles input, cv2.waitKey only refreshes the window

        while self.running:
            # 1. Refresh OpenCV window (1ms delay)
            key = cv2.waitKey(1)
            # If user presses ESC (27) in window, can also exit
            if key == 27:
                print("Exiting via Window ESC...")
                break

            # 2. Display prompt in terminal and wait for input
            # Note: input() blocks here until user presses Enter
            # At this time OpenCV window may freeze due to long time without refresh (depends on system)
            # But this is the cost of pure command-line interaction
            try:
                user_input = input(
                    f"[{self.current_index + 1}/{len(self.data)}] Command (Value/n/p/q): "
                )
            except (EOFError, KeyboardInterrupt):
                # Handle Ctrl+C or Ctrl+D
                print("\nExiting...")
                break

            user_input = user_input.strip().lower()

            # 3. Process commands
            if user_input == "q":
                print("Exiting...")
                self.running = False

            elif user_input == "n":  # Next
                if self.current_index < len(self.data) - 1:
                    self.current_index += 1
                    self.load_current_image()
                else:
                    print("Already at last image.")

            elif user_input == "p":  # Previous
                if self.current_index > 0:
                    self.current_index -= 1
                    self.load_current_image()
                else:
                    print("Already at first image.")

            else:
                # Try to parse as numeric label
                new_type = self.parse_input(user_input)
                if new_type is not None:
                    # Update data
                    self.data[self.current_index]["type"] = new_type
                    print(f"Updated Type to: {new_type}")
                    self.update_csv()

                    # Auto jump to next image
                    if self.current_index < len(self.data) - 1:
                        self.current_index += 1
                        self.load_current_image()
                else:
                    print("Invalid command or number format.")

        cv2.destroyAllWindows()


if __name__ == "__main__":
    if not os.path.exists(CSV_FILE_PATH):
        print(f"Error: CSV file not found at {CSV_FILE_PATH}")
    else:
        annotator = DatasetAnnotator(CSV_FILE_PATH, IMAGE_ROOT_DIR)
        annotator.run()
