import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import csv
from collections import defaultdict

from utils.face import FaceLandmarkerApp


INPUT_CSV_PATH = "./dataset/dataset.csv"
IMAGE_ROOT_DIR = "./dataset/processed/"
OUTPUT_CSV_PATH = "./results/evaluation_results.csv"
FIELDNAMES = [
    "filename",
    "group_id",
    "type",
    "pitch",
    "roll",
    "yaw",
    "d_pitch",
    "d_roll",
    "d_yaw",
    "head_down",
    "head_tilted",
]
os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)


class FaceEvaluator:
    def __init__(self, input_csv, output_csv):
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.fieldnames = FIELDNAMES
        self.data = []
        self.results = []

        # Create FaceLandmarkerApp instance
        self.face_app = FaceLandmarkerApp(test=True)
        self.face_app.prepare_run_test()

        # Check if input file exists
        if not os.path.exists(self.input_csv):
            print(f"Error: Input CSV file not found at {self.input_csv}")
            raise FileNotFoundError(f"Input file not found: {self.input_csv}")

        # Load test data
        self.load_csv()

    def load_csv(self):
        """Read CSV file into self.data list"""
        try:
            with open(self.input_csv, mode="r", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                if reader.fieldnames != ["filename", "group_id", "type"]:
                    print(
                        f"Warning: CSV headers {reader.fieldnames} do not match expected"
                    )
                for row in reader:
                    row["group_id"] = int(row["group_id"])
                    row["type"] = int(row["type"])
                    self.data.append(row)
            print(f"Loaded {len(self.data)} rows from input CSV.")
        except Exception as e:
            print(f"Error reading CSV: {e}")
            raise

    def save_results(self):
        """Write results to output CSV file"""
        try:
            with open(self.output_csv, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=self.fieldnames)
                writer.writeheader()
                writer.writerows(self.results)
            print(f"Saved {len(self.results)} results to {self.output_csv}")
        except Exception as e:
            print(f"Error saving results: {e}")
            raise

    def process_groups(self):
        """Process all grouped data"""
        # Group by group_id
        groups = defaultdict(list)
        for item in self.data:
            groups[item["group_id"]].append(item)

        # Process each group
        for group_id in sorted(groups.keys()):
            self.process_group(group_id, groups[group_id])

    def process_group(self, group_id, items):
        """Process data for a single group"""
        print(f"\nProcessing group {group_id}...")

        # Find calibration file with type=0
        calibrate_file = None
        test_files = []

        for item in items:
            if item["type"] == 0:
                calibrate_file = item["filename"]
            else:
                test_files.append(item)

        if not calibrate_file:
            print(f"Warning: No calibration file (type=0) found for group {group_id}")
            return

        # Run calibration first
        print(f"  Calibration file: {calibrate_file}")
        calibrate_result = self.face_app.run_test(
            os.path.join(IMAGE_ROOT_DIR, calibrate_file), is_calibration=True
        )
        print(f"  Calibration result: {calibrate_result}")

        # Run test files
        for test_file in test_files:
            print(f"  Testing file: {test_file['filename']}")
            try:
                result = self.face_app.run_test(
                    os.path.join(IMAGE_ROOT_DIR, test_file["filename"])
                )
                print(f"    Result: {result}")

                # Record results
                self.results.append(
                    {
                        "filename": test_file["filename"],
                        "group_id": group_id,
                        "type": test_file["type"],
                        "pitch": result["pitch"],
                        "roll": result["roll"],
                        "yaw": result["yaw"],
                        "d_pitch": result["d_pitch"],
                        "d_roll": result["d_roll"],
                        "head_down": result["head_down"],
                        "head_tilted": result["head_tilted"],
                    }
                )
            except Exception as e:
                print(f"    Error processing {test_file['filename']}: {e}")
                # Record error results
                self.results.append(
                    {
                        "filename": test_file["filename"],
                        "group_id": group_id,
                        "type": test_file["type"],
                        "result": f"Error: {str(e)}",
                    }
                )

    def evaluate(self):
        """Execute complete evaluation workflow"""
        print("Starting evaluation...")
        self.process_groups()
        self.save_results()
        print("\nEvaluation completed.")


if __name__ == "__main__":
    try:
        evaluator = FaceEvaluator(INPUT_CSV_PATH, OUTPUT_CSV_PATH)
        evaluator.evaluate()
    except Exception as e:
        print(f"Evaluation failed: {e}")
        exit(1)
