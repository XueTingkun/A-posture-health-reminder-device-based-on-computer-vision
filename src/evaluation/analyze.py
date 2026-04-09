import csv
import json
from typing import Dict

TYPE_BITS = {
    0: "Neutral pose",
    1: "Left head turn",
    2: "Right head turn",
    4: "Turtle neck",
    8: "Left tilt",
    16: "Right tilt",
    32: "Head up",
    64: "Normal",
}

INPUT_CSV = "./results/evaluation_results.csv"
OUTPUT_JSON = "./results/metrics.json"


def parse_type(type_value: int) -> Dict[str, bool]:
    """Parse type bit enum into individual pose flags"""
    return {
        "neutral": type_value == 0,
        "left_turn": bool(type_value & 1),
        "right_turn": bool(type_value & 2),
        "turtle_neck": bool(type_value & 4),
        "left_tilt": bool(type_value & 8),
        "right_tilt": bool(type_value & 16),
        "head_up": bool(type_value & 32),
        "normal": bool(type_value & 64),
    }


def calculate_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Calculate evaluation metrics"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def evaluate_results(csv_path: str, output_json_path: str):
    """Evaluate model results and save metrics"""
    turtle_neck_metrics = {"tp": 0, "fp": 0, "fn": 0}
    head_tilted_metrics = {"tp": 0, "fp": 0, "fn": 0}

    with open(csv_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            type_flags = parse_type(int(row["type"]))

            predicted_turtle_neck = row["turtle_neck"].lower() == "true"
            predicted_head_tilted = row["head_tilted"].lower() == "true"

            true_turtle_neck = type_flags["turtle_neck"]
            true_head_tilted = type_flags["left_tilt"] or type_flags["right_tilt"]

            if true_turtle_neck and predicted_turtle_neck:
                turtle_neck_metrics["tp"] += 1
            elif not true_turtle_neck and predicted_turtle_neck:
                turtle_neck_metrics["fp"] += 1
            elif true_turtle_neck and not predicted_turtle_neck:
                turtle_neck_metrics["fn"] += 1

            if true_head_tilted and predicted_head_tilted:
                head_tilted_metrics["tp"] += 1
            elif not true_head_tilted and predicted_head_tilted:
                head_tilted_metrics["fp"] += 1
            elif true_head_tilted and not predicted_head_tilted:
                head_tilted_metrics["fn"] += 1

    turtle_neck_results = calculate_metrics(
        turtle_neck_metrics["tp"], turtle_neck_metrics["fp"], turtle_neck_metrics["fn"]
    )

    head_tilted_results = calculate_metrics(
        head_tilted_metrics["tp"], head_tilted_metrics["fp"], head_tilted_metrics["fn"]
    )

    results = {"turtle_neck": turtle_neck_results, "head_tilted": head_tilted_results}

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_json_path}")
    return results


if __name__ == "__main__":
    evaluate_results(INPUT_CSV, OUTPUT_JSON)
