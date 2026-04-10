import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import csv
import json

import matplotlib.pyplot as plt
import seaborn as sns

from utils.config import PITCH_THRESHOLD, ROLL_THRESHOLD

# Set unified style
plt.style.use("seaborn-v0_8-whitegrid")

# Uniformly define color scheme - only use two main colors to distinguish correct/incorrect
COLORS = {
    "correct": "#2ECC71",  # Emerald - correct prediction
    "error": "#E74C3C",  # Coral red - prediction error
    "threshold": "#34495E",  # Dark slate gray - threshold line
    "cmap": "Blues",  # Blue palette - confusion matrix
    "background": "#ECF0F1",  # Light gray background
}

CSV_FILE_PATH = "./results/evaluation_results.csv"
TYPE_JSON_PATH = "./src/config/type.json"
METRICS_JSON_PATH = "./results/metrics.json"
RESULT_DIR = "./results/"


def parse_type_flags(type_val):
    """Parse type integer, return whether it contains head_down and head_tilted"""
    t_val = int(type_val)
    return {
        "is_head_down": bool(t_val & 4),  # 0b100
        "is_head_tilted": bool(t_val & 8 or t_val & 16),  # 0b1000 or 0b10000
    }


def load_type_config(type_path):
    """Read type.json configuration file"""
    if not os.path.exists(type_path):
        raise FileNotFoundError(f"Error: Type config file {type_path} does not exist.")
    with open(type_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_metrics_config(metrics_path):
    """Read metrics.json evaluation metrics file"""
    if not os.path.exists(metrics_path):
        print(
            f"Warning: Metrics file {metrics_path} does not exist, skipping metrics display."
        )
        return None
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_data(csv_path):
    """Read CSV file data"""
    data = []
    if not os.path.exists(csv_path):
        print(f"Error: CSV file {csv_path} does not exist.")
        return data

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row["d_pitch"] = float(row["d_pitch"])
                row["d_roll"] = float(row["d_roll"])
                data.append(row)
            except ValueError:
                continue
    return data


def plot_metric_bars(metrics, ax, title):
    """Plot metrics bar chart"""
    if not metrics or not isinstance(metrics, dict):
        ax.text(
            0.5,
            0.5,
            "No metrics available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(title)
        return

    # Extract common metrics
    metric_names = ["precision", "recall", "f1_score", "accuracy", "f1"]
    available_metrics = {}

    for key in metric_names:
        if key in metrics:
            available_metrics[key] = metrics[key]

    if not available_metrics:
        # If standard metrics not found, display all numeric values
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and 0 <= value <= 1:
                available_metrics[key] = value

    if available_metrics:
        names = list(available_metrics.keys())
        values = list(available_metrics.values())
        bars = ax.bar(
            names,
            values,
            color=[COLORS["correct"] if v >= 0.8 else "#3498DB" for v in values],
            alpha=0.8,
        )
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.set_title(title, fontweight="bold")
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    else:
        ax.text(
            0.5,
            0.5,
            "No numeric metrics found",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(title)


def main(csv_file, type_json_file, metrics_json_file):
    # 1. Read configuration
    type_config = load_type_config(type_json_file)
    metrics_config = load_metrics_config(metrics_json_file)

    # 2. Read CSV data
    csv_data = read_csv_data(csv_file)
    if not csv_data:
        print("No CSV data read, program terminated.")
        return

    # Get metrics data (if exists)
    tn_metrics = metrics_config.get("head_down", {}) if metrics_config else {}
    ht_metrics = metrics_config.get("head_tilted", {}) if metrics_config else {}

    # --- Process d_pitch and head_down ---
    fig, ax_scatter = plt.subplots(figsize=(16, 6))

    pitch_x = []
    pitch_y = []
    pitch_colors = []
    pitch_markers = []

    # Confusion matrix counters: TN, FP, FN, TP
    tn_tn, tn_fp, tn_fn, tn_tp = 0, 0, 0, 0

    for idx, row in enumerate(csv_data):
        pitch_x.append(idx)
        pitch_y.append(row["d_pitch"])

        # Get predicted and actual values
        predicted = str(row.get("head_down", "False")).lower() == "true"
        try:
            type_val = int(row.get("type", 0))
            actual = bool(type_val & 4)
        except ValueError:
            actual = False

        # Record confusion matrix statistics
        if actual and predicted:
            tn_tp += 1
        elif actual and not predicted:
            tn_fn += 1
        elif not actual and predicted:
            tn_fp += 1
        else:
            tn_tn += 1

        # Simplified to two colors: correct (green) vs error (red)
        if predicted == actual:
            pitch_colors.append(COLORS["correct"])
        else:
            pitch_colors.append(COLORS["error"])

        # Shape distinguishes actual class: circle=positive (is head_down), triangle=negative (non head_down)
        if actual:
            pitch_markers.append("o")
        else:
            pitch_markers.append("^")

    # Group plot d_pitch
    pitch_groups = {
        ("o", COLORS["correct"]): [],
        ("o", COLORS["error"]): [],
        ("^", COLORS["correct"]): [],
        ("^", COLORS["error"]): [],
    }

    for x, y, m, c in zip(pitch_x, pitch_y, pitch_markers, pitch_colors):
        pitch_groups[(m, c)].append((x, y))

    for (marker, color), points in pitch_groups.items():
        if points:
            px, py = zip(*points)
            is_correct = color == COLORS["correct"]
            label_text = f"{'Head down' if marker == 'o' else 'Non Head down'} ({'Correct' if is_correct else 'Error'})"
            ax_scatter.scatter(
                px,
                py,
                c=color,
                marker=marker,
                s=80,
                alpha=0.7,
                edgecolors="black",
                linewidth=0.5,
                label=label_text,
            )

    ax_scatter.axhline(
        y=PITCH_THRESHOLD,
        color=COLORS["threshold"],
        linestyle="--",
        linewidth=2,
        label=f"Threshold ({PITCH_THRESHOLD})",
    )
    ax_scatter.set_title("d_pitch Analysis", fontsize=14, fontweight="bold")
    ax_scatter.set_xlabel("Sample Index", fontsize=12)
    ax_scatter.set_ylabel("d_pitch Value", fontsize=12)
    ax_scatter.legend(loc="best", frameon=True, fancybox=True)

    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}d_pitch_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"The chart has been saved to: {RESULT_DIR}d_pitch_analysis.png")

    # Plot head_down metrics separately
    fig, ax_metrics = plt.subplots(figsize=(8, 6))
    plot_metric_bars(tn_metrics, ax_metrics, "head_down Metrics")

    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}head_down_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"The chart has been saved to: {RESULT_DIR}head_down_metrics.png")

    # Plot head_down confusion matrix (separate figure)
    fig, ax = plt.subplots(figsize=(8, 6))
    tn_cm = [[tn_tn, tn_fp], [tn_fn, tn_tp]]
    total = tn_tn + tn_fp + tn_fn + tn_tp

    # Calculate percentage
    tn_cm_pct = [
        [tn_tn / total * 100, tn_fp / total * 100],
        [tn_fn / total * 100, tn_tp / total * 100],
    ]

    sns.heatmap(
        tn_cm,
        annot=True,
        fmt="d",
        cmap=COLORS["cmap"],
        xticklabels=["Predicted Negative", "Predicted Positive"],
        yticklabels=["Actual Negative", "Actual Positive"],
        cbar_kws={"label": "Count"},
        ax=ax,
        annot_kws={"size": 14, "weight": "bold"},
    )

    # Add percentage annotations
    for i in range(2):
        for j in range(2):
            ax.text(
                j + 0.5,
                i + 0.7,
                f"({tn_cm_pct[i][j]:.1f}%)",
                ha="center",
                va="center",
                color="white" if tn_cm[i][j] > total / 4 else "black",
                fontsize=10,
                alpha=0.8,
            )

    ax.set_title("head_down Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(
        f"{RESULT_DIR}head_down_confusion_matrix.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"The chart has been saved to: {RESULT_DIR}head_down_confusion_matrix.png")

    # --- Process d_roll and head_tilted ---
    fig, ax_scatter = plt.subplots(figsize=(16, 6))

    roll_x = []
    roll_y = []
    roll_colors = []
    roll_markers = []

    ht_tn, ht_fp, ht_fn, ht_tp = 0, 0, 0, 0

    for idx, row in enumerate(csv_data):
        roll_x.append(idx)
        roll_y.append(row["d_roll"])

        predicted = str(row.get("head_tilted", "False")).lower() == "true"
        try:
            type_val = int(row.get("type", 0))
            actual = bool(type_val & 8 or type_val & 16)
        except ValueError:
            actual = False

        if actual and predicted:
            ht_tp += 1
        elif actual and not predicted:
            ht_fn += 1
        elif not actual and predicted:
            ht_fp += 1
        else:
            ht_tn += 1

        # Simplified to two colors: correct (green) vs error (red)
        if predicted == actual:
            roll_colors.append(COLORS["correct"])
        else:
            roll_colors.append(COLORS["error"])

        if actual:
            roll_markers.append("o")
        else:
            roll_markers.append("^")

    # Group plot d_roll
    roll_groups = {
        ("o", COLORS["correct"]): [],
        ("o", COLORS["error"]): [],
        ("^", COLORS["correct"]): [],
        ("^", COLORS["error"]): [],
    }

    for x, y, m, c in zip(roll_x, roll_y, roll_markers, roll_colors):
        roll_groups[(m, c)].append((x, y))

    for (marker, color), points in roll_groups.items():
        if points:
            px, py = zip(*points)
            is_correct = color == COLORS["correct"]
            label_text = f"{'Head Tilted' if marker == 'o' else 'Non Head Tilted'} ({'Correct' if is_correct else 'Error'})"
            ax_scatter.scatter(
                px,
                py,
                c=color,
                marker=marker,
                s=80,
                alpha=0.7,
                edgecolors="black",
                linewidth=0.5,
                label=label_text,
            )

    ax_scatter.axhline(
        y=ROLL_THRESHOLD,
        color=COLORS["threshold"],
        linestyle="--",
        linewidth=2,
        label=f"Threshold (+{ROLL_THRESHOLD})",
    )
    ax_scatter.axhline(
        y=-ROLL_THRESHOLD,
        color=COLORS["threshold"],
        linestyle="--",
        linewidth=2,
        label=f"Threshold (-{ROLL_THRESHOLD})",
    )
    ax_scatter.set_title("d_roll Analysis", fontsize=14, fontweight="bold")
    ax_scatter.set_xlabel("Sample Index", fontsize=12)
    ax_scatter.set_ylabel("d_roll Value", fontsize=12)
    ax_scatter.legend(loc="best", frameon=True, fancybox=True)

    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}d_roll_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"The chart has been saved to: {RESULT_DIR}d_roll_analysis.png")

    # Plot head_tilted metrics separately
    fig, ax_metrics = plt.subplots(figsize=(8, 6))
    plot_metric_bars(ht_metrics, ax_metrics, "head_tilted Metrics")

    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}head_tilted_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"The chart has been saved to: {RESULT_DIR}head_tilted_metrics.png")

    # Plot head_tilted confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    ht_cm = [[ht_tn, ht_fp], [ht_fn, ht_tp]]
    total_ht = ht_tn + ht_fp + ht_fn + ht_tp

    sns.heatmap(
        ht_cm,
        annot=True,
        fmt="d",
        cmap=COLORS["cmap"],
        xticklabels=["Predicted Negative", "Predicted Positive"],
        yticklabels=["Actual Negative", "Actual Positive"],
        cbar_kws={"label": "Count"},
        ax=ax,
        annot_kws={"size": 14, "weight": "bold"},
    )

    # Add percentage
    for i in range(2):
        for j in range(2):
            pct = ht_cm[i][j] / total_ht * 100 if total_ht > 0 else 0
            ax.text(
                j + 0.5,
                i + 0.7,
                f"({pct:.1f}%)",
                ha="center",
                va="center",
                color="white" if ht_cm[i][j] > total_ht / 4 else "black",
                fontsize=10,
                alpha=0.8,
            )

    ax.set_title("head_tilted Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(
        f"{RESULT_DIR}head_tilted_confusion_matrix.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"The chart has been saved to: {RESULT_DIR}head_tilted_confusion_matrix.png")


if __name__ == "__main__":
    main(CSV_FILE_PATH, TYPE_JSON_PATH, METRICS_JSON_PATH)
