import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse

from utils.face import FaceLandmarkerApp


def process_images(image_paths, output_paths=None):
    if output_paths is None:
        output_paths = image_paths.copy()

    app = FaceLandmarkerApp(test=True)
    app.prepare_run_test()

    for img_path, out_path in zip(image_paths, output_paths):
        if not os.path.exists(img_path):
            print(f"[ERROR] Image not found: {img_path}")
            continue

        print(f"Processing: {img_path} -> {out_path}")
        app.draw_landmarks_only(img_path, out_path)

    print("All images processed.")


def get_user_input():
    print("Enter image paths (one per line), empty line to finish:")
    image_paths = []
    while True:
        path = input("> ").strip()
        if not path:
            break
        image_paths.append(path)
    return image_paths


def main():
    parser = argparse.ArgumentParser(description="Draw face landmarks on images")
    parser.add_argument(
        "images",
        nargs="*",
        help="Input image paths (if not provided, will be asked interactively)",
    )
    parser.add_argument(
        "-o",
        "--output",
        nargs="*",
        help="Output image paths (if not provided, will overwrite input images)",
    )

    args = parser.parse_args()

    if not args.images:
        image_paths = get_user_input()
        if not image_paths:
            print("[INFO] No images provided. Exiting.")
            return
    else:
        image_paths = args.images

    process_images(image_paths, args.output)


if __name__ == "__main__":
    main()
