# Posture Health Reminder Device Based on Computer Vision

A real-time posture monitoring system that uses computer vision to detect unhealthy sitting postures and provide timely reminders.

## Features

- **Real-time Posture Detection**: Uses MediaPipe Face Landmarker to track head pose
- **Multi-threaded Architecture**: Efficient frame processing using double-buffer pattern
- **Interactive UI**: Tkinter-based interface with live camera feed and status dashboard
- **Evaluation System**: Complete pipeline for dataset labeling, preprocessing, and performance metrics
- **Configurable Thresholds**: Customizable pitch and roll angle thresholds for posture detection

## Prerequisites

- Python 3.8+ (3.10.11 for developer)
- FFmpeg (optional for video preprocessing)
- Camera device

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/XueTingkun/A-posture-health-reminder-device-based-on-computer-vision.git
    ```

2. Checkout the `dev-Xeler-ync` branch:
    ```bash
    git checkout dev-Xeler-ync
    ```

3. Download MediaPipe Face Mesh Model:
    ```
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
    ```
    Place the model file in the `models` directory.

4. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

5. Prepare dataset:
   - If you are a CDS540 course instructor or teaching assistant, you can find the Google Drive link in the report.
   - If you are a user, then simply run the command
       ```bash
       python ./src/main.py
       ```
       Select camera and you can see your face and landmarks in the window.
   - If you want to evaluate performance, please prepare your own dataset.
     - If your data are videos, place the videos under `./dataset/raw`, and then perform the following Dataset Preprocessing steps.
     - If your data are images, place the images under `./dataset/processed`, and then write `./dataset/dataset.csv` on your hand.

## Project Structure

```
├── dataset/              # Dataset storage
│   ├── raw/             # Raw video files
│   └── processed/       # Processed images
├── models/              # MediaPipe model files
├── src/
│   ├── config/          # Configuration files
│   ├── evaluation/      # Evaluation and analysis tools
│   ├── ui/              # User interface
│   └── utils/           # Utility modules
└── results/             # Evaluation results
```

## Usage

### Running the Application

```bash
python src/main.py
```

### Dataset Preprocessing

1. Strip audio from raw videos:
```bash
bash src/evaluation/strip_audio.sh
```

2. Process videos to extract frames:
```bash
python src/evaluation/preprocess_data.py
```

3. Label the dataset:
```bash
python src/evaluation/label_dataset.py
```

### Evaluation

1. Run evaluation:
```bash
python src/evaluation/evaluate.py
```

2. Analyze results:
```bash
python src/evaluation/analyze.py
```

3. Plot metrics:
```bash
python src/evaluation/plot.py
```

## Configuration

Posture detection thresholds can be configured in `src/config/config.py`:

```python
DEFAULT_CONFIG = {
    "PITCH_THRESHOLD": -10,  # Degrees
    "ROLL_THRESHOLD": 25,    # Degrees
}
```

## Posture Types

The system detects the following posture types (defined in `src/config/type.json`):

- Neutral pose (0)
- Left head turn (1)
- Right head turn (2)
- Head down (4)
- Left tilt (8)
- Right tilt (16)
- Head up (32)
- Normal (64)

## Architecture

The application uses a multi-threaded architecture:

1. **Face Detection Thread**: Captures frames and processes them using MediaPipe
2. **UI Thread**: Updates the Tkinter interface with processed frames
3. **Double Buffer**: Efficient frame exchange between threads

## Team

- Project Manager
- Lead Developer (Core Algorithms)
- Systems Developer (Environment & Integration)
- Analysis & Evaluation Specialist
- UI/UX & Design Specialist
- Documentation Specialist

## License

For me, Xeler-ync, the code in this repository is licensed under the GNU General Public License Version 3 (GPL3) open-source license.

## Acknowledgments

- MediaPipe for face landmark detection
- OpenCV for image processing
- Tkinter for user interface

