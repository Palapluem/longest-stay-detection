# Connected Tech Internship 2026 Take-Home Test: Longest Stay Detection

<div align="center">
  <img height="96" alt="Connected Tech logo" src="https://www.connectedtech.co.th/assets/connected-logo-lg-Cto5_QN-.png" />
</div>

Coding test solution for identifying the tracked person who stayed stationary for the longest continuous duration in `entrance.mov`.

The final answer is a tracker identity (`track_id`), not a real-world person identity.

## Project Overview

This repository provides an end-to-end computer vision pipeline for:

- **Part 1: Problem and data understanding** - define the task, video metadata, and stationary meaning.
- **Part 2: Method design** - explain YOLO detection, tracking, foot-point movement, smoothing, IoU support, and hysteresis.
- **Part 3: Video pipeline** - runnable Python implementation that generates all required outputs.
- **Part 4: Evaluation and results** - summarize the detected winner, limitations, and recommended improvements.

The repository includes both:

- `main.py` for command-line execution.
- `0_Longest_Stay_Detection_Report.ipynb` for a full runnable Thai walkthrough with code cells and executed outputs.

## Final Result

The generated run in this workspace used `yolo11n.pt` with `imgsz=640` for speed.

```text
winner track_id: 161
source_track_ids: [161]
longest stationary duration: 41.094 seconds
stationary segment: 33.699s - 74.793s
```

Video metadata was read from OpenCV at runtime:

```text
resolution: 1920x1080
fps: 29.883
frame_count: 2556
duration_sec: 85.535
```

## Project Structure

```text
longest-stay-detection/
├── 0_Longest_Stay_Detection_Report.ipynb   # Full runnable Thai walkthrough notebook
├── 1_Problem_and_Data/
│   └── README.md                           # Problem, input video, stationary definition
├── 2_Method_Design/
│   └── README.md                           # Algorithm design and movement measurement
├── 3_Video_Pipeline/
│   ├── README.md                           # Pipeline-specific instructions
│   └── src/
│       └── longest_stay_detection.py       # Main implementation
├── 4_Evaluation_and_Results/
│   └── README.md                           # Result, assessment, and improvement plan
├── results/
│   ├── annotated_entrance.mp4              # Annotated output video
│   ├── summary.json                        # Structured final result
│   └── tracks.csv                          # Per-track stationary segment table
├── entrance.mov                            # Input video
├── IDEA.md                                 # Short idea description
├── longest-stay-detection-info.txt         # Original coding-test brief
├── main.py                                 # CLI entrypoint
├── references.md                           # Research and documentation references
├── requirements.txt                        # Python dependencies
└── README.md                               # This file
```

## Submission Notes

Recommended files to keep in the submitted repository:

- `0_Longest_Stay_Detection_Report.ipynb`
- `1_Problem_and_Data/`
- `2_Method_Design/`
- `3_Video_Pipeline/`
- `4_Evaluation_and_Results/`
- `main.py`
- `requirements.txt`
- `README.md`
- `IDEA.md`
- `references.md`
- `longest-stay-detection-info.txt`
- `results/summary.json`
- `results/tracks.csv`
- `results/annotated_entrance.mp4` through Git LFS, or an external Drive/YouTube link to this video

Local/reference files that should not be part of the final submission:

- `fraud-transaction-detection-main/` - reference project only
- old unrelated notebooks such as `5-domains-hackathon-house-recognition.ipynb`
- `__pycache__/`
- `*.pt` YOLO weight files downloaded at runtime
- temporary logs

Note: `results/annotated_entrance.mp4` is a large generated artifact and is tracked with Git LFS.

## Getting Started

### Prerequisites

- Python 3.11 or 3.12
- pip
- Enough disk space for YOLO weights and annotated video output
- CPU is supported; GPU is faster but not required

### Step 1: Create a Virtual Environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2: Install Dependencies

```bash
python -m pip install -r requirements.txt
```

The first run may download YOLO weights such as `yolo11s.pt` or `yolo11n.pt`.

## How to Run

### Option A: Command-Line Pipeline

Recommended quality run:

```bash
python main.py --video entrance.mov --output results/annotated_entrance.mp4 --model yolo11s.pt --tracker botsort.yaml --conf 0.25 --imgsz 960 --stationary-threshold 0.05 --window-sec 1.0 --min-stationary-sec 0.75
```

Faster run:

```bash
python main.py --video entrance.mov --output results/annotated_entrance.mp4 --model yolo11n.pt --imgsz 640
```

Try ByteTrack:

```bash
python main.py --video entrance.mov --tracker bytetrack.yaml
```

### Option B: Notebook Walkthrough

Open and run:

```text
0_Longest_Stay_Detection_Report.ipynb
```

The notebook contains the complete pipeline code split into readable sections and has already been executed with outputs. It is useful for reviewing how each step works.

## Workflow

### 1. Read Video Metadata

The pipeline opens `entrance.mov` with OpenCV and reads:

- width
- height
- FPS
- frame count
- duration

This satisfies the rule that FPS and video length must not be hard-coded.

### 2. Detect People with YOLO

The detector uses Ultralytics YOLO and filters only person detections:

```python
classes=[0]
```

The default command uses `yolo11s.pt` for better quality. The faster tested run used `yolo11n.pt`.

### 3. Track People Across Frames

The pipeline uses:

```python
model.track(..., persist=True, tracker="botsort.yaml")
```

BoT-SORT is the default tracker because the video contains multiple people and possible occlusion. ByteTrack is available through the CLI as an alternative.

### 4. Measure Stationary Behavior

For every tracked person, the pipeline stores the bbox foot point:

```text
foot_point = (center_x, y2)
```

The foot point is used because it better approximates the person's floor position than the bbox center.

Movement is measured with:

```text
norm_disp = distance(current_smoothed_foot, old_smoothed_foot) / median_bbox_height
```

The foot point is smoothed with EMA to reduce detector jitter. Movement is normalized by bbox height so the same threshold works better for people near and far from the camera.

### 5. Build Stationary Segments

The system uses a sliding time window and hysteresis:

- require stationary evidence for at least `--min-stationary-sec`
- tolerate short detection jitter before ending a segment
- tolerate short missed detections before splitting a segment

Each track gets:

- total visible time
- stationary segments
- longest stationary segment

### 6. Select Winner

The winner is the track with the longest stationary segment duration.

## Outputs

### `results/summary.json`

Contains:

- video metadata
- method settings
- winner track id
- winner start/end/duration
- all per-track stationary segments

### `results/tracks.csv`

Flat table of each track and stationary segment.

### `results/annotated_entrance.mp4`

Annotated video with:

- bounding boxes
- track ids
- moving/stationary status
- stationary duration
- winner highlight


## Method Rationale

Frame differencing or background subtraction alone is not used as the main method because:

- a person standing still for a long time can be absorbed into a background model
- frame differencing does not preserve identity across frames
- lighting, shadows, and small body motion can create false motion

This task needs persistent per-person identity and position stability, so detector + tracker is a better backbone.

## Tuning Guide

- Increase `--imgsz` to `960` or `1280` if small or far people are missed.
- Lower `--conf` if detections are fragmented, but expect more false positives.
- Increase `--stationary-threshold` if stationary people are marked as moving due to detector jitter.
- Decrease `--stationary-threshold` if slow walking is counted as stationary.
- Increase `--min-stationary-sec` to require stronger evidence before a segment starts.
- Try `--tracker bytetrack.yaml` if BoT-SORT creates too many broken tracks.

## Limitations

- ID switches can happen when people overlap or occlude each other.
- Detector jitter can affect bbox and foot-point position.
- Perspective makes pixel movement different for people near and far from the camera.
- A person may stand in place while moving arms or head; this method focuses on scene position rather than full-body stillness.
- Camera shake would require stabilization or global motion compensation.

## Recommended Improvements

High-value next steps:

- Run a comparison using `yolo11s.pt` with `imgsz=960`.
- Compare `botsort.yaml` and `bytetrack.yaml`.
- Add a short visual validation table for timestamps around the winning segment.
- Export a debug CSV for the winner track with `norm_disp`, IoU, and status per frame.

## References

See `references.md`.
