# Part 3: Video Processing Pipeline

This folder contains the runnable computer-vision pipeline for the longest-stay detection task.

## Entry Point

Run from the repository root:

```bash
python main.py --video entrance.mov --output results/annotated_entrance.mp4
```

The root `main.py` is a small wrapper around:

```text
3_Video_Pipeline/src/longest_stay_detection.py
```

## What It Does

- Reads FPS, frame count, width, height, and duration from OpenCV video metadata.
- Runs Ultralytics YOLO person detection with `classes=[0]`.
- Tracks people with BoT-SORT by default, with `bytetrack.yaml` available through CLI.
- Measures stationary behavior from EMA-smoothed foot-point displacement normalized by bbox height.
- Uses bbox IoU and hysteresis to reduce jitter-driven segment breaks.
- Saves annotated video, JSON summary, and CSV segment table.

## Important CLI Arguments

```bash
python main.py \
  --video entrance.mov \
  --output results/annotated_entrance.mp4 \
  --model yolo11s.pt \
  --tracker botsort.yaml \
  --conf 0.25 \
  --imgsz 960 \
  --stationary-threshold 0.05 \
  --window-sec 1.0 \
  --min-stationary-sec 0.75
```
