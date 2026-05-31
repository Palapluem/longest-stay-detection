# Part 1: Problem and Data Understanding

## Objective

Identify the tracked person who remained stationary for the longest continuous duration in `entrance.mov`.

The answer must include:

- winner track id
- longest stationary duration in seconds
- start and end time of that stationary segment
- method used to measure stationary time

## Dataset

- Local input video: `entrance.mov` (intentionally not distributed in this public repository)
- Video metadata is read with OpenCV at runtime.
- The solution does not hard-code FPS, frame count, or video duration.

Current video metadata from the generated run:

```text
width: 1920
height: 1080
fps: 29.883
frame_count: 2556
duration_sec: 85.535
```

## Stationary Definition

Stationary means the person's position in the scene is almost unchanged. It does not require the whole body to be perfectly still.

The pipeline uses the bbox foot point `(center_x, y2)` because it better approximates the person's floor contact point than the bbox center.
