# Part 4: Evaluation and Results

## Generated Result

The generated run in this workspace used:

```text
model: yolo11n.pt
tracker: botsort.yaml
confidence: 0.25
image size: 640
stationary threshold: 0.05
window: 1.0 sec
minimum stationary duration: 0.75 sec
```

Winner:

```text
track_id: 161
source_track_ids: [161]
longest stationary duration: 41.094 seconds
start_time: 33.699 seconds
end_time: 74.793 seconds
```

Output files:

- `results/annotated_entrance.mp4` (generated locally and intentionally excluded from this public repository)
- `results/summary.json`
- `results/tracks.csv`

## Is It Good Enough?

For a coding-test submission, the current version is solid because it satisfies the required artifacts and uses a real detector plus tracker instead of a toy heuristic.

The strongest parts are:

- full runnable Python pipeline
- OpenCV metadata instead of hard-coded FPS/duration
- pretrained person detector and multi-object tracker
- annotated result video with bbox, track id, status, and duration
- JSON and CSV summaries
- clear README, method, limitations, and references

The main quality risk is not code completeness; it is tracking reliability. If the reviewer visually checks the video, the answer can still be affected by missed detections, ID switches, and threshold choices.

## Recommended Improvements

High value:

- Run a second result with `yolo11s.pt` and `imgsz=960` for better detection quality, then compare winner stability against the current fast `yolo11n.pt` result.
- Add a short qualitative review table with 5-8 timestamps showing whether the winner track is visually stationary.
- Review private/local preview frames around the winning interval without publishing them.

Medium value:

- Add an optional `--save-debug-csv` with per-frame `norm_disp`, IoU, status, and bbox for the winning track.
- Try both `botsort.yaml` and `bytetrack.yaml`, then report if the winner remains the same.
- Add optional global motion compensation if camera shake exists.

Research-backed extensions:

- Use BoT-SORT because it combines motion, appearance cues, and camera-motion compensation ideas for robust pedestrian tracking.
- Use ByteTrack as an alternative because it can recover fragmented tracks by associating lower-confidence detections.
- If ground truth is ever available, evaluate tracking with HOTA/IDF1-style metrics rather than only visual inspection.
