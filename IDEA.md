# Short Idea Description

Use a pretrained YOLO person detector plus a multi-object tracker to keep a track id for each person. For every track, measure the floor-position proxy of the person with the bbox foot point `(center_x, y2)`, smooth it with EMA, then check how far it moved during a sliding one-second window after normalizing by bbox height.

A person is counted as stationary when the normalized foot-point displacement stays below a threshold, with bbox IoU used as supporting evidence and hysteresis used to avoid splitting segments from detector jitter or short missed detections. The final answer is the track whose longest stationary segment has the largest duration.