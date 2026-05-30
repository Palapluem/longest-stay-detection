from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

try:
    import cv2
except ImportError as exc:  # pragma: no cover - exercised only on incomplete environments.
    raise SystemExit(
        "Missing dependency 'opencv-python'. Install dependencies with "
        "`python -m pip install -r requirements.txt`."
    ) from exc

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - exercised only on incomplete environments.
    raise SystemExit(
        "Missing dependency 'numpy'. Install dependencies with "
        "`python -m pip install -r requirements.txt`."
    ) from exc


BBox = tuple[float, float, float, float]
Point = tuple[float, float]


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float


@dataclass
class TrackObservation:
    frame_idx: int
    timestamp: float
    bbox: BBox
    foot: Point
    smoothed_foot: Point
    bbox_height: float
    confidence: float
    norm_disp: float | None = None
    window_iou: float | None = None
    stationary_candidate: bool = False


@dataclass
class StationarySegment:
    start_time: float
    end_time: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    def to_dict(self) -> dict[str, float]:
        return {
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "duration": round(self.duration, 3),
        }


@dataclass
class TrackState:
    track_id: int
    observations: list[TrackObservation] = field(default_factory=list)
    recent: deque[TrackObservation] = field(default_factory=deque)
    is_stationary: bool = False
    stationary_candidate_since: float | None = None
    stationary_started_at: float | None = None
    moving_since: float | None = None
    last_seen_frame: int = -1
    last_seen_time: float = 0.0
    last_smoothed_foot: Point | None = None

    @property
    def first_seen_time(self) -> float:
        return self.observations[0].timestamp if self.observations else 0.0

    @property
    def visible_time(self) -> float:
        if not self.observations:
            return 0.0
        return self.observations[-1].timestamp - self.observations[0].timestamp


@dataclass
class FrameAnnotation:
    track_id: int
    bbox: BBox
    confidence: float
    status: str
    stationary_duration: float
    norm_disp: float | None


@dataclass
class TrackSummary:
    track_id: int
    source_track_ids: list[int]
    total_visible_time: float
    first_seen_time: float
    last_seen_time: float
    stationary_segments: list[StationarySegment]

    @property
    def longest_segment(self) -> StationarySegment | None:
        if not self.stationary_segments:
            return None
        return max(self.stationary_segments, key=lambda segment: segment.duration)


def load_video_metadata(video_path: str | Path) -> VideoMetadata:
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {path}")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()

    if width <= 0 or height <= 0:
        raise RuntimeError("Video metadata is invalid: width/height are unavailable.")
    if fps <= 0:
        raise RuntimeError("Video metadata is invalid: FPS is unavailable.")
    if frame_count <= 0:
        raise RuntimeError("Video metadata is invalid: frame count is unavailable.")

    return VideoMetadata(
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_sec=frame_count / fps,
    )


def bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def foot_point(bbox: BBox) -> Point:
    x1, _, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)


def euclidean(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def update_track_state(
    state: TrackState,
    frame_idx: int,
    timestamp: float,
    bbox: BBox,
    confidence: float,
    window_sec: float,
    stationary_threshold: float,
    min_stationary_sec: float,
    iou_threshold: float,
    exit_hysteresis_sec: float,
    ema_alpha: float = 0.35,
) -> FrameAnnotation:
    foot = foot_point(bbox)
    bbox_height = max(1.0, bbox[3] - bbox[1])

    if state.last_smoothed_foot is None:
        smoothed_foot = foot
    else:
        smoothed_foot = (
            ema_alpha * foot[0] + (1.0 - ema_alpha) * state.last_smoothed_foot[0],
            ema_alpha * foot[1] + (1.0 - ema_alpha) * state.last_smoothed_foot[1],
        )

    observation = TrackObservation(
        frame_idx=frame_idx,
        timestamp=timestamp,
        bbox=bbox,
        foot=foot,
        smoothed_foot=smoothed_foot,
        bbox_height=bbox_height,
        confidence=confidence,
    )

    state.recent.append(observation)
    while state.recent and state.recent[0].timestamp < timestamp - window_sec:
        state.recent.popleft()

    candidate = False
    norm_disp: float | None = None
    window_iou: float | None = None
    if len(state.recent) >= 2:
        reference = state.recent[0]
        window_age = timestamp - reference.timestamp
        if window_age >= max(0.20, window_sec * 0.65):
            median_height = max(1.0, median(obs.bbox_height for obs in state.recent))
            norm_disp = euclidean(smoothed_foot, reference.smoothed_foot) / median_height
            window_iou = bbox_iou(bbox, reference.bbox)
            candidate = norm_disp <= stationary_threshold or (
                norm_disp <= stationary_threshold * 1.5 and window_iou >= iou_threshold
            )

    observation.norm_disp = norm_disp
    observation.window_iou = window_iou
    observation.stationary_candidate = candidate
    state.observations.append(observation)

    if candidate:
        state.moving_since = None
        if state.stationary_candidate_since is None:
            state.stationary_candidate_since = timestamp
        if (
            not state.is_stationary
            and timestamp - state.stationary_candidate_since >= min_stationary_sec
        ):
            state.is_stationary = True
            state.stationary_started_at = state.stationary_candidate_since
    else:
        state.stationary_candidate_since = None
        if state.is_stationary:
            if state.moving_since is None:
                state.moving_since = timestamp
            if timestamp - state.moving_since >= exit_hysteresis_sec:
                state.is_stationary = False
                state.stationary_started_at = None
                state.moving_since = None
        else:
            state.moving_since = None

    state.last_seen_frame = frame_idx
    state.last_seen_time = timestamp
    state.last_smoothed_foot = smoothed_foot

    stationary_duration = (
        timestamp - state.stationary_started_at
        if state.is_stationary and state.stationary_started_at is not None
        else 0.0
    )
    return FrameAnnotation(
        track_id=state.track_id,
        bbox=bbox,
        confidence=confidence,
        status="stationary" if state.is_stationary else "moving",
        stationary_duration=stationary_duration,
        norm_disp=norm_disp,
    )


def compute_stationary_segments(
    observations: list[TrackObservation],
    min_stationary_sec: float,
    exit_hysteresis_sec: float,
    max_missing_sec: float,
) -> list[StationarySegment]:
    if not observations:
        return []

    sorted_observations = sorted(observations, key=lambda obs: (obs.timestamp, obs.frame_idx))
    segments: list[StationarySegment] = []
    candidate_since: float | None = None
    active_start: float | None = None
    moving_since: float | None = None
    previous_time = sorted_observations[0].timestamp

    for obs in sorted_observations:
        if active_start is not None and obs.timestamp - previous_time > max_missing_sec:
            segments.append(StationarySegment(active_start, previous_time))
            active_start = None
            candidate_since = None
            moving_since = None

        if obs.stationary_candidate:
            moving_since = None
            if candidate_since is None:
                candidate_since = obs.timestamp
            if active_start is None and obs.timestamp - candidate_since >= min_stationary_sec:
                active_start = candidate_since
        else:
            candidate_since = None
            if active_start is not None:
                if moving_since is None:
                    moving_since = obs.timestamp
                if obs.timestamp - moving_since >= exit_hysteresis_sec:
                    segments.append(StationarySegment(active_start, moving_since))
                    active_start = None
                    moving_since = None

        previous_time = obs.timestamp

    if active_start is not None:
        segments.append(StationarySegment(active_start, sorted_observations[-1].timestamp))

    return [segment for segment in segments if segment.duration >= min_stationary_sec]


def merge_track_fragments(
    tracks: dict[int, TrackState],
    max_gap_sec: float = 0.70,
    max_norm_distance: float = 0.35,
    min_height_ratio: float = 0.60,
) -> dict[int, list[int]]:
    """Conservatively merge short track fragments that look spatially continuous."""
    ordered = sorted(
        (track for track in tracks.values() if track.observations),
        key=lambda track: track.first_seen_time,
    )
    parent: dict[int, int] = {track.track_id: track.track_id for track in ordered}

    def root(track_id: int) -> int:
        while parent[track_id] != track_id:
            parent[track_id] = parent[parent[track_id]]
            track_id = parent[track_id]
        return track_id

    for current in ordered:
        current_root = root(current.track_id)
        current_last = tracks[current_root].observations[-1]
        for candidate in ordered:
            if candidate.track_id == current_root or root(candidate.track_id) != candidate.track_id:
                continue
            gap = candidate.first_seen_time - current_last.timestamp
            if gap < 0 or gap > max_gap_sec:
                continue

            candidate_first = candidate.observations[0]
            size_ratio = min(current_last.bbox_height, candidate_first.bbox_height) / max(
                current_last.bbox_height,
                candidate_first.bbox_height,
            )
            median_height = max(1.0, (current_last.bbox_height + candidate_first.bbox_height) / 2.0)
            norm_distance = euclidean(
                current_last.smoothed_foot,
                candidate_first.smoothed_foot,
            ) / median_height

            if size_ratio >= min_height_ratio and norm_distance <= max_norm_distance:
                parent[candidate.track_id] = current_root

    groups: dict[int, list[int]] = defaultdict(list)
    for track_id in parent:
        groups[root(track_id)].append(track_id)
    return {root_id: sorted(ids) for root_id, ids in groups.items()}


def build_track_summaries(
    tracks: dict[int, TrackState],
    groups: dict[int, list[int]],
    min_stationary_sec: float,
    exit_hysteresis_sec: float,
    max_missing_sec: float,
) -> list[TrackSummary]:
    summaries: list[TrackSummary] = []
    for root_id, source_ids in groups.items():
        observations: list[TrackObservation] = []
        for source_id in source_ids:
            observations.extend(tracks[source_id].observations)
        observations.sort(key=lambda obs: (obs.timestamp, obs.frame_idx))
        if not observations:
            continue

        segments = compute_stationary_segments(
            observations,
            min_stationary_sec=min_stationary_sec,
            exit_hysteresis_sec=exit_hysteresis_sec,
            max_missing_sec=max_missing_sec,
        )
        summaries.append(
            TrackSummary(
                track_id=root_id,
                source_track_ids=source_ids,
                total_visible_time=observations[-1].timestamp - observations[0].timestamp,
                first_seen_time=observations[0].timestamp,
                last_seen_time=observations[-1].timestamp,
                stationary_segments=segments,
            )
        )
    return sorted(summaries, key=lambda summary: summary.track_id)


def load_yolo_model(model_name: str) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'ultralytics'. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    candidates = [model_name]
    for fallback in ("yolo11n.pt", "yolov8n.pt"):
        if fallback not in candidates:
            candidates.append(fallback)

    errors: list[str] = []
    for candidate in candidates:
        try:
            print(f"Loading YOLO model: {candidate}", flush=True)
            return YOLO(candidate)
        except Exception as exc:  # noqa: BLE001 - keep fallback helpful for model download failures.
            errors.append(f"{candidate}: {exc}")

    raise RuntimeError("Could not load any YOLO model:\n" + "\n".join(errors))


def extract_track_detections(result: Any) -> list[tuple[int, BBox, float]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.id is None:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    ids = boxes.id.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(ids), dtype=float)

    detections: list[tuple[int, BBox, float]] = []
    for track_id, bbox_arr, conf in zip(ids, xyxy, confs, strict=False):
        x1, y1, x2, y2 = [float(value) for value in bbox_arr]
        detections.append((int(track_id), (x1, y1, x2, y2), float(conf)))
    return detections


def run_detection_tracking(args: argparse.Namespace, metadata: VideoMetadata) -> tuple[
    dict[int, TrackState],
    dict[int, list[FrameAnnotation]],
]:
    model = load_yolo_model(args.model)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {args.video}")

    tracks: dict[int, TrackState] = {}
    frame_annotations: dict[int, list[FrameAnnotation]] = {}
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            timestamp = frame_idx / metadata.fps
            try:
                results = model.track(
                    frame,
                    persist=True,
                    tracker=args.tracker,
                    classes=[0],
                    conf=args.conf,
                    imgsz=args.imgsz,
                    verbose=False,
                )
            except Exception as exc:  # noqa: BLE001
                if args.tracker == "botsort.yaml":
                    print("BoT-SORT failed; retrying this frame with ByteTrack.", flush=True)
                    args.tracker = "bytetrack.yaml"
                    results = model.track(
                        frame,
                        persist=True,
                        tracker=args.tracker,
                        classes=[0],
                        conf=args.conf,
                        imgsz=args.imgsz,
                        verbose=False,
                    )
                else:
                    raise RuntimeError(f"Tracking failed with {args.tracker}: {exc}") from exc

            annotations: list[FrameAnnotation] = []
            detections = extract_track_detections(results[0]) if results else []
            for track_id, bbox, confidence in detections:
                state = tracks.setdefault(track_id, TrackState(track_id=track_id))
                annotation = update_track_state(
                    state=state,
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    bbox=bbox,
                    confidence=confidence,
                    window_sec=args.window_sec,
                    stationary_threshold=args.stationary_threshold,
                    min_stationary_sec=args.min_stationary_sec,
                    iou_threshold=args.iou_threshold,
                    exit_hysteresis_sec=args.exit_hysteresis_sec,
                )
                annotations.append(annotation)

            frame_annotations[frame_idx] = annotations
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx}/{metadata.frame_count} frames", flush=True)
    finally:
        cap.release()

    return tracks, frame_annotations


def draw_label(frame: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.52
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    y = max(text_h + baseline + 4, y)
    cv2.rectangle(
        frame,
        (x, y - text_h - baseline - 6),
        (x + text_w + 8, y + baseline + 2),
        color,
        thickness=-1,
    )
    cv2.putText(frame, text, (x + 4, y - 4), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_annotations(
    frame: np.ndarray,
    annotations: list[FrameAnnotation],
    timestamp: float,
    winner_source_ids: set[int],
) -> np.ndarray:
    for annotation in annotations:
        x1, y1, x2, y2 = [int(round(value)) for value in annotation.bbox]
        is_winner = annotation.track_id in winner_source_ids
        if is_winner:
            color = (0, 0, 255)
        elif annotation.status == "stationary":
            color = (0, 170, 0)
        else:
            color = (255, 120, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        duration = annotation.stationary_duration if annotation.status == "stationary" else 0.0
        label = f"ID {annotation.track_id} {annotation.status} {duration:.1f}s"
        if is_winner:
            label = "WINNER " + label
        draw_label(frame, label, (x1, max(0, y1 - 8)), color)

    cv2.putText(
        frame,
        f"t={timestamp:.1f}s",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def write_annotated_video(
    video_path: Path,
    output_path: Path,
    metadata: VideoMetadata,
    frame_annotations: dict[int, list[FrameAnnotation]],
    winner_source_ids: set[int],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not reopen video for annotation: {video_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        metadata.fps,
        (metadata.width, metadata.height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"OpenCV could not create output video: {output_path}")

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = frame_idx / metadata.fps
            annotated = draw_annotations(
                frame,
                frame_annotations.get(frame_idx, []),
                timestamp,
                winner_source_ids,
            )
            writer.write(annotated)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()


def save_summary(
    summaries: list[TrackSummary],
    metadata: VideoMetadata,
    args: argparse.Namespace,
    output_video: Path,
) -> dict[str, Any]:
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    winner = max(
        summaries,
        key=lambda summary: summary.longest_segment.duration if summary.longest_segment else 0.0,
        default=None,
    )
    winner_segment = winner.longest_segment if winner else None

    summary_payload: dict[str, Any] = {
        "video": str(args.video),
        "metadata": {
            "width": metadata.width,
            "height": metadata.height,
            "fps": round(metadata.fps, 3),
            "frame_count": metadata.frame_count,
            "duration_sec": round(metadata.duration_sec, 3),
        },
        "method": {
            "detector": "Ultralytics YOLO person class only (classes=[0])",
            "tracker": args.tracker,
            "movement_measure": "EMA-smoothed foot-point displacement normalized by median bbox height, with bbox IoU support and hysteresis.",
            "stationary_threshold": args.stationary_threshold,
            "window_sec": args.window_sec,
            "min_stationary_sec": args.min_stationary_sec,
            "iou_threshold": args.iou_threshold,
        },
        "winner": None,
        "tracks": [],
        "outputs": {
            "summary_json": str(results_dir / "summary.json"),
            "tracks_csv": str(results_dir / "tracks.csv"),
            "annotated_video": str(output_video),
        },
    }

    if winner and winner_segment:
        summary_payload["winner"] = {
            "track_id": winner.track_id,
            "source_track_ids": winner.source_track_ids,
            "longest_stationary_duration": round(winner_segment.duration, 3),
            "start_time": round(winner_segment.start_time, 3),
            "end_time": round(winner_segment.end_time, 3),
        }

    for summary in summaries:
        longest = summary.longest_segment
        summary_payload["tracks"].append(
            {
                "track_id": summary.track_id,
                "source_track_ids": summary.source_track_ids,
                "total_visible_time": round(summary.total_visible_time, 3),
                "first_seen_time": round(summary.first_seen_time, 3),
                "last_seen_time": round(summary.last_seen_time, 3),
                "stationary_segments": [segment.to_dict() for segment in summary.stationary_segments],
                "longest_stationary_segment": longest.to_dict() if longest else None,
            }
        )

    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    csv_path = results_dir / "tracks.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "track_id",
                "source_track_ids",
                "total_visible_time",
                "stationary_segment_start",
                "stationary_segment_end",
                "stationary_segment_duration",
                "is_longest_for_track",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            longest = summary.longest_segment
            if not summary.stationary_segments:
                writer.writerow(
                    {
                        "track_id": summary.track_id,
                        "source_track_ids": " ".join(str(track_id) for track_id in summary.source_track_ids),
                        "total_visible_time": f"{summary.total_visible_time:.3f}",
                        "stationary_segment_start": "",
                        "stationary_segment_end": "",
                        "stationary_segment_duration": "0.000",
                        "is_longest_for_track": "false",
                    }
                )
                continue
            for segment in summary.stationary_segments:
                writer.writerow(
                    {
                        "track_id": summary.track_id,
                        "source_track_ids": " ".join(str(track_id) for track_id in summary.source_track_ids),
                        "total_visible_time": f"{summary.total_visible_time:.3f}",
                        "stationary_segment_start": f"{segment.start_time:.3f}",
                        "stationary_segment_end": f"{segment.end_time:.3f}",
                        "stationary_segment_duration": f"{segment.duration:.3f}",
                        "is_longest_for_track": str(segment is longest).lower(),
                    }
                )

    return summary_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect the person who stayed stationary the longest.")
    parser.add_argument("--video", type=Path, default=Path("entrance.mov"), help="Input video path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/annotated_entrance.mp4"),
        help="Annotated output video path.",
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Directory for JSON/CSV outputs.")
    parser.add_argument("--model", default="yolo11s.pt", help="YOLO model weights, e.g. yolo11s.pt or yolo11n.pt.")
    parser.add_argument("--tracker", default="botsort.yaml", help="Ultralytics tracker config.")
    parser.add_argument("--conf", type=float, default=0.25, help="Detector confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO inference image size.")
    parser.add_argument(
        "--stationary-threshold",
        type=float,
        default=0.05,
        help="Max normalized foot-point displacement inside the time window.",
    )
    parser.add_argument("--window-sec", type=float, default=1.0, help="Sliding window size in seconds.")
    parser.add_argument(
        "--min-stationary-sec",
        type=float,
        default=0.75,
        help="Minimum continuous stationary evidence before starting a segment.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.82,
        help="BBox IoU support threshold for stationary classification.",
    )
    parser.add_argument(
        "--exit-hysteresis-sec",
        type=float,
        default=0.35,
        help="Moving evidence duration needed before closing a stationary segment.",
    )
    parser.add_argument(
        "--max-missing-sec",
        type=float,
        default=0.50,
        help="Allowed detection gap before closing a stationary segment.",
    )
    parser.add_argument(
        "--no-merge-fragments",
        action="store_true",
        help="Disable conservative short-gap track fragment merging in summaries.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        metadata = load_video_metadata(args.video)
        print(
            "Video metadata: "
            f"{metadata.width}x{metadata.height}, "
            f"{metadata.fps:.3f} FPS, "
            f"{metadata.frame_count} frames, "
            f"{metadata.duration_sec:.2f}s"
        )

        tracks, frame_annotations = run_detection_tracking(args, metadata)
        if args.no_merge_fragments:
            groups = {track_id: [track_id] for track_id in tracks}
        else:
            groups = merge_track_fragments(tracks)

        summaries = build_track_summaries(
            tracks=tracks,
            groups=groups,
            min_stationary_sec=args.min_stationary_sec,
            exit_hysteresis_sec=args.exit_hysteresis_sec,
            max_missing_sec=args.max_missing_sec,
        )
        winner = max(
            summaries,
            key=lambda summary: summary.longest_segment.duration if summary.longest_segment else 0.0,
            default=None,
        )
        winner_source_ids = set(winner.source_track_ids) if winner else set()

        write_annotated_video(
            video_path=args.video,
            output_path=args.output,
            metadata=metadata,
            frame_annotations=frame_annotations,
            winner_source_ids=winner_source_ids,
        )
        summary_payload = save_summary(summaries, metadata, args, args.output)

        print("\nMethod summary:")
        print("- Person detection: Ultralytics YOLO with classes=[0].")
        print(f"- Tracking: {args.tracker}.")
        print("- Stationary measure: normalized EMA foot-point displacement + bbox IoU + hysteresis.")
        if summary_payload["winner"]:
            winner_info = summary_payload["winner"]
            print("\nWinner:")
            print(f"- track_id: {winner_info['track_id']}")
            print(f"- source_track_ids: {winner_info['source_track_ids']}")
            print(f"- longest stationary duration: {winner_info['longest_stationary_duration']:.3f}s")
            print(f"- start/end: {winner_info['start_time']:.3f}s - {winner_info['end_time']:.3f}s")
        else:
            print("\nWinner: no stationary segment found with current thresholds.")

        print("\nOutputs:")
        print(f"- {args.output}")
        print(f"- {Path(args.results_dir) / 'summary.json'}")
        print(f"- {Path(args.results_dir) / 'tracks.csv'}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
