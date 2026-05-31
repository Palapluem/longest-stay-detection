from __future__ import annotations

from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "longest_stay_pipeline.png"

WIDTH = 2160
HEIGHT = 1120

BACKGROUND = "#F8F9FB"
INK = "#18212B"
MUTED = "#5F6B76"
LINE = "#AAB4BE"
YELLOW = "#F2B705"
BLUE = "#2B6CB0"
TEAL = "#1F8A8A"
GREEN = "#2F855A"
ORANGE = "#C05621"
PURPLE = "#6B46C1"
WHITE = "#FFFFFF"


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = (
        "C:/Windows/Fonts/seguisb.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    )
    for name in names:
        path = Path(name)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


TITLE = load_font(52, bold=True)
SUBTITLE = load_font(25)
ZONE = load_font(25, bold=True)
BOX_TITLE = load_font(27, bold=True)
BOX_BODY = load_font(20)
SMALL = load_font(18)
SMALL_BOLD = load_font(18, bold=True)


def rounded_box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    fill: str,
    outline: str,
    radius: int = 18,
    width: int = 3,
) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: str = INK,
    width: int = 5,
    dashed: bool = False,
) -> None:
    x1, y1 = start
    x2, y2 = end
    if dashed:
        steps = 16
        for index in range(0, steps, 2):
            t1 = index / steps
            t2 = min(1.0, (index + 1) / steps)
            draw.line(
                (
                    x1 + (x2 - x1) * t1,
                    y1 + (y2 - y1) * t1,
                    x1 + (x2 - x1) * t2,
                    y1 + (y2 - y1) * t2,
                ),
                fill=color,
                width=width,
            )
    else:
        draw.line((x1, y1, x2, y2), fill=color, width=width)

    if abs(x2 - x1) >= abs(y2 - y1):
        direction = 1 if x2 >= x1 else -1
        head = [(x2, y2), (x2 - 18 * direction, y2 - 12), (x2 - 18 * direction, y2 + 12)]
    else:
        direction = 1 if y2 >= y1 else -1
        head = [(x2, y2), (x2 - 12, y2 - 18 * direction), (x2 + 12, y2 - 18 * direction)]
    draw.polygon(head, fill=color)


def box_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    title: str,
    lines: list[str],
    accent: str,
) -> None:
    x1, y1, x2, y2 = xy
    rounded_box(draw, xy, WHITE, accent)
    draw.rounded_rectangle((x1, y1, x2, y1 + 18), radius=8, fill=accent)
    title_font = BOX_TITLE
    while draw.textlength(title, font=title_font) > (x2 - x1 - 44) and getattr(title_font, "size", 18) > 20:
        title_font = load_font(title_font.size - 1, bold=True)
    draw.text((x1 + 22, y1 + 35), title, fill=INK, font=title_font)
    cursor = y1 + 82
    for line in lines:
        for wrapped in wrap(line, width=28):
            draw.text((x1 + 22, cursor), wrapped, fill=MUTED, font=BOX_BODY)
            cursor += 29
        cursor += 3


def zone_label(draw: ImageDraw.ImageDraw, x1: int, x2: int, title: str, color: str) -> None:
    draw.rounded_rectangle((x1, 150, x2, 195), radius=18, fill=color)
    width = draw.textlength(title, font=ZONE)
    draw.text(((x1 + x2 - width) / 2, 158), title, fill=WHITE, font=ZONE)


def main() -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)

    draw.text((78, 48), "Longest Stay Detection Pipeline", fill=INK, font=TITLE)
    draw.text(
        (80, 111),
        "Pretrained person detection + multi-object tracking + normalized stationary-time analysis",
        fill=MUTED,
        font=SUBTITLE,
    )

    zone_label(draw, 78, 305, "INPUT", BLUE)
    zone_label(draw, 350, 900, "DETECTION & TRACKING", TEAL)
    zone_label(draw, 950, 1710, "STATIONARY ANALYTICS", ORANGE)
    zone_label(draw, 1760, 2070, "OUTPUTS", GREEN)

    input_box = (78, 255, 305, 485)
    metadata_box = (78, 570, 305, 800)
    detector_box = (350, 255, 600, 485)
    tracker_box = (650, 255, 900, 485)
    history_box = (950, 255, 1210, 485)
    decision_box = (1260, 255, 1535, 485)
    winner_box = (1585, 255, 1710, 485)
    outputs_box = (1760, 255, 2070, 560)

    box_text(draw, input_box, "Input Video", ["entrance.mov", "Full frame sequence"], BLUE)
    box_text(
        draw,
        metadata_box,
        "OpenCV Metadata",
        ["FPS", "Frame count", "Width / height", "Duration from FPS"],
        BLUE,
    )
    box_text(draw, detector_box, "YOLO Detector", ["Ultralytics YOLO", "classes=[0] for person", "conf threshold", "imgsz tuning"], TEAL)
    box_text(draw, tracker_box, "Multi-Object Tracker", ["BoT-SORT default", "persist=True", "ByteTrack optional", "Stable track_id"], TEAL)
    box_text(draw, history_box, "Per-Track History", ["bbox + timestamp", "Foot point = (center_x, y2)", "EMA smoothing", "Recent sliding window"], ORANGE)
    box_text(draw, decision_box, "Stationary Decision", ["Normalized displacement", "BBox IoU support", "Entry / exit hysteresis", "Missing-gap tolerance"], ORANGE)
    box_text(draw, winner_box, "Winner", ["Longest", "stationary", "segment"], PURPLE)
    box_text(draw, outputs_box, "Artifacts", ["Annotated MP4", "summary.json", "tracks.csv", "Console winner summary"], GREEN)

    arrow(draw, (305, 370), (350, 370), TEAL)
    arrow(draw, (600, 370), (650, 370), TEAL)
    arrow(draw, (900, 370), (950, 370), ORANGE)
    arrow(draw, (1210, 370), (1260, 370), ORANGE)
    arrow(draw, (1535, 370), (1585, 370), PURPLE)
    arrow(draw, (1710, 370), (1760, 370), GREEN)
    arrow(draw, (190, 485), (190, 570), BLUE)

    merge_box = (650, 650, 1210, 930)
    box_text(
        draw,
        merge_box,
        "Conservative Track Fragment Merge",
        [
            "Optional summary-stage heuristic",
            "Short time gap",
            "Nearby foot point",
            "Similar bbox height",
        ],
        PURPLE,
    )
    arrow(draw, (775, 485), (775, 650), PURPLE, dashed=True)
    arrow(draw, (1210, 780), (1395, 485), PURPLE, dashed=True)

    draw.text((1370, 650), "Core stationary metric", fill=INK, font=SMALL_BOLD)
    draw.text((1370, 681), "distance(smoothed_foot_now, smoothed_foot_old)", fill=MUTED, font=SMALL)
    draw.line((1370, 710, 1880, 710), fill=MUTED, width=2)
    draw.text((1370, 727), "median bbox height", fill=MUTED, font=SMALL)
    draw.text((1370, 775), "Stationary when normalized movement stays low", fill=ORANGE, font=SMALL_BOLD)

    draw.text((78, 1015), "Design principle:", fill=INK, font=SMALL_BOLD)
    draw.text(
        (225, 1015),
        "measure each tracked person's scene position over time; do not rely on frame differencing alone.",
        fill=MUTED,
        font=SMALL,
    )
    draw.text((78, 1055), "Result:", fill=INK, font=SMALL_BOLD)
    draw.text(
        (150, 1055),
        "track_id 161 | longest stationary duration 41.094 seconds | interval 33.699s - 74.793s",
        fill=GREEN,
        font=SMALL_BOLD,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, optimize=True)
    print(OUTPUT)


if __name__ == "__main__":
    main()
