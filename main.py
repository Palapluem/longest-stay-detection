from __future__ import annotations

import sys
from pathlib import Path


PIPELINE_SRC = Path(__file__).resolve().parent / "3_Video_Pipeline" / "src"
sys.path.insert(0, str(PIPELINE_SRC))

from longest_stay_detection import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
