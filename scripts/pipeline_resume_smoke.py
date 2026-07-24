from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lecture_scribe.config import AppSettings
from lecture_scribe.pipeline import run_pipeline


TEST_VIDEO = "https://www.youtube.com/watch?v=jNQXAC9IVRw"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    with tempfile.TemporaryDirectory(prefix="lecture-scribe-resume-") as directory:
        settings = AppSettings(
            output_dir=directory,
            language="en",
            keep_audio=True,
        )
        downloaded = run_pipeline(
            TEST_VIDEO,
            settings,
            target_stage="download",
            log=print,
        )
        edited = run_pipeline(
            TEST_VIDEO,
            settings,
            target_stage="edit",
            edit_audio=lambda _request: ((7.0, 10.0),),
            log=print,
        )
        transcribed = run_pipeline(
            TEST_VIDEO,
            settings,
            target_stage="transcribe",
            review_transcript=lambda _review: True,
            log=print,
        )
        if len({downloaded.output_dir, edited.output_dir, transcribed.output_dir}) != 1:
            print("Stages did not resume the same job directory.", file=sys.stderr)
            return 3
        checkpoint = json.loads(
            (transcribed.output_dir / "checkpoint.json").read_text(encoding="utf-8")
        )
        if checkpoint["completed_stages"] != ["download", "edit", "transcribe"]:
            print(f"Unexpected checkpoint: {checkpoint['completed_stages']}", file=sys.stderr)
            return 4
        if not (transcribed.output_dir / "edited_lecture.flac").exists():
            print("Edited audio checkpoint is missing.", file=sys.stderr)
            return 5
    print("Download -> edit -> transcribe checkpoint resume passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
