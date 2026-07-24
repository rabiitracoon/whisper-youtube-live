from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lecture_scribe.audio_edit import extract_waveform, render_edited_audio
from lecture_scribe.transcription import transcribe_audio
from lecture_scribe.youtube import download_audio, fetch_video_info


DEFAULT_TEST_VIDEO = "https://www.youtube.com/watch?v=jNQXAC9IVRw"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEST_VIDEO
    print("Reading video metadata...")
    video = fetch_video_info(url, log=print)
    print(f"Video: {video.title} ({video.duration:.1f}s)")
    with tempfile.TemporaryDirectory(prefix="lecture-scribe-smoke-") as directory:
        root = Path(directory)
        audio = download_audio(
            video.webpage_url,
            video.video_id,
            root,
            progress=lambda ratio, detail: print(f"download {ratio:.0%}: {detail}"),
            log=print,
        )
        waveform_duration, peaks = extract_waveform(audio, video.duration, bins=600)
        if waveform_duration < 18 or not any(peaks):
            print("Waveform extraction failed.", file=sys.stderr)
            return 4
        kept_ranges = ((2, 7), (10, min(16, video.duration)))
        edited_audio = render_edited_audio(
            audio,
            root / "edited_lecture.flac",
            kept_ranges,
            waveform_duration,
            progress=lambda ratio: print(f"render {ratio:.0%}"),
        )
        if not edited_audio.exists() or edited_audio.stat().st_size == 0:
            print("Edited FLAC rendering failed.", file=sys.stderr)
            return 5
        result = transcribe_audio(
            audio_path=edited_audio,
            output_dir=root,
            video=video,
            model_name="large-v3",
            language="auto",
            compute_type="float16",
            beam_size=5,
            vad_filter=True,
            original_ranges=kept_ranges,
            progress=lambda ratio, detail: print(f"transcribe {ratio:.0%}: {detail}"),
            log=print,
        )
        if not result.segments:
            print("No transcript segments were produced.", file=sys.stderr)
            return 3
        print(f"Detected language: {result.language} ({result.language_probability:.1%})")
        print(f"Segments: {len(result.segments)}")
        print("First segment:", result.segments[0].text)
        if result.clip_ranges != ((2, 7), (10, 16)):
            print("Requested transcription range was not preserved.", file=sys.stderr)
            return 6
        if result.segments[0].start < 1.9 or result.segments[-1].end > 16.1:
            print("A segment escaped the requested transcription range.", file=sys.stderr)
            return 7
        if any(7 < item.start < 10 for item in result.segments):
            print("A segment started inside the removed break range.", file=sys.stderr)
            return 8
    print("YouTube + Whisper large-v3 integration passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
