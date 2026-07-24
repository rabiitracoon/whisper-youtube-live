from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lecture_scribe.audio_edit import (
    build_included_ranges,
    cuts_to_segments,
    find_long_non_speech_ranges,
    map_edited_time_to_original,
    normalize_exclusions,
    refine_edge_exclusion_ranges,
)
from lecture_scribe.config import AppSettings
from lecture_scribe.paths import safe_filename
from lecture_scribe.pipeline import (
    _build_prompt,
    _ensure_state,
    _invalidate_after,
    _load_checkpoint,
    _mark_stage,
    _remove_original_audio,
)
from lecture_scribe.prompting import PromptContext, render_prompt, validate_prompt
from lecture_scribe.quality import assess_transcript
from lecture_scribe.time_range import format_timecode, parse_timecode, validate_clip_range
from lecture_scribe.transcription import TranscriptResult, TranscriptSegment
from lecture_scribe.youtube import VideoInfo, format_duration, is_youtube_url


class UrlTests(unittest.TestCase):
    def test_accepts_common_youtube_urls(self) -> None:
        self.assertTrue(is_youtube_url("https://www.youtube.com/watch?v=abc123"))
        self.assertTrue(is_youtube_url("https://youtu.be/abc123?t=30"))
        self.assertTrue(is_youtube_url("https://music.youtube.com/watch?v=abc123"))

    def test_rejects_non_youtube_and_invalid_schemes(self) -> None:
        self.assertFalse(is_youtube_url("https://example.com/watch?v=abc123"))
        self.assertFalse(is_youtube_url("javascript:alert(1)"))
        self.assertFalse(is_youtube_url("youtube.com/watch?v=abc123"))


class FormattingTests(unittest.TestCase):
    def test_duration(self) -> None:
        self.assertEqual(format_duration(0), "00:00:00")
        self.assertEqual(format_duration(3661.9), "01:01:01")

    def test_safe_windows_filename(self) -> None:
        self.assertEqual(safe_filename('A: B / C? "D"'), "A B C D")


class TimeRangeTests(unittest.TestCase):
    def test_parses_supported_timecode_formats(self) -> None:
        self.assertEqual(parse_timecode("90"), 90)
        self.assertEqual(parse_timecode("02:30"), 150)
        self.assertEqual(parse_timecode("01:02:03"), 3723)
        self.assertEqual(parse_timecode("01:02:03.250"), 3723.25)
        self.assertIsNone(parse_timecode(""))
        self.assertEqual(format_timecode(3723), "01:02:03")

    def test_validates_order_and_components(self) -> None:
        self.assertEqual(validate_clip_range(None, None), (0.0, None))
        with self.assertRaises(ValueError):
            validate_clip_range(120, 60)
        with self.assertRaises(ValueError):
            parse_timecode("01:70")


class AudioEditTests(unittest.TestCase):
    def test_merges_overlapping_breaks_and_builds_lecture_ranges(self) -> None:
        exclusions = normalize_exclusions(
            [(600, 900), (850, 1200), (2000, 2100)], 0, 3600
        )
        self.assertEqual(exclusions, ((600, 1200), (2000, 2100)))
        included = build_included_ranges(0, 3600, exclusions)
        self.assertEqual(included, ((0, 600), (1200, 2000), (2100, 3600)))

    def test_clips_breaks_to_editable_range(self) -> None:
        exclusions = normalize_exclusions([(0, 150), (850, 1200)], 100, 900)
        self.assertEqual(exclusions, ((100, 150), (850, 900)))

    def test_rejects_excluding_the_entire_lecture(self) -> None:
        with self.assertRaises(ValueError):
            build_included_ranges(0, 60, [(0, 60)])

    def test_razor_cuts_create_adjacent_clips(self) -> None:
        self.assertEqual(
            cuts_to_segments([20, 10], 0, 30),
            ((0, 10), (10, 20), (20, 30)),
        )

    def test_maps_edited_timeline_back_to_original(self) -> None:
        ranges = ((10, 20), (40, 60))
        self.assertEqual(map_edited_time_to_original(5, ranges), 15)
        self.assertEqual(map_edited_time_to_original(12, ranges), 42)

    def test_finds_long_leading_break_and_trailing_non_speech(self) -> None:
        candidates = find_long_non_speech_ranges(
            [(120, 600), (1200, 2400)], 0, 3000
        )
        self.assertEqual(candidates, ((0, 118), (602, 1198), (2402, 3000)))

    def test_ignores_short_normal_pauses(self) -> None:
        candidates = find_long_non_speech_ranges([(0, 100), (130, 300)], 0, 300)
        self.assertEqual(candidates, ())

    def test_extends_intro_candidate_to_the_exact_start(self) -> None:
        refined = refine_edge_exclusion_ranges(
            [(0, 60), (600, 1000)], [(62, 598)], 0, 1000
        )
        self.assertEqual(refined, ((0, 598),))

    def test_merges_brief_vad_false_positive_inside_intro_music(self) -> None:
        refined = refine_edge_exclusion_ranges(
            [(20, 51), (6007, 10000)],
            [(0, 18), (53, 6005)],
            0,
            10000,
        )
        self.assertEqual(refined, ((0, 6005),))

    def test_detects_a_short_outro_only_at_the_end_edge(self) -> None:
        refined = refine_edge_exclusion_ranges([(0, 987)], [], 0, 1000)
        self.assertEqual(refined, ((989, 1000),))

    def test_merges_brief_vad_false_positive_inside_outro_music(self) -> None:
        refined = refine_edge_exclusion_ranges(
            [(0, 898), (952, 968)],
            [(900, 950), (970, 995)],
            0,
            1000,
        )
        self.assertEqual(refined, ((900, 1000),))

    def test_does_not_apply_short_edge_threshold_to_middle_pauses(self) -> None:
        refined = refine_edge_exclusion_ranges(
            [(0, 100), (110, 200)], [], 0, 200
        )
        self.assertEqual(refined, ())


class PromptTests(unittest.TestCase):
    def test_transcript_placeholder_is_required(self) -> None:
        with self.assertRaises(ValueError):
            validate_prompt("x" * 100)

    def test_renders_all_context(self) -> None:
        template = (
            "강의 내용을 정확히 정리하세요. " * 8
            + "{{title}}|{{channel}}|{{url}}|{{duration}}|{{language}}|{{transcript}}"
        )
        rendered = render_prompt(
            template,
            PromptContext("제목", "채널", "https://youtu.be/x", "00:10:00", "ko", "본문"),
        )
        self.assertIn("제목|채널|https://youtu.be/x|00:10:00|ko|본문", rendered)

    def test_pipeline_sends_plain_transcript_without_timestamp_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text_path = root / "transcript.txt"
            markdown_path = root / "transcript.md"
            text_path.write_text("일반 전사 내용", encoding="utf-8")
            markdown_path.write_text("[00:01:20](https://youtu.be/x) 타임스탬프 전사", encoding="utf-8")
            transcript = TranscriptResult(
                text_path=text_path,
                markdown_path=markdown_path,
                srt_path=root / "transcript.srt",
                language="ko",
                language_probability=0.95,
                segments=(TranscriptSegment(80, 82, "일반 전사 내용"),),
                clip_start=80,
                clip_end=82,
                clip_ranges=((80, 82),),
            )
            template = "강의 필기 작성 지침입니다. " * 8 + "{{transcript}}"
            rendered = _build_prompt(
                template,
                VideoInfo("x", "제목", "채널", 100, "https://youtu.be/x"),
                transcript,
            )
            self.assertIn("일반 전사 내용", rendered)
            self.assertNotIn("00:01:20", rendered)


class SettingsTests(unittest.TestCase):
    def test_settings_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            settings = AppSettings(beam_size=7, codex_model="gpt-5.6-sol")
            settings.save(path)
            loaded = AppSettings.load(path)
            self.assertEqual(loaded.beam_size, 7)
            self.assertEqual(loaded.codex_model, "gpt-5.6-sol")


class CheckpointTests(unittest.TestCase):
    def test_stage_checkpoint_round_trip_and_invalidation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = VideoInfo("vid", "강의", "채널", 100, "https://youtu.be/vid")
            state: dict = {}
            _ensure_state(state, video, AppSettings())
            _mark_stage(root, state, "download", {"original_audio": "vid.webm"})
            _mark_stage(root, state, "edit", {"edited_audio": "edited_lecture.flac"})
            loaded = _load_checkpoint(root)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["completed_stages"], ["download", "edit"])
            _mark_stage(root, state, "transcribe")
            _invalidate_after(state, "edit")
            self.assertEqual(state["completed_stages"], ["download", "edit"])

    def test_original_audio_cleanup_removes_an_unlocked_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "source.webm"
            audio_path.write_bytes(b"audio")
            self.assertTrue(_remove_original_audio(audio_path, retry_delays=(0,)))
            self.assertFalse(audio_path.exists())

    def test_locked_original_is_preserved_without_failing_completed_work(self) -> None:
        audio_path = Path("locked.webm")
        messages: list[str] = []
        with patch.object(Path, "unlink", side_effect=PermissionError("locked")):
            removed = _remove_original_audio(
                audio_path, messages.append, retry_delays=(0, 0)
            )
        self.assertFalse(removed)
        self.assertTrue(any("결과는 정상 완료" in message for message in messages))


class TranscriptQualityTests(unittest.TestCase):
    def test_flags_sparse_long_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            text_path = Path(directory) / "transcript.txt"
            text_path.write_text("짧은 문장 하나\n", encoding="utf-8")
            result = TranscriptResult(
                text_path=text_path,
                markdown_path=Path(directory) / "transcript.md",
                srt_path=Path(directory) / "transcript.srt",
                language="ko",
                language_probability=0.9,
                segments=(TranscriptSegment(0, 2, "짧은 문장 하나", -0.4),),
                clip_start=0,
                clip_end=1200,
                clip_ranges=((0, 1200),),
            )
            review = assess_transcript(result)
            self.assertEqual(review.status, "warning")
            self.assertTrue(any("말의 양" in item for item in review.warnings))

    def test_default_prompt_is_note_style_without_timestamp_input(self) -> None:
        prompt = (Path(__file__).parent.parent / "prompts" / "default_lecture_notes_ko.md").read_text(encoding="utf-8")
        self.assertIn("실제 대학생", prompt)
        self.assertIn("타임스탬프", prompt)
        self.assertNotIn("{{url}}", prompt)


if __name__ == "__main__":
    unittest.main()
