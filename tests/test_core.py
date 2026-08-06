from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

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
from lecture_scribe.time_range import (
    format_timecode,
    parse_timecode,
    validate_clip_range,
)
from lecture_scribe.transcription import TranscriptResult, TranscriptSegment
from lecture_scribe.video_edit import build_video_filter_graph, render_edited_video
from lecture_scribe.youtube import (
    VideoInfo,
    _is_http_403,
    download_audio,
    download_video,
    format_duration,
    is_youtube_url,
)


class UrlTests(unittest.TestCase):
    def test_accepts_common_youtube_urls(self) -> None:
        self.assertTrue(is_youtube_url("https://www.youtube.com/watch?v=abc123"))
        self.assertTrue(is_youtube_url("https://youtu.be/abc123?t=30"))
        self.assertTrue(is_youtube_url("https://music.youtube.com/watch?v=abc123"))

    def test_rejects_non_youtube_and_invalid_schemes(self) -> None:
        self.assertFalse(is_youtube_url("https://example.com/watch?v=abc123"))
        self.assertFalse(is_youtube_url("javascript:alert(1)"))
        self.assertFalse(is_youtube_url("youtube.com/watch?v=abc123"))

    def test_identifies_wrapped_http_403_download_errors(self) -> None:
        wrapped = RuntimeError("download failed")
        wrapped.__cause__ = OSError("HTTP Error 403: Forbidden")
        self.assertTrue(_is_http_403(wrapped))
        self.assertFalse(_is_http_403(RuntimeError("HTTP Error 429")))

    def test_download_retries_http_403_with_mweb_client(self) -> None:
        from yt_dlp.utils import DownloadError

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            downloaded = root / "video.mp4"
            downloaded.write_bytes(b"audio")
            first = MagicMock()
            first.__enter__.return_value.extract_info.side_effect = DownloadError(
                "HTTP Error 403: Forbidden"
            )
            second = MagicMock()
            second.__enter__.return_value.extract_info.return_value = {
                "requested_downloads": [{"filepath": str(downloaded)}]
            }
            with patch("yt_dlp.YoutubeDL", side_effect=[first, second]) as ydl:
                result = download_audio("https://youtu.be/video", "video", root)

            self.assertEqual(result, downloaded.resolve())
            fallback_options = ydl.call_args_list[1].args[0]
            self.assertEqual(
                fallback_options["extractor_args"]["youtube"]["player_client"],
                ["mweb"],
            )

    def test_video_download_retries_http_403_with_mweb_client(self) -> None:
        from yt_dlp.utils import DownloadError

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            downloaded = root / "original_video.mp4"
            downloaded.write_bytes(b"video")
            first = MagicMock()
            first.__enter__.return_value.extract_info.side_effect = DownloadError(
                "HTTP Error 403: Forbidden"
            )
            second = MagicMock()
            second.__enter__.return_value.extract_info.return_value = {
                "requested_downloads": [{"filepath": str(downloaded)}]
            }
            with patch("yt_dlp.YoutubeDL", side_effect=[first, second]) as ydl:
                result = download_video("https://youtu.be/video", "video", root)

            self.assertEqual(result, downloaded.resolve())
            fallback_options = ydl.call_args_list[1].args[0]
            self.assertEqual(fallback_options["format"], "best[height<=1080]/best")
            self.assertEqual(
                fallback_options["extractor_args"]["youtube"]["player_client"],
                ["mweb"],
            )


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


class VideoEditTests(unittest.TestCase):
    def test_builds_synchronized_video_and_audio_concat_filter(self) -> None:
        graph = build_video_filter_graph(((10, 20), (40.5, 60)))
        self.assertIn("[0:v]trim=start=10.000000:end=20.000000", graph)
        self.assertIn("[0:a]atrim=start=40.500000:end=60.000000", graph)
        self.assertIn("[v0][a0][v1][a1]concat=n=2:v=1:a=1[outv][outa]", graph)

    def test_renders_video_and_audio_with_removed_middle_range(self) -> None:
        import av
        import imageio_ffmpeg

        from lecture_scribe.video_edit import _ffmpeg_nvenc_usable

        if not _ffmpeg_nvenc_usable(imageio_ffmpeg.get_ffmpeg_exe()):
            self.skipTest("NVIDIA NVENC is not currently available")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            output = root / "edited.mp4"
            messages: list[str] = []
            subprocess.run(
                [
                    imageio_ffmpeg.get_ffmpeg_exe(),
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=320x180:rate=24:duration=4",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=440:duration=4",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-shortest",
                    str(source),
                ],
                check=True,
                capture_output=True,
            )
            render_edited_video(
                source, output, ((0, 1), (2, 3.5)), log=messages.append
            )

            with av.open(str(output)) as container:
                self.assertTrue(any(stream.type == "video" for stream in container.streams))
                self.assertTrue(any(stream.type == "audio" for stream in container.streams))
                duration = float(container.duration / av.time_base)
            self.assertGreater(duration, 2.35)
            self.assertLess(duration, 2.7)
            self.assertTrue(any("NVIDIA NVENC" in item for item in messages))

    def test_does_not_fall_back_to_cpu_when_nvenc_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            source.write_bytes(b"source")
            with (
                patch(
                    "lecture_scribe.video_edit._ffmpeg_nvenc_usable",
                    return_value=False,
                ),
                patch("lecture_scribe.video_edit._run_ffmpeg") as run_ffmpeg,
                self.assertRaisesRegex(RuntimeError, "CPU로 대체하지 않았습니다"),
            ):
                render_edited_video(source, root / "edited.mp4", ((0, 1),))
            run_ffmpeg.assert_not_called()


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
            settings = AppSettings(
                beam_size=7, codex_model="gpt-5.6-sol", keep_video=True
            )
            settings.save(path)
            loaded = AppSettings.load(path)
            self.assertEqual(loaded.beam_size, 7)
            self.assertEqual(loaded.codex_model, "gpt-5.6-sol")
            self.assertTrue(loaded.keep_video)


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
