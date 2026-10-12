from __future__ import annotations

import subprocess
import tempfile
import unittest
import wave
from types import SimpleNamespace
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

from lecture_scribe.audio_edit import (
    build_included_ranges,
    cuts_to_segments,
    find_long_non_speech_ranges,
    map_edited_time_to_original,
    normalize_exclusions,
    refine_edge_exclusion_ranges,
)
from lecture_scribe.config import AppSettings, load_prompt, save_prompt
from lecture_scribe.ai_providers import (
    CLAUDE,
    OPENAI,
    _parse_claude_models,
    _parse_codex_models,
    _parse_transcription_terms,
    fetch_catalog,
)
from lecture_scribe.live_caption import (
    SAMPLE_RATE,
    LiveCaptioner,
    LiveCaptionOptions,
    LiveSource,
    build_ffmpeg_command,
    is_silent,
    keep_segment,
    quietest_cut,
    resolve_live_source,
    trailing_quiet_seconds,
)
from lecture_scribe import model_download
from lecture_scribe.qwen_asr_engine import (
    AlignedWord,
    context_from_terms,
    group_words,
    is_qwen_model,
    split_pieces,
    transcribe_qwen,
)
from lecture_scribe.local_media import LocalMedia, copy_local_media, resolve_local_media
from lecture_scribe.paths import safe_filename
from lecture_scribe.pipeline import (
    _build_prompt,
    _ensure_state,
    _invalidate_after,
    _load_checkpoint,
    _mark_stage,
    _prepare_transcription_terms,
    _remove_original_audio,
)
from lecture_scribe.prompting import PromptContext, render_prompt, validate_prompt
from lecture_scribe.quality import assess_transcript
from lecture_scribe.time_range import format_timecode, parse_timecode, validate_clip_range
from lecture_scribe.transcription import (
    TranscriptResult,
    TranscriptSegment,
    _build_initial_prompt,
    resolve_mlx_model,
    transcribe_audio,
    transcribe_samples,
)
from lecture_scribe.video_edit import (
    _select_hardware_encoder,
    build_video_filter_graph,
    render_edited_video,
)
from lecture_scribe.youtube import (
    VideoInfo,
    _is_http_403,
    _progress_hook,
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

    def test_audio_download_always_uses_bundled_ffmpeg(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            downloaded = root / "video.webm"
            downloaded.write_bytes(b"audio")
            downloads = []
            for _ in range(2):
                ydl_instance = MagicMock()
                ydl_instance.__enter__.return_value.extract_info.return_value = {
                    "requested_downloads": [{"filepath": str(downloaded)}]
                }
                downloads.append(ydl_instance)

            with (
                patch("imageio_ffmpeg.get_ffmpeg_exe", return_value="bundled-ffmpeg"),
                patch("yt_dlp.YoutubeDL", side_effect=downloads) as ydl,
            ):
                download_audio("https://youtu.be/video", "video", root)
                download_audio("https://youtu.be/video", "video", root)

            self.assertEqual(ydl.call_count, 2)
            for call in ydl.call_args_list:
                self.assertEqual(call.args[0]["ffmpeg_location"], "bundled-ffmpeg")

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

    def test_live_snapshot_downloads_from_start(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            downloaded = root / "video.mp4"
            downloaded.write_bytes(b"audio")
            leftover = root / "video.f140.mp4.part-Frag9.part"
            leftover.write_bytes(b"partial")
            instance = MagicMock()
            instance.__enter__.return_value.extract_info.return_value = {
                "requested_downloads": [{"filepath": str(downloaded)}]
            }
            with patch("yt_dlp.YoutubeDL", return_value=instance) as ydl:
                result = download_audio(
                    "https://youtu.be/video", "video", root, live_snapshot=True
                )

            self.assertEqual(result, downloaded.resolve())
            self.assertTrue(ydl.call_args.args[0]["live_from_start"])
            self.assertFalse(leftover.exists())

    def test_live_snapshot_stops_at_the_live_edge_seen_at_start(self) -> None:
        updates: list[tuple[float, str]] = []
        hook = _progress_hook(
            lambda ratio, detail: updates.append((ratio, detail)),
            None,
            "오디오",
            live_snapshot=True,
        )
        base = {"status": "downloading", "filename": "a.mp4"}
        hook({**base, "fragment_index": 1, "fragment_count": 4})
        # The broadcast keeps growing, but the stop point must not move.
        hook({**base, "fragment_index": 4, "fragment_count": 9})
        self.assertEqual(updates[-1][0], 1.0)
        with self.assertRaises(KeyboardInterrupt):
            hook({**base, "fragment_index": 5, "fragment_count": 9})



class LocalMediaTests(unittest.TestCase):
    def test_resolves_audio_file_to_local_video_info(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "recording.wav"
            with wave.open(str(source), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16_000)
                output.writeframes(b"\x00\x00" * 16_000)
            media = resolve_local_media(str(source))
            self.assertFalse(media.has_video)
            self.assertEqual(media.video.title, "recording")
            self.assertAlmostEqual(media.video.duration, 1.0, places=2)
            self.assertTrue(media.video.video_id.startswith("local_"))

    def test_copy_uses_a_job_folder_copy_without_touching_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "recording.m4a"
            source.write_bytes(b"original audio")
            media = LocalMedia(
                path=source,
                video=VideoInfo("local_abc", "recording", "내 컴퓨터의 파일", 3, str(source)),
                has_video=False,
            )
            target = copy_local_media(media, root / "job")
            self.assertEqual(target.name, "original_media.m4a")
            self.assertEqual(target.read_bytes(), b"original audio")
            self.assertTrue(source.exists())


class FormattingTests(unittest.TestCase):
    def test_duration(self) -> None:
        self.assertEqual(format_duration(0), "00:00:00")
        self.assertEqual(format_duration(3661.9), "01:01:01")

    def test_safe_windows_filename(self) -> None:
        self.assertEqual(safe_filename('A: B / C? "D"'), "A B C D")


class MlxTranscriptionTests(unittest.TestCase):
    def test_resolves_legacy_model_alias(self) -> None:
        with patch.dict(
            "lecture_scribe.transcription.LOCAL_MLX_MODELS", {}, clear=True
        ):
            self.assertEqual(
                resolve_mlx_model("large-v3"),
                "mlx-community/whisper-large-v3-mlx",
            )
        self.assertEqual(resolve_mlx_model("/tmp/local-model"), "/tmp/local-model")

    def test_prefers_complete_project_local_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            (model_dir / "config.json").touch()
            (model_dir / "weights.npz").touch()
            with patch.dict(
                "lecture_scribe.transcription.LOCAL_MLX_MODELS",
                {"large-v3": model_dir},
                clear=True,
            ):
                self.assertEqual(resolve_mlx_model("large-v3"), str(model_dir))

    def test_converts_mlx_dictionary_segments_to_output_files(self) -> None:
        fake_module = SimpleNamespace(
            transcribe=MagicMock(
                return_value={
                    "language": "ko",
                    "text": "안녕하세요",
                    "segments": [
                        {
                            "start": 1.0,
                            "end": 2.5,
                            "text": " 안녕하세요 ",
                            "avg_logprob": -0.1,
                            "no_speech_prob": 0.02,
                            "compression_ratio": 1.1,
                        }
                    ],
                }
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "edited.flac"
            audio.touch()
            with (
                patch.dict(sys.modules, {"mlx_whisper": fake_module}),
                patch.dict(
                    "lecture_scribe.transcription.LOCAL_MLX_MODELS", {}, clear=True
                ),
                patch("lecture_scribe.transcription.runtime_backend", return_value="mlx"),
                patch("lecture_scribe.transcription._audio_duration", return_value=3.0),
            ):
                result = transcribe_audio(
                    audio,
                    root,
                    VideoInfo("x", "제목", "채널", 3, "https://youtu.be/x"),
                    prompt_terms=("역전파", "backpropagation"),
                )
            self.assertEqual(result.language, "ko")
            self.assertEqual(result.segments[0].text, "안녕하세요")
            self.assertIn("안녕하세요", result.text_path.read_text(encoding="utf-8"))
            self.assertEqual(
                fake_module.transcribe.call_args.kwargs["path_or_hf_repo"],
                "mlx-community/whisper-large-v3-mlx",
            )
            self.assertNotIn("beam_size", fake_module.transcribe.call_args.kwargs)
            self.assertIn(
                "역전파",
                fake_module.transcribe.call_args.kwargs["initial_prompt"],
            )

    def test_builds_compact_deduplicated_initial_prompt(self) -> None:
        prompt = _build_initial_prompt(("PyTorch", " pytorch ", "역전파"), limit=40)
        self.assertEqual(prompt, "이 강의의 주요 전문용어: PyTorch, 역전파")


class CudaTranscriptionTests(unittest.TestCase):
    def test_applies_beam_and_terminology_to_faster_whisper(self) -> None:
        raw = SimpleNamespace(
            start=0.5,
            end=1.5,
            text=" CUDA 전사 ",
            avg_logprob=-0.1,
            no_speech_prob=0.01,
            compression_ratio=1.0,
        )
        info = SimpleNamespace(duration=2.0, language="ko", language_probability=0.98)
        model = MagicMock()
        model.transcribe.return_value = (iter([raw]), info)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "edited.flac"
            audio.touch()
            with (
                patch("lecture_scribe.transcription.runtime_backend", return_value="cuda"),
                patch("lecture_scribe.transcription._get_cuda_model", return_value=model),
            ):
                result = transcribe_audio(
                    audio,
                    root,
                    VideoInfo("x", "제목", "채널", 2, "https://youtu.be/x"),
                    beam_size=7,
                    prompt_terms=("CUDA", "CTranslate2"),
                )
        self.assertEqual(result.language_probability, 0.98)
        self.assertEqual(model.transcribe.call_args.kwargs["beam_size"], 7)
        self.assertIn("CTranslate2", model.transcribe.call_args.kwargs["initial_prompt"])


class TerminologyTests(unittest.TestCase):
    def test_parses_numbered_terms_and_removes_duplicates(self) -> None:
        terms = _parse_transcription_terms(
            "```text\n1. Transformer\n- 어텐션\ntransformer\n```",
            max_terms=10,
        )
        self.assertEqual(terms, ("Transformer", "어텐션"))

    def test_reuses_saved_terms_for_the_same_lecture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            terms_path = root / "transcription_terms.txt"
            terms_path.write_text("Transformer\n어텐션\n", encoding="utf-8")
            settings = AppSettings(
                auto_terminology=True,
                terminology_lecture_name="딥러닝 자연어 처리",
            )
            state = {
                "artifacts": {"transcription_terms": terms_path.name},
                "terminology": {"lecture_name": "딥러닝 자연어 처리", "count": 2},
            }
            with patch("lecture_scribe.pipeline.generate_transcription_terms") as generate:
                terms = _prepare_transcription_terms(
                    root,
                    state,
                    settings,
                    VideoInfo("x", "영상", "채널", 10, "https://youtu.be/x"),
                )
            self.assertEqual(terms, ("Transformer", "어텐션"))
            generate.assert_not_called()

    def test_generates_and_saves_terms_for_a_new_lecture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = AppSettings(
                auto_terminology=True,
                terminology_lecture_name="운영체제와 스케줄링",
                terminology_model="gpt-5.6-luna",
            )
            state = {"artifacts": {}}
            generated = ("운영체제", "CPU 스케줄링", "라운드 로빈")
            with patch(
                "lecture_scribe.pipeline.generate_transcription_terms",
                return_value=generated,
            ) as generate:
                terms = _prepare_transcription_terms(
                    root,
                    state,
                    settings,
                    VideoInfo("x", "프로세스 스케줄링", "채널", 10, "https://youtu.be/x"),
                )

            self.assertEqual(terms, generated)
            self.assertEqual(
                (root / "transcription_terms.txt").read_text(encoding="utf-8"),
                "운영체제\nCPU 스케줄링\n라운드 로빈\n",
            )
            self.assertEqual(
                state["artifacts"]["transcription_terms"],
                "transcription_terms.txt",
            )
            generate.assert_called_once()
            self.assertEqual(generate.call_args.kwargs["model"], "gpt-5.6-luna")
            self.assertEqual(state["terminology"]["model"], "gpt-5.6-luna")

    def test_uses_the_selected_claude_models_for_terms(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = AppSettings(
                auto_terminology=True,
                terminology_lecture_name="운영체제",
                ai_provider="claude",
                claude_terminology_model="sonnet",
            )
            state = {"artifacts": {}}
            with patch(
                "lecture_scribe.pipeline.generate_transcription_terms",
                return_value=("운영체제",),
            ) as generate:
                _prepare_transcription_terms(
                    Path(directory),
                    state,
                    settings,
                    VideoInfo("x", "영상", "채널", 10, "https://youtu.be/x"),
                )
            self.assertEqual(generate.call_args.kwargs["provider"], "claude")
            self.assertEqual(generate.call_args.kwargs["model"], "sonnet")
            self.assertEqual(state["terminology"]["provider"], "claude")


class ModelCatalogTests(unittest.TestCase):
    def test_parses_listed_codex_models_in_priority_order(self) -> None:
        models = _parse_codex_models(
            [
                {
                    "slug": "gpt-b",
                    "display_name": "GPT-B",
                    "visibility": "list",
                    "priority": 5,
                    "default_reasoning_level": "medium",
                    "supported_reasoning_levels": [{"effort": "low"}, {"effort": "medium"}],
                },
                {"slug": "hidden", "visibility": "hide", "priority": 1},
                {"slug": "gpt-a", "display_name": "GPT-A", "visibility": "list", "priority": 1},
            ]
        )
        self.assertEqual([model.value for model in models], ["gpt-a", "gpt-b"])
        self.assertEqual(models[1].efforts, ("low", "medium"))
        self.assertEqual(models[1].default_effort, "medium")

    def test_parses_claude_models_without_the_default_alias(self) -> None:
        models = _parse_claude_models(
            [
                {"value": "default", "displayName": "Default", "description": "Use Opus"},
                {
                    "value": "opus",
                    "displayName": "Opus",
                    "description": "Opus 5.5 · Best for everyday, complex tasks · $4/$20 per Mtok",
                    "supportedEffortLevels": ["low", "medium", "high", "xhigh", "max"],
                },
                {"value": "haiku", "displayName": "Haiku", "description": "Haiku 4.5 · Fastest"},
            ]
        )
        self.assertEqual([model.value for model in models], ["opus", "haiku"])
        self.assertEqual(models[0].label, "Opus 5.5")
        self.assertEqual(models[0].default_effort, "high")
        self.assertEqual(models[1].efforts, ())

    def test_falls_back_to_built_in_models_when_the_cli_fails(self) -> None:
        for provider, client in ((OPENAI, "codex_client"), (CLAUDE, "claude_client")):
            with patch(
                f"lecture_scribe.ai_providers.{client}.fetch_model_catalog",
                side_effect=FileNotFoundError("CLI 없음"),
            ):
                catalog = fetch_catalog(provider)
            self.assertFalse(catalog.live)
            self.assertIn("CLI 없음", catalog.detail)
            self.assertTrue(catalog.models)


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
    def test_selects_platform_specific_hardware_encoder(self) -> None:
        with (
            patch("lecture_scribe.video_edit.platform.system", return_value="Windows"),
            patch("lecture_scribe.video_edit._ffmpeg_nvenc_usable", return_value=True),
        ):
            windows_args, windows_label = _select_hardware_encoder("ffmpeg")
        with (
            patch("lecture_scribe.video_edit.platform.system", return_value="Darwin"),
            patch(
                "lecture_scribe.video_edit._ffmpeg_videotoolbox_usable",
                return_value=True,
            ),
        ):
            mac_args, mac_label = _select_hardware_encoder("ffmpeg")
        self.assertIn("h264_nvenc", windows_args)
        self.assertIn("NVIDIA NVENC", windows_label)
        self.assertIn("h264_videotoolbox", mac_args)
        self.assertIn("Apple VideoToolbox", mac_label)

    def test_builds_synchronized_video_and_audio_concat_filter(self) -> None:
        graph = build_video_filter_graph(((10, 20), (40.5, 60)))
        self.assertIn("[0:v]trim=start=10.000000:end=20.000000", graph)
        self.assertIn("[0:a]atrim=start=40.500000:end=60.000000", graph)
        self.assertIn("[v0][a0][v1][a1]concat=n=2:v=1:a=1[outv][outa]", graph)

    def test_renders_video_and_audio_with_removed_middle_range(self) -> None:
        import av
        import imageio_ffmpeg

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        try:
            _, expected_encoder = _select_hardware_encoder(ffmpeg_exe)
        except RuntimeError:
            self.skipTest("Platform hardware video encoder is not currently available")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            output = root / "edited.mp4"
            messages: list[str] = []
            subprocess.run(
                [
                    ffmpeg_exe,
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
                    "mpeg4",
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
            self.assertTrue(any(expected_encoder in item for item in messages))

    def test_does_not_fall_back_to_cpu_when_hardware_encoder_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            source.write_bytes(b"source")
            with (
                patch(
                    "lecture_scribe.video_edit._select_hardware_encoder",
                    side_effect=RuntimeError("CPU 인코더로 대체하지 않았습니다"),
                ),
                patch("lecture_scribe.video_edit._run_ffmpeg") as run_ffmpeg,
                self.assertRaisesRegex(RuntimeError, "CPU 인코더로 대체하지 않았습니다"),
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
                beam_size=7,
                codex_model="gpt-5.6-sol",
                terminology_model="gpt-5.6-luna",
                keep_video=True,
                auto_terminology=True,
                terminology_lecture_name="운영체제",
            )
            settings.save(path)
            loaded = AppSettings.load(path)
            self.assertEqual(loaded.beam_size, 7)
            self.assertEqual(loaded.codex_model, "gpt-5.6-sol")
            self.assertEqual(loaded.terminology_model, "gpt-5.6-luna")
            self.assertTrue(loaded.keep_video)
            self.assertTrue(loaded.auto_terminology)
            self.assertEqual(loaded.terminology_lecture_name, "운영체제")

    def test_keeps_a_model_selection_per_provider(self) -> None:
        settings = AppSettings()
        settings.set_ai_selection("claude", "opus", "xhigh", "haiku")
        settings.set_ai_selection("openai", "gpt-5.6-sol", "high", "gpt-5.6-luna")
        self.assertEqual(settings.ai_selection("claude"), ("opus", "xhigh", "haiku"))
        settings.ai_provider = "claude"
        self.assertEqual(settings.ai_selection(), ("opus", "xhigh", "haiku"))
        settings.ai_provider = "openai"
        self.assertEqual(settings.ai_selection(), ("gpt-5.6-sol", "high", "gpt-5.6-luna"))

    def test_prompt_slots_persist_and_switch_active_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy_prompt = root / "legacy.md"
            legacy_prompt.write_text("기본 {{transcript}}" + " 내용" * 20, encoding="utf-8")
            settings_path = root / "settings.json"
            settings = AppSettings(prompt_path=str(legacy_prompt))
            settings.ensure_prompt_slots()
            settings.add_prompt_slot("주식방송용", "주식 {{transcript}}" + " 분석" * 20)
            save_prompt(settings, settings.active_prompt_slot()["content"])
            settings.save(settings_path)

            loaded = AppSettings.load(settings_path)
            self.assertEqual(len(loaded.prompt_slots), 2)
            self.assertEqual(loaded.active_prompt_slot()["name"], "주식방송용")
            self.assertIn("주식", load_prompt(loaded))
            self.assertTrue(loaded.remove_active_prompt_slot())
            self.assertEqual(loaded.active_prompt_slot()["name"], "강의용 기본")


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


class LiveCaptionTests(unittest.TestCase):
    @staticmethod
    def _tone(seconds: float, level: float = 0.3):
        import numpy as np

        count = int(SAMPLE_RATE * seconds)
        return (level * np.sin(np.arange(count) * 2 * np.pi * 220 / SAMPLE_RATE)).astype("float32")

    @staticmethod
    def _silence(seconds: float):
        import numpy as np

        return np.zeros(int(SAMPLE_RATE * seconds), dtype="float32")

    def test_ffmpeg_command_pulls_network_audio_without_loopback(self) -> None:
        live = LiveSource("t", "https://example.com/a.m3u8", True, {"User-Agent": "UA"})
        command = build_ffmpeg_command("ffmpeg", live)
        self.assertIn("-headers", command)
        self.assertNotIn("-re", command)
        self.assertEqual(command[command.index("-ar") + 1], "16000")
        recording = LiveSource("t", "/tmp/a.mp4", False)
        command = build_ffmpeg_command("ffmpeg", recording)
        self.assertIn("-re", command)
        self.assertNotIn("-reconnect", command)

    def test_resolve_rejects_missing_input(self) -> None:
        with self.assertRaises(ValueError):
            resolve_live_source("")
        with self.assertRaises(ValueError):
            resolve_live_source("not-a-link-or-file")
        direct = resolve_live_source("https://cdn.example.com/live/index.m3u8?token=1")
        self.assertTrue(direct.is_live)

    def test_silence_helpers(self) -> None:
        import numpy as np

        audio = np.concatenate([self._tone(2.0), self._silence(0.8)])
        self.assertTrue(is_silent(self._silence(3)))
        self.assertFalse(is_silent(audio))
        self.assertAlmostEqual(trailing_quiet_seconds(audio), 0.8, places=1)
        gap = np.concatenate([self._tone(4.0), self._silence(0.5), self._tone(4.0)])
        cut = quietest_cut(gap, search_seconds=6)
        self.assertTrue(4.0 * SAMPLE_RATE <= cut <= 4.5 * SAMPLE_RATE)

    def test_filters_hallucinated_and_repeated_lines(self) -> None:
        self.assertFalse(keep_segment(TranscriptSegment(0, 1, "시청해주셔서 감사합니다.")))
        self.assertFalse(keep_segment(TranscriptSegment(0, 1, "같은 문장"), "같은 문장."))
        self.assertFalse(
            keep_segment(TranscriptSegment(0, 1, "음", avg_logprob=-1.5, no_speech_probability=0.9))
        )
        self.assertTrue(keep_segment(TranscriptSegment(0, 1, "적분인자를 곱합니다")))

    def test_captioner_finalizes_on_pause_and_offsets_times(self) -> None:
        import numpy as np

        calls: list[float] = []

        def fake_transcribe(samples, **_kwargs):
            calls.append(len(samples) / SAMPLE_RATE)
            return [TranscriptSegment(0.5, len(samples) / SAMPLE_RATE - 0.5, f"문장 {len(calls)}")]

        captions: list = []
        partials: list[str] = []
        captioner = LiveCaptioner(
            LiveCaptionOptions(), captions.append, partials.append, transcribe=fake_transcribe
        )
        stream = np.concatenate([self._tone(3.5), self._silence(0.7), self._tone(14.0)])
        for start in range(0, len(stream), SAMPLE_RATE // 10):
            captioner.feed(stream[start:start + SAMPLE_RATE // 10])
        captioner.flush()
        self.assertGreaterEqual(len(captions), 3)
        self.assertAlmostEqual(captions[0].start, 0.5, places=1)
        # The second sentence starts after the first pause, not at zero.
        self.assertGreater(captions[1].start, 3.5)
        self.assertTrue(any(partials))
        self.assertTrue(all(duration <= 12.01 for duration in calls))


class ModelDownloadTests(unittest.TestCase):
    def test_message_shows_percent_speed_and_eta(self) -> None:
        progress = model_download.DownloadProgress(
            "Whisper large-v3-turbo 모델", 400 * 1024**2, 1600 * 1024**2, 20 * 1024**2, 0.0
        )
        self.assertAlmostEqual(progress.fraction or 0, 0.25)
        for expected in ("25%", "400.0 MB / 1.6 GB", "20.0 MB/초", "남은 시간 약 1분 0초"):
            self.assertIn(expected, progress.message)

    def test_message_flags_a_stalled_download(self) -> None:
        stalled = model_download.DownloadProgress("모델", 10, 100, 0.0, 30.0)
        self.assertIn("30초째 진행이 없습니다", stalled.message)
        unknown = model_download.DownloadProgress("모델", 2048, None, 0.0, 0.0)
        self.assertIsNone(unknown.fraction)
        self.assertIn("2.0 KB 받음", unknown.message)

    def test_download_reports_growth_and_surfaces_failures(self) -> None:
        import time

        repo = model_download.ModelRepo("org/model", "테스트 모델")
        with tempfile.TemporaryDirectory() as folder:
            blobs = Path(folder)

            def fake_download(*_args, **_kwargs):
                for _ in range(4):
                    with (blobs / "x.incomplete").open("ab") as handle:
                        handle.write(b"a" * 250)
                    time.sleep(0.15)

            seen: list[model_download.DownloadProgress] = []
            with (
                patch.object(model_download, "is_cached", return_value=False),
                patch.object(model_download, "remote_size", return_value=1000),
                patch.object(model_download, "_blob_dir", return_value=blobs),
                patch.object(model_download, "POLL_SECONDS", 0.05),
                patch("huggingface_hub.snapshot_download", fake_download),
            ):
                self.assertTrue(model_download.download_repo(repo, seen.append))
            self.assertEqual(seen[-1].fraction, 1.0)
            self.assertTrue(any(0 < (item.fraction or 0) < 1 for item in seen))

            def broken_download(*_args, **_kwargs):
                raise OSError("offline")

            with (
                patch.object(model_download, "is_cached", return_value=False),
                patch.object(model_download, "remote_size", return_value=None),
                patch.object(model_download, "_blob_dir", return_value=blobs),
                patch.object(model_download, "POLL_SECONDS", 0.05),
                patch("huggingface_hub.snapshot_download", broken_download),
            ):
                with self.assertRaisesRegex(RuntimeError, "offline"):
                    model_download.download_repo(repo, seen.append)

    def test_cancel_stops_waiting_for_the_download(self) -> None:
        import time
        from threading import Event

        repo = model_download.ModelRepo("org/model", "테스트 모델")
        cancel = Event()
        cancel.set()
        with tempfile.TemporaryDirectory() as folder:
            with (
                patch.object(model_download, "is_cached", return_value=False),
                patch.object(model_download, "remote_size", return_value=None),
                patch.object(model_download, "_blob_dir", return_value=Path(folder)),
                patch("huggingface_hub.snapshot_download", lambda *a, **k: time.sleep(0.5)),
            ):
                with self.assertRaises(model_download.DownloadCancelled):
                    model_download.download_repo(repo, lambda _p: None, cancel)

    def test_cached_model_skips_download(self) -> None:
        repo = model_download.ModelRepo("org/model", "테스트 모델")
        with patch.object(model_download, "is_cached", return_value=True):
            self.assertFalse(model_download.download_repo(repo, lambda _p: self.fail("no progress")))


class QwenEngineTests(unittest.TestCase):
    def test_model_names_and_context(self) -> None:
        self.assertTrue(is_qwen_model("qwen3-asr-1.7b"))
        self.assertFalse(is_qwen_model("large-v3"))
        self.assertIn("미분방정식", context_from_terms(("미분방정식", " 고유값 ")))
        self.assertEqual(context_from_terms(()), "")

    def test_groups_aligned_korean_words_with_original_spacing(self) -> None:
        text = "안녕하세요. 오늘은 미분을 배웁니다 다음은 적분입니다."
        words = [
            AlignedWord("안녕하세요", 0.0, 0.8),
            AlignedWord("오늘은", 1.0, 1.4),
            AlignedWord("미분을", 1.5, 1.9),
            AlignedWord("배웁니다", 2.0, 2.6),
            AlignedWord("다음은", 4.0, 4.4),
            AlignedWord("적분입니다", 4.5, 5.2),
        ]
        segments = group_words(text, words)
        self.assertEqual(
            segments,
            [
                (0.0, 0.8, "안녕하세요."),
                (1.0, 2.6, "오늘은 미분을 배웁니다"),
                (4.0, 5.2, "다음은 적분입니다."),
            ],
        )

    def test_groups_long_run_and_unmatched_words(self) -> None:
        words = [AlignedWord(f"w{index}", index * 1.0, index * 1.0 + 0.9) for index in range(30)]
        segments = group_words(" ".join(item.text for item in words), words, max_seconds=12)
        self.assertGreater(len(segments), 1)
        self.assertTrue(all(end - start <= 13 for start, end, _ in segments))
        fallback = group_words("완전히 다른 글", [AlignedWord("없는말", 0, 1)])
        self.assertEqual(fallback, [(0, 1, "없는말")])

    def test_split_pieces_covers_audio(self) -> None:
        import numpy as np

        samples = np.ones(16_000 * 300, dtype="float32") * 0.2
        samples[16_000 * 115: 16_000 * 116] = 0
        pieces = split_pieces(samples, piece_seconds=120)
        self.assertEqual(pieces[0][0], 0)
        self.assertEqual(pieces[-1][1], len(samples))
        self.assertTrue(115 * 16_000 <= pieces[0][1] <= 116 * 16_000)
        for (_, end), (start, _) in zip(pieces, pieces[1:]):
            self.assertEqual(end, start)

    def test_mlx_backend_maps_result(self) -> None:
        import numpy as np

        fake = MagicMock()
        fake.transcribe.return_value = SimpleNamespace(
            text="안녕하세요. 반갑습니다.",
            language="Korean",
            segments=[
                {"text": "안녕하세요", "start": 0.1, "end": 0.9},
                {"text": "반갑습니다", "start": 1.1, "end": 1.8},
            ],
        )
        with patch.dict(sys.modules, {"mlx_qwen3_asr": fake}), patch(
            "lecture_scribe.qwen_asr_engine.platform.system", return_value="Darwin"
        ), patch("lecture_scribe.qwen_asr_engine.platform.machine", return_value="arm64"), patch(
            "lecture_scribe.qwen_asr_engine._MLX_ALIGNER", []
        ):
            segments, language = transcribe_qwen(
                np.zeros(32_000, dtype="float32"), "qwen3-asr-1.7b", "ko", context="용어"
            )
        kwargs = fake.transcribe.call_args.kwargs
        self.assertEqual(kwargs["model"], "Qwen/Qwen3-ASR-1.7B")
        self.assertEqual(kwargs["language"], "Korean")
        self.assertTrue(kwargs["return_timestamps"])
        self.assertEqual(language, "ko")
        self.assertEqual(segments, [(0.1, 0.9, "안녕하세요."), (1.1, 1.8, "반갑습니다.")])

    def test_transcribe_audio_dispatches_to_qwen_with_clip_offsets(self) -> None:
        import numpy as np

        calls: list[int] = []

        def fake_qwen(samples, model_name, language, context, timestamps=True, log=None):
            calls.append(len(samples))
            return [(0.5, 1.5, f"구간 {len(calls)}")], "ko"

        video = VideoInfo("vid", "제목", "채널", 60, "https://youtu.be/vid")
        with tempfile.TemporaryDirectory() as directory, patch(
            "faster_whisper.audio.decode_audio",
            return_value=np.full(16_000 * 60, 0.2, dtype="float32"),
        ), patch("lecture_scribe.qwen_asr_engine.transcribe_qwen", side_effect=fake_qwen):
            result = transcribe_audio(
                Path(directory) / "audio.flac", Path(directory), video,
                model_name="qwen3-asr-1.7b", language="ko",
                clip_ranges=((10.0, 20.0), (30.0, 40.0)),
            )
        self.assertEqual(calls, [16_000 * 10, 16_000 * 10])
        self.assertEqual([item.start for item in result.segments], [10.5, 30.5])
        self.assertEqual(result.language, "ko")

    def test_live_samples_skip_aligner(self) -> None:
        import numpy as np

        with patch(
            "lecture_scribe.transcription.transcribe_qwen",
            return_value=([(0.0, 2.0, "실시간 문장")], "ko"),
        ) as fake:
            segments = transcribe_samples(np.zeros(32_000, dtype="float32"), "qwen3-asr-1.7b", "ko")
        self.assertFalse(fake.call_args.kwargs["timestamps"])
        self.assertEqual(segments[0].text, "실시간 문장")


if __name__ == "__main__":
    unittest.main()
