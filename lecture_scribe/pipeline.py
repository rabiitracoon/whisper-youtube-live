from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from threading import Event
from time import sleep

from .audio_edit import (
    AudioEditRequest,
    build_included_ranges,
    extract_waveform,
    render_edited_audio,
)
from .ai_providers import (
    PROVIDER_NAMES,
    auth_status,
    generate_notes,
    generate_transcription_terms,
    normalize_provider,
)
from .config import AppSettings, load_prompt
from .local_media import LocalMediaCancelled, copy_local_media, resolve_local_media
from .paths import safe_filename
from .prompting import PromptContext, render_prompt
from .quality import TranscriptReview, assess_transcript
from .time_range import validate_clip_range
from .transcription import TranscriptResult, transcribe_audio, verify_accelerator
from .video_edit import render_edited_video
from .youtube import (
    VideoInfo,
    download_audio,
    download_video,
    fetch_video_info,
    format_duration,
    is_youtube_url,
)

ProgressCallback = Callable[[int, str, str], None]
LogCallback = Callable[[str], None]
ReviewCallback = Callable[[TranscriptReview], bool]
AudioEditCallback = Callable[[AudioEditRequest], tuple[tuple[float, float], ...] | None]
PIPELINE_STAGES = ("download", "edit", "transcribe", "notes")


class PipelineCancelled(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PipelineResult:
    output_dir: Path
    notes_path: Path | None
    transcript_path: Path | None
    srt_path: Path | None
    video_path: Path | None
    video_title: str
    completed_stage: str


def run_pipeline(
    source: str,
    settings: AppSettings,
    progress: ProgressCallback | None = None,
    log: LogCallback | None = None,
    cancel_event: Event | None = None,
    edit_audio: AudioEditCallback | None = None,
    review_transcript: ReviewCallback | None = None,
    target_stage: str = "all",
) -> PipelineResult:
    if target_stage not in {*PIPELINE_STAGES, "all"}:
        raise ValueError(f"지원하지 않는 작업 단계입니다: {target_stage}")

    def emit(percent: int, stage: str, detail: str) -> None:
        if progress:
            progress(max(0, min(100, percent)), stage, detail)

    needs_transcriber = target_stage in {"all", "transcribe"}
    needs_auth = target_stage in {"all", "notes"} or (
        needs_transcriber and settings.auto_terminology
    )
    emit(1, "사전 점검", "현재 단계에 필요한 환경을 확인합니다.")
    if needs_auth:
        provider = normalize_provider(settings.ai_provider)
        status = auth_status(provider)
        if not status.available:
            raise RuntimeError(status.detail)
        if not status.logged_in:
            raise RuntimeError(
                f"{PROVIDER_NAMES[provider]} OAuth 로그인이 필요합니다. "
                "연결과 업데이트 화면에서 로그인해주세요."
            )
    if needs_transcriber:
        accelerator_name = verify_accelerator()
        if log:
            log(f"전사 가속기 확인 완료: {accelerator_name}")

    source = source.strip()
    local_media = None
    if is_youtube_url(source):
        emit(3, "영상 확인", "YouTube 영상 정보를 읽는 중입니다.")
        video = fetch_video_info(source, settings.cookie_browser, log)
    else:
        emit(3, "파일 확인", "선택한 영상 또는 음성 파일을 읽는 중입니다.")
        local_media = resolve_local_media(source)
        video = local_media.video
        if settings.keep_video and not local_media.has_video:
            raise ValueError("음성 파일에는 ‘쉬는 시간을 뺀 영상도 보관하기’를 사용할 수 없습니다.")
    clip_start, clip_end = _resolve_clip_range(
        settings.clip_start_seconds, settings.clip_end_seconds, video, log
    )
    job_dir, state = _resolve_job(
        Path(settings.output_dir), video, target_stage, log
    )
    _ensure_state(state, video, settings)
    _save_checkpoint(job_dir, state)
    if log:
        completed = ", ".join(state["completed_stages"]) or "없음"
        log(f"작업 폴더: {job_dir}")
        log(f"재개 가능한 완료 단계: {completed}")

    prompt_terms: tuple[str, ...] = ()
    if (
        settings.auto_terminology
        and needs_transcriber
        and (target_stage == "transcribe" or "transcribe" not in state["completed_stages"])
    ):
        emit(4, "전문용어 준비", "강의명으로 전사 전문용어를 만들고 있습니다.")
        prompt_terms = _prepare_transcription_terms(
            job_dir,
            state,
            settings,
            video,
            log=log,
            cancel_event=cancel_event,
        )

    audio_path = _state_artifact(job_dir, state, "original_audio")
    video_path = _state_artifact(job_dir, state, "original_video")
    should_download = target_stage == "download" or (
        target_stage == "all"
        and (
            "download" not in state["completed_stages"]
            or (settings.keep_video and (not video_path or not video_path.exists()))
        )
    )
    if should_download:
        if audio_path and audio_path.exists():
            if log:
                log("이미 저장된 원본 오디오를 재사용합니다. 다시 다운로드하지 않습니다.")
        else:
            if local_media is not None:
                emit(5, "파일 가져오기", "선택한 파일을 작업 폴더로 안전하게 복사합니다.")
                audio_path = copy_local_media(local_media, job_dir, cancel_event=cancel_event)
            else:
                emit(5, "원본 다운로드", "최고 품질 오디오 스트림을 저장합니다.")
                audio_path = download_audio(
                    video.webpage_url,
                    video.video_id,
                    job_dir,
                    settings.cookie_browser,
                    progress=lambda ratio, detail: emit(
                        5 + round(ratio * (7 if settings.keep_video else 15)),
                        "원본 다운로드",
                        detail,
                    ),
                    log=log,
                    cancel_event=cancel_event,
                )
        artifacts = {"original_audio": audio_path.name}
        if settings.keep_video:
            if video_path and video_path.exists():
                if log:
                    log("이미 저장된 원본 영상을 재사용합니다. 다시 다운로드하지 않습니다.")
            else:
                if local_media is not None:
                    video_path = audio_path
                else:
                    emit(12, "원본 영상 다운로드", "최대 1080p 영상과 오디오를 저장합니다.")
                    video_path = download_video(
                        video.webpage_url,
                        video.video_id,
                        job_dir,
                        settings.cookie_browser,
                        progress=lambda ratio, detail: emit(
                            12 + round(ratio * 8), "원본 영상 다운로드", detail
                        ),
                        log=log,
                        cancel_event=cancel_event,
                    )
            artifacts["original_video"] = video_path.name
        _mark_stage(
            job_dir,
            state,
            "download",
            artifacts=artifacts,
        )
    elif not audio_path or not audio_path.exists():
        raise RuntimeError(
            "저장된 원본 오디오가 없습니다. 고급 모드의 ‘1. 원본 다운로드’를 먼저 실행해주세요."
        )
    if target_stage == "download":
        emit(100, "원본 다운로드 완료", "다음에 편집 단계부터 이어서 할 수 있습니다.")
        return _make_result(job_dir, video, state, "download")

    edited_path = _state_artifact(job_dir, state, "edited_audio")
    edited_video_path = _state_artifact(job_dir, state, "edited_video")
    should_edit = target_stage == "edit" or (
        target_stage == "all"
        and (
            "edit" not in state["completed_stages"]
            or not edited_path
            or not edited_path.exists()
            or (
                settings.keep_video
                and (not edited_video_path or not edited_video_path.exists())
            )
        )
    )
    if should_edit:
        emit(20, "파형 준비", "저장된 원본에서 파형을 생성합니다.")
        audio_duration, peaks = extract_waveform(
            audio_path,
            video.duration,
            cancel_event=cancel_event,
            progress=lambda ratio: emit(
                20 + round(ratio * 8),
                "파형 준비",
                f"오디오 파형 분석 중 · {ratio:.0%}",
            ),
        )
        edit_end = min(clip_end, audio_duration) if clip_end is not None else audio_duration
        if edit_end <= clip_start:
            raise ValueError("파형에서 편집할 강의 구간이 비어 있습니다.")
        request = AudioEditRequest(
            audio_path=audio_path.resolve(),
            duration=audio_duration,
            peaks=peaks,
            range_start=clip_start,
            range_end=edit_end,
        )
        exclusions: tuple[tuple[float, float], ...] = ()
        if edit_audio is not None:
            emit(28, "Razor 편집", "Razor로 쉬는 시간 클립을 잘라 제외해주세요.")
            edited = edit_audio(request)
            if edited is None:
                raise PipelineCancelled(
                    "편집을 저장하지 않고 중단했습니다. 원본 다운로드는 보존되어 다음에 편집부터 계속할 수 있습니다."
                )
            exclusions = edited
        included_ranges = build_included_ranges(clip_start, edit_end, exclusions)
        edited_path = job_dir / "edited_lecture.flac"
        emit(30, "편집본 저장", "쉬는 시간을 제거한 무손실 음성 파일을 만듭니다.")
        render_edited_audio(
            audio_path,
            edited_path,
            included_ranges,
            audio_duration,
            cancel_event=cancel_event,
            progress=lambda ratio: emit(
                30 + round(ratio * 15),
                "편집본 저장",
                f"16 kHz mono FLAC 저장 중 · {ratio:.0%}",
            ),
        )
        edit_artifacts = {"edited_audio": edited_path.name}
        if settings.keep_video:
            video_path = _state_artifact(job_dir, state, "original_video")
            if not video_path or not video_path.exists():
                raise RuntimeError(
                    "저장된 원본 영상이 없습니다. ‘쉬는 시간을 뺀 영상도 보관하기’를 켠 상태로 "
                    "고급 모드의 ‘1. 원본 다운로드’를 먼저 실행해주세요."
                )
            edited_video_path = job_dir / "edited_lecture.mp4"
            emit(38, "영상 편집본 저장", "쉬는 시간을 제거한 MP4 영상을 만듭니다.")
            render_edited_video(
                video_path,
                edited_video_path,
                included_ranges,
                cancel_event=cancel_event,
                progress=lambda ratio: emit(
                    38 + round(ratio * 7),
                    "영상 편집본 저장",
                    f"H.264 MP4 저장 중 · {ratio:.0%}",
                ),
                log=log,
            )
            edit_artifacts["edited_video"] = edited_video_path.name
        state["edit"] = {
            "excluded_ranges": [list(item) for item in exclusions],
            "included_ranges": [list(item) for item in included_ranges],
            "edited_duration": sum(end - start for start, end in included_ranges),
        }
        _invalidate_after(state, "edit")
        _mark_stage(
            job_dir,
            state,
            "edit",
            artifacts=edit_artifacts,
        )
        _write_metadata(
            job_dir,
            video,
            settings,
            clip_start,
            clip_end,
            exclusions,
            included_ranges,
            original_video=video_path if settings.keep_video else None,
            edited_video=edited_video_path if settings.keep_video else None,
        )
        if log:
            removed = sum(end - start for start, end in exclusions)
            log(
                f"편집본 저장 완료: 쉬는 시간 {len(exclusions)}개, "
                f"{format_duration(removed)} 제거 · {edited_path.name}"
            )
            if edited_video_path and edited_video_path.exists():
                log(f"영상 편집본 저장 완료: {edited_video_path.name}")
    elif not edited_path or not edited_path.exists():
        raise RuntimeError(
            "저장된 편집본이 없습니다. 고급 모드의 ‘2. Razor 편집본 저장’을 먼저 실행해주세요."
        )
    if target_stage == "edit":
        emit(100, "편집 완료", "편집본을 저장했습니다. 전사 단계부터 이어서 할 수 있습니다.")
        return _make_result(job_dir, video, state, "edit")

    transcript: TranscriptResult | None = None
    should_transcribe = target_stage == "transcribe" or (
        target_stage == "all" and "transcribe" not in state["completed_stages"]
    )
    if should_transcribe:
        included_ranges = tuple(
            (float(start), float(end)) for start, end in state["edit"]["included_ranges"]
        )
        emit(46, "GPU 전사", "저장된 편집본을 이 컴퓨터의 Whisper 가속기에 전달합니다.")
        transcript = transcribe_audio(
            audio_path=edited_path,
            output_dir=job_dir,
            video=video,
            model_name=settings.whisper_model,
            language=settings.language,
            compute_type=settings.compute_type,
            beam_size=settings.beam_size,
            vad_filter=True,
            original_ranges=included_ranges,
            prompt_terms=prompt_terms,
            progress=lambda ratio, detail: emit(
                46 + round(ratio * 34), "GPU 전사", detail
            ),
            log=log,
            cancel_event=cancel_event,
        )
        emit(81, "전사 확인", "반복·언어·분량을 점검했습니다. 전사문을 확인해주세요.")
        review = assess_transcript(transcript)
        _write_review(job_dir, review)
        if review_transcript is not None and not review_transcript(review):
            raise PipelineCancelled(
                "전사 확인 단계에서 중단했습니다. 편집본과 전사본은 보존되어 다시 확인할 수 있습니다."
            )
        state["transcript"] = {
            "language": transcript.language,
            "language_probability": transcript.language_probability,
            "clip_ranges": [list(item) for item in transcript.clip_ranges],
        }
        _invalidate_after(state, "transcribe")
        _mark_stage(
            job_dir,
            state,
            "transcribe",
            artifacts={
                "transcript_text": transcript.text_path.name,
                "transcript_markdown": transcript.markdown_path.name,
                "transcript_srt": transcript.srt_path.name,
            },
        )
    elif "transcribe" not in state["completed_stages"]:
        raise RuntimeError(
            "확인 완료된 전사본이 없습니다. 고급 모드의 ‘3. GPU 전사’를 먼저 실행해주세요."
        )
    if target_stage == "transcribe":
        emit(100, "전사 완료", "전사본을 저장했습니다. AI 노트 단계부터 이어서 할 수 있습니다.")
        return _make_result(job_dir, video, state, "transcribe")

    if target_stage in {"all", "notes"}:
        template = load_prompt(settings)
        (job_dir / "prompt_used.md").write_text(
            template.rstrip() + "\n", encoding="utf-8"
        )
        emit(84, "AI 노트", "확인된 일반 전사문으로 필기 노트를 구성합니다.")
        transcript_for_prompt = transcript or _load_transcript_state(job_dir, state)
        notes_path = job_dir / "lecture_notes.md"
        notes_model, reasoning_effort, _ = settings.ai_selection()
        generate_notes(
            prompt=_build_prompt(template, video, transcript_for_prompt),
            output_path=notes_path,
            provider=normalize_provider(settings.ai_provider),
            model=notes_model,
            reasoning_effort=reasoning_effort,
            log=log,
            cancel_event=cancel_event,
        )
        _mark_stage(
            job_dir,
            state,
            "notes",
            artifacts={"notes": notes_path.name},
        )
        if not settings.keep_audio and audio_path.exists():
            if _remove_original_audio(audio_path, log):
                state["artifacts"].pop("original_audio", None)
                _save_checkpoint(job_dir, state)
                if log:
                    log("전체 완료 후 원본 오디오만 정리했습니다. 편집본은 재전사를 위해 보존합니다.")
        if settings.keep_video and video_path and video_path.exists():
            if _remove_original_audio(video_path, log, media_label="영상"):
                state["artifacts"].pop("original_video", None)
                _save_checkpoint(job_dir, state)
                if log:
                    log("전체 완료 후 원본 영상은 정리하고 쉬는 시간을 뺀 MP4만 보존합니다.")
        emit(100, "완료", "단계별 결과와 강의 노트를 모두 저장했습니다.")
        return _make_result(job_dir, video, state, "notes")

    raise RuntimeError("실행할 파이프라인 단계가 없습니다.")


def _remove_original_audio(
    audio_path: Path,
    log: LogCallback | None = None,
    retry_delays: tuple[float, ...] = (0.0, 0.2, 0.5, 1.0, 2.0),
    media_label: str = "오디오",
) -> bool:
    """Best-effort source cleanup that never invalidates completed output."""
    last_error: OSError | None = None
    for delay in retry_delays:
        if delay:
            sleep(delay)
        try:
            audio_path.unlink()
            return True
        except FileNotFoundError:
            return True
        except PermissionError as exc:
            last_error = exc
        except OSError as exc:
            if getattr(exc, "winerror", None) != 32:
                raise
            last_error = exc
    if log:
        log(
            f"원본 {media_label}가 미디어 재생기 또는 다른 프로그램에서 사용 중이라 "
            f"삭제하지 못했습니다. 결과는 정상 완료됐으며 원본은 그대로 보존합니다: {audio_path} "
            f"({last_error})"
        )
    return False


def _resolve_job(
    root: Path,
    video: VideoInfo,
    target_stage: str,
    log: LogCallback | None,
) -> tuple[Path, dict]:
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    candidates = sorted(root.glob(f"*_{video.video_id}_*"), reverse=True)
    for directory in candidates:
        state = _load_checkpoint(directory)
        if not state:
            state = _migrate_legacy_job(directory, video)
        if not state or state.get("video", {}).get("video_id") != video.video_id:
            continue
        if target_stage == "all" and "notes" in state.get("completed_stages", []):
            continue
        required_artifact = {
            "edit": "original_audio",
            "transcribe": "edited_audio",
            "notes": "transcript_text",
        }.get(target_stage)
        if required_artifact:
            artifact = _state_artifact(directory, state, required_artifact)
            if not artifact or not artifact.exists():
                continue
        if log:
            log(f"기존 작업을 이어서 사용합니다: {directory.name}")
        return directory.resolve(), state
    directory = _create_job_dir(root, video)
    return directory, {}


def _ensure_state(state: dict, video: VideoInfo, settings: AppSettings) -> None:
    state.setdefault("version", 2)
    state.setdefault("video", asdict(video))
    state.setdefault("created_at", datetime.now().astimezone().isoformat(timespec="seconds"))
    state.setdefault("updated_at", state["created_at"])
    state.setdefault("completed_stages", [])
    state.setdefault("artifacts", {})
    state["settings"] = {
        "whisper_model": settings.whisper_model,
        "language": settings.language,
        "beam_size": settings.beam_size,
        "ai_provider": settings.ai_provider,
        "notes_model": settings.ai_selection()[0],
        "reasoning_effort": settings.ai_selection()[1],
        "keep_video": settings.keep_video,
        "auto_terminology": settings.auto_terminology,
        "terminology_lecture_name": settings.terminology_lecture_name,
        "terminology_model": settings.ai_selection()[2],
    }


def _prepare_transcription_terms(
    job_dir: Path,
    state: dict,
    settings: AppSettings,
    video: VideoInfo,
    *,
    log: LogCallback | None = None,
    cancel_event: Event | None = None,
) -> tuple[str, ...]:
    lecture_name = " ".join(settings.terminology_lecture_name.split()).strip()
    if not lecture_name:
        raise ValueError("전문용어 자동 입력을 사용하려면 강의명을 입력해주세요.")

    saved_info = state.get("terminology", {})
    saved_path = _state_artifact(job_dir, state, "transcription_terms")
    if (
        saved_path
        and saved_path.exists()
        and saved_info.get("lecture_name") == lecture_name
    ):
        terms = tuple(
            line.strip()
            for line in saved_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if terms:
            if log:
                log(f"저장된 전사 전문용어 {len(terms)}개를 재사용합니다: {saved_path.name}")
            return terms

    terminology_model = settings.ai_selection()[2]
    terms = generate_transcription_terms(
        lecture_name,
        provider=normalize_provider(settings.ai_provider),
        video_title=video.title,
        model=terminology_model,
        reasoning_effort="medium",
        log=log,
        cancel_event=cancel_event,
    )
    terms_path = job_dir / "transcription_terms.txt"
    terms_path.write_text("\n".join(terms) + "\n", encoding="utf-8")
    state["terminology"] = {
        "lecture_name": lecture_name,
        "provider": normalize_provider(settings.ai_provider),
        "model": terminology_model,
        "count": len(terms),
    }
    state.setdefault("artifacts", {})["transcription_terms"] = terms_path.name
    _save_checkpoint(job_dir, state)
    if log:
        preview = ", ".join(terms[:8])
        suffix = " …" if len(terms) > 8 else ""
        log(f"전사 전문용어 {len(terms)}개 준비 완료: {preview}{suffix}")
    return terms


def _mark_stage(
    job_dir: Path,
    state: dict,
    stage: str,
    artifacts: dict[str, str] | None = None,
) -> None:
    if stage not in state["completed_stages"]:
        state["completed_stages"].append(stage)
    state["completed_stages"].sort(key=PIPELINE_STAGES.index)
    if artifacts:
        state["artifacts"].update(artifacts)
    _save_checkpoint(job_dir, state)


def _invalidate_after(state: dict, stage: str) -> None:
    stage_index = PIPELINE_STAGES.index(stage)
    state["completed_stages"] = [
        item
        for item in state.get("completed_stages", [])
        if PIPELINE_STAGES.index(item) <= stage_index
    ]


def _save_checkpoint(job_dir: Path, state: dict) -> None:
    state["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    path = job_dir / "checkpoint.json"
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _load_checkpoint(job_dir: Path) -> dict | None:
    path = job_dir / "checkpoint.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def _migrate_legacy_job(job_dir: Path, video: VideoInfo) -> dict | None:
    metadata_path = job_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if metadata.get("video", {}).get("video_id") != video.video_id:
        return None
    state = {
        "version": 2,
        "video": metadata["video"],
        "created_at": metadata.get("created_at"),
        "updated_at": metadata.get("created_at"),
        "completed_stages": [],
        "artifacts": {},
    }
    audio = next(
        (
            item
            for item in job_dir.glob(f"{video.video_id}.*")
            if item.suffix not in {".part", ".ytdl"}
        ),
        None,
    )
    if audio:
        state["completed_stages"].append("download")
        state["artifacts"]["original_audio"] = audio.name
    return state


def _state_artifact(job_dir: Path, state: dict, key: str) -> Path | None:
    value = state.get("artifacts", {}).get(key)
    return (job_dir / value).resolve() if value else None


def _make_result(
    job_dir: Path, video: VideoInfo, state: dict, completed_stage: str
) -> PipelineResult:
    return PipelineResult(
        output_dir=job_dir.resolve(),
        notes_path=_state_artifact(job_dir, state, "notes"),
        transcript_path=_state_artifact(job_dir, state, "transcript_markdown"),
        srt_path=_state_artifact(job_dir, state, "transcript_srt"),
        video_path=_state_artifact(job_dir, state, "edited_video"),
        video_title=video.title,
        completed_stage=completed_stage,
    )


def _load_transcript_state(job_dir: Path, state: dict) -> TranscriptResult:
    text_path = _state_artifact(job_dir, state, "transcript_text")
    markdown_path = _state_artifact(job_dir, state, "transcript_markdown")
    srt_path = _state_artifact(job_dir, state, "transcript_srt")
    if not text_path or not markdown_path or not srt_path:
        raise RuntimeError("체크포인트에서 전사 파일을 찾지 못했습니다.")
    info = state.get("transcript", {})
    ranges = tuple(
        (float(start), float(end)) for start, end in info.get("clip_ranges", [])
    )
    if not ranges:
        ranges = ((0.0, float(state["video"].get("duration") or 1)),)
    return TranscriptResult(
        text_path=text_path,
        markdown_path=markdown_path,
        srt_path=srt_path,
        language=str(info.get("language") or "unknown"),
        language_probability=float(info.get("language_probability") or 0),
        segments=(),
        clip_start=ranges[0][0],
        clip_end=ranges[-1][1],
        clip_ranges=ranges,
    )


def _create_job_dir(root: Path, video: VideoInfo) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    directory = root / f"{stamp}_{video.video_id}_{safe_filename(video.title, 55)}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory.resolve()


def _write_metadata(
    job_dir: Path,
    video: VideoInfo,
    settings: AppSettings,
    clip_start: float,
    clip_end: float | None,
    exclusions: tuple[tuple[float, float], ...] = (),
    included_ranges: tuple[tuple[float, float], ...] = (),
    original_video: Path | None = None,
    edited_video: Path | None = None,
) -> None:
    data = {
        "video": asdict(video),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "transcription": {
            "model": settings.whisper_model,
            "language": settings.language,
            "compute_type": settings.compute_type,
            "beam_size": settings.beam_size,
            "vad_filter": True,
            "clip_start_seconds": clip_start,
            "clip_end_seconds": clip_end,
            "excluded_break_ranges": [list(item) for item in exclusions],
            "included_lecture_ranges": [list(item) for item in included_ranges],
            "source": "edited_lecture.flac",
            "auto_terminology": settings.auto_terminology,
            "terminology_lecture_name": settings.terminology_lecture_name,
            "terminology_model": settings.ai_selection()[2],
        },
        "notes": {
            "provider": settings.ai_provider,
            "model": settings.ai_selection()[0],
            "reasoning_effort": settings.ai_selection()[1],
        },
        "video_output": {
            "enabled": settings.keep_video,
            "original": original_video.name if original_video else None,
            "edited": edited_video.name if edited_video else None,
        },
    }
    (job_dir / "metadata.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _build_prompt(
    template: str, video: VideoInfo, transcript: TranscriptResult
) -> str:
    transcript_text = transcript.text_path.read_text(encoding="utf-8")
    probability = f"{transcript.language_probability:.1%}"
    return render_prompt(
        template,
        PromptContext(
            title=video.title,
            channel=video.channel,
            url=video.webpage_url,
            duration=format_duration(video.duration),
            language=f"{transcript.language} (신뢰도 {probability})",
            transcript=transcript_text,
        ),
    )


def _resolve_clip_range(
    start: float | None,
    end: float | None,
    video: VideoInfo,
    log: LogCallback | None,
) -> tuple[float, float | None]:
    clip_start, clip_end = validate_clip_range(start, end)
    if video.duration > 0 and clip_start >= video.duration:
        raise ValueError(
            f"전사 시작 시간이 영상 길이({format_duration(video.duration)})보다 뒤입니다."
        )
    if video.duration > 0 and clip_end is not None and clip_end > video.duration:
        if log:
            log("전사 종료 시간이 영상보다 길어 영상 끝으로 조정했습니다.")
        clip_end = video.duration
    return clip_start, clip_end


def _write_review(job_dir: Path, review: TranscriptReview) -> None:
    data = {
        "status": review.status,
        "headline": review.headline,
        "metrics": review.metrics,
        "warnings": list(review.warnings),
    }
    (job_dir / "transcript_review.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
