from __future__ import annotations

from threading import Event

from PySide6.QtCore import QObject, Signal, Slot

from ..audio_edit import WaveformCancelled
from ..codex_client import CodexCancelled, auth_status, login, logout
from ..config import AppSettings
from ..pipeline import PipelineCancelled, run_pipeline
from ..transcription import TranscriptionCancelled
from ..updater import check_and_update
from ..youtube import DownloadCancelled


class PipelineWorker(QObject):
    progress = Signal(int, str, str)
    log = Signal(str)
    result = Signal(object)
    audio_edit_ready = Signal(object)
    review_ready = Signal(object)
    cancelled = Signal(str)
    error = Signal(str)
    finished = Signal()

    def __init__(self, url: str, settings: AppSettings, target_stage: str = "all"):
        super().__init__()
        self.url = url
        self.settings = settings
        self.target_stage = target_stage
        self.cancel_event = Event()
        self._audio_edit_event = Event()
        self._audio_exclusions: tuple[tuple[float, float], ...] | None = None
        self._review_event = Event()
        self._review_approved: bool | None = None

    @Slot()
    def run(self) -> None:
        try:
            value = run_pipeline(
                self.url,
                self.settings,
                progress=self.progress.emit,
                log=self.log.emit,
                cancel_event=self.cancel_event,
                edit_audio=self._await_audio_edit,
                review_transcript=self._await_review,
                target_stage=self.target_stage,
            )
            self.result.emit(value)
        except (
            PipelineCancelled,
            TranscriptionCancelled,
            DownloadCancelled,
            CodexCancelled,
            WaveformCancelled,
        ) as exc:
            self.cancelled.emit(str(exc))
        except Exception as exc:  # UI boundary: present an actionable error to the user.
            self.error.emit(str(exc))
        finally:
            self.finished.emit()

    def cancel(self) -> None:
        self.cancel_event.set()
        self._audio_edit_event.set()
        self._review_event.set()

    def submit_audio_edit(
        self, exclusions: tuple[tuple[float, float], ...] | None
    ) -> None:
        self._audio_exclusions = exclusions
        self._audio_edit_event.set()

    def _await_audio_edit(
        self, request: object
    ) -> tuple[tuple[float, float], ...] | None:
        self._audio_exclusions = None
        self._audio_edit_event.clear()
        self.audio_edit_ready.emit(request)
        while not self._audio_edit_event.wait(0.2):
            if self.cancel_event.is_set():
                raise PipelineCancelled("사용자가 작업을 취소했습니다.")
        if self.cancel_event.is_set():
            raise PipelineCancelled("사용자가 작업을 취소했습니다.")
        return self._audio_exclusions

    def submit_review(self, approved: bool) -> None:
        self._review_approved = approved
        self._review_event.set()

    def _await_review(self, review: object) -> bool:
        self._review_approved = None
        self._review_event.clear()
        self.review_ready.emit(review)
        while not self._review_event.wait(0.2):
            if self.cancel_event.is_set():
                raise PipelineCancelled("사용자가 작업을 취소했습니다.")
        if self.cancel_event.is_set():
            raise PipelineCancelled("사용자가 작업을 취소했습니다.")
        return self._review_approved is True


class AuthWorker(QObject):
    log = Signal(str)
    result = Signal(object)
    error = Signal(str)
    finished = Signal()

    def __init__(self, action: str):
        super().__init__()
        self.action = action
        self.cancel_event = Event()

    @Slot()
    def run(self) -> None:
        try:
            if self.action == "login":
                value = login(self.log.emit, self.cancel_event)
            elif self.action == "logout":
                value = logout()
            else:
                value = auth_status()
            self.result.emit(value)
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.finished.emit()

    def cancel(self) -> None:
        self.cancel_event.set()


class UpdateWorker(QObject):
    log = Signal(str)
    result = Signal(object)
    error = Signal(str)
    finished = Signal()

    def __init__(self, channel: str):
        super().__init__()
        self.channel = channel
        self.cancel_event = Event()

    @Slot()
    def run(self) -> None:
        try:
            value = check_and_update(
                self.channel, log=self.log.emit, cancel_event=self.cancel_event
            )
            self.result.emit(value)
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.finished.emit()
