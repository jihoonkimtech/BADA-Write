"""Transcription orchestration shared by every engine, independent of any GUI toolkit."""

from __future__ import annotations

import os
import threading
from typing import Callable, Optional

from .backends import (  # noqa: F401  (re-exported for callers and tests)
    BACKENDS,
    Backend,
    FFmpegNotFoundError,
    ProgressCallback,
    cuda_arch_supported,
    ensure_ffmpeg,
)
from .writers import OUTPUT_FORMATS, write_result  # noqa: F401

# Model names understood by both engines, ordered from fastest to most accurate
MODEL_NAMES = ["tiny", "base", "small", "medium", "turbo", "large-v3"]
DEFAULT_MODEL = "turbo"

# Display label -> Whisper language code (None means auto-detect)
LANGUAGES = {
    "자동 감지": None,
    "한국어": "ko",
    "영어": "en",
    "일본어": "ja",
    "중국어": "zh",
}
DEFAULT_LANGUAGE = "한국어"

MEDIA_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov", ".webm", ".mp3", ".wav", ".m4a", ".flac", ".ogg")

# Substrings that mean the GPU stack itself is unusable, so CPU is a safe retry
_GPU_FAILURE_MARKERS = (
    "no kernel image",
    "cublas",
    "cudnn",
    "cuda driver",
    "cannot be loaded",
    "is not found",
    "not compiled with cuda",
)


def available_backends() -> list:
    # Engines whose Python package is installed, in preference order
    return [name for name, backend in BACKENDS.items() if backend.is_available()]


def is_gpu_failure(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in _GPU_FAILURE_MARKERS)


def format_timestamp(seconds: float) -> str:
    # Render seconds as HH:MM:SS.mmm, dropping hours when zero
    millis = int(round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    prefix = f"{hours:02d}:" if hours else ""
    return f"{prefix}{minutes:02d}:{secs:02d}.{millis:03d}"


def format_segments(result: dict) -> str:
    # One line per segment with its start and end time
    lines = []
    for seg in result.get("segments", []):
        start = format_timestamp(seg["start"])
        end = format_timestamp(seg["end"])
        lines.append(f"[{start} --> {end}] {seg['text'].strip()}")
    return "\n".join(lines)


class Transcriber:
    """Loads models once per (engine, model, device) and reuses them across runs."""

    def __init__(self, backends: Optional[dict] = None) -> None:
        self.backends = backends if backends is not None else BACKENDS
        self._model = None
        self._model_key: Optional[tuple] = None
        self._lock = threading.Lock()

    def _get_backend(self, name: Optional[str]) -> Backend:
        if name is None:
            installed = [n for n, b in self.backends.items() if b.is_available()]
            if not installed:
                raise RuntimeError("설치된 엔진이 없습니다. faster-whisper 또는 openai-whisper를 설치해 주세요.")
            name = installed[0]
        if name not in self.backends:
            raise ValueError(f"알 수 없는 엔진입니다: {name}")
        return self.backends[name]

    def load_model(self, backend: Backend, model_name: str, device: str):
        # Keep only one model in memory; reload when engine, name or device changes
        key = (backend.name, model_name, device)
        if self._model_key != key:
            self._model = None
            self._model = backend.load(model_name, device)
            self._model_key = key
        return self._model

    def transcribe(
        self,
        path: str,
        backend: Optional[str] = None,
        model_name: str = DEFAULT_MODEL,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        suppress_hallucination: bool = False,
        device: Optional[str] = None,
        on_status: Optional[Callable[[str], None]] = None,
        on_progress: Optional[ProgressCallback] = None,
    ) -> dict:
        # Validate inputs before spending time on model loading
        if not os.path.isfile(path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

        engine = self._get_backend(backend)
        engine.prepare()
        status = on_status or (lambda _msg: None)
        hint = (initial_prompt or "").strip()

        note = None
        if device is None:
            device, note = engine.detect_device()

        # Serialize runs so a model is never shared between concurrent transcriptions
        with self._lock:
            try:
                return self._run(engine, path, model_name, device, language, hint,
                                 suppress_hallucination, status, on_progress, note)
            except Exception as exc:
                if device != "cuda" or not is_gpu_failure(exc):
                    raise
                # GPU libraries are missing or incompatible; retry once on CPU
                self._model, self._model_key = None, None
                note = f"GPU 실행 실패({type(exc).__name__})로 CPU로 다시 실행합니다"
                return self._run(engine, path, model_name, "cpu", language, hint,
                                 suppress_hallucination, status, on_progress, note)

    def _run(self, engine, path, model_name, device, language, hint, suppress, status, on_progress, note):
        prefix = f"{note} / " if note else ""
        status(f"{prefix}모델 로딩 중... ({engine.name}, {model_name}, {device.upper()})")
        model = self.load_model(engine, model_name, device)

        status(f"{prefix}음성 인식 중... ({engine.name}, {device.upper()})")
        if on_progress:
            on_progress(0.0)
        return engine.run(
            model, path,
            device=device, language=language, hint=hint,
            suppress=suppress, on_progress=on_progress,
        )
