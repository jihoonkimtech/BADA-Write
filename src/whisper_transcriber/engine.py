"""Whisper transcription engine, independent of any GUI toolkit."""

from __future__ import annotations

import contextlib
import importlib
import os
import shutil
import threading
from typing import Callable, Iterator, Optional

# Model names exposed in the UI, ordered from fastest to most accurate
MODEL_NAMES = ["tiny", "base", "small", "medium", "turbo", "large-v3"]
DEFAULT_MODEL = "small"

# Display label -> Whisper language code (None means auto-detect)
LANGUAGES = {
    "자동 감지": None,
    "한국어": "ko",
    "영어": "en",
    "일본어": "ja",
    "중국어": "zh",
}
DEFAULT_LANGUAGE = "한국어"

# Output formats supported by whisper.utils.get_writer
OUTPUT_FORMATS = ["txt", "srt", "vtt", "tsv", "json"]

MEDIA_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov", ".webm", ".mp3", ".wav", ".m4a", ".flac", ".ogg")

ProgressCallback = Callable[[float], None]


class FFmpegNotFoundError(RuntimeError):
    """Raised when the ffmpeg binary is not available on PATH."""


def ensure_ffmpeg() -> None:
    # Whisper shells out to ffmpeg for decoding, so fail early with a clear message
    if shutil.which("ffmpeg") is None:
        raise FFmpegNotFoundError(
            "ffmpeg를 찾을 수 없습니다. ffmpeg를 설치하고 PATH에 추가한 뒤 다시 실행해 주세요."
        )


def detect_device() -> str:
    # Lazy import keeps app startup fast since torch is heavy
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


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


class _ProgressBar:
    """Minimal stand-in for tqdm.tqdm that forwards progress to a callback."""

    def __init__(self, callback: ProgressCallback, total: Optional[int] = None, **_kwargs):
        self._callback = callback
        self._total = total or 0
        self._done = 0

    def __enter__(self) -> "_ProgressBar":
        return self

    def __exit__(self, *exc) -> None:
        return None

    def update(self, n: int = 1) -> None:
        # Report a 0.0-1.0 ratio, clamped because whisper may overshoot slightly
        self._done += n
        if self._total > 0:
            self._callback(min(self._done / self._total, 1.0))


@contextlib.contextmanager
def _patched_progress(callback: Optional[ProgressCallback]) -> Iterator[None]:
    # whisper.transcribe has no progress hook, so swap its tqdm reference temporarily
    if callback is None:
        yield
        return

    # import_module returns the submodule, not the same-named function on the package
    module = importlib.import_module("whisper.transcribe")
    original = module.tqdm

    class _TqdmShim:
        @staticmethod
        def tqdm(*args, **kwargs):
            kwargs.pop("disable", None)
            return _ProgressBar(callback, *args, **kwargs)

    module.tqdm = _TqdmShim
    try:
        yield
    finally:
        module.tqdm = original


class Transcriber:
    """Loads Whisper models once and reuses them across runs."""

    def __init__(self) -> None:
        self._model = None
        self._model_key: Optional[tuple] = None
        self._lock = threading.Lock()

    def load_model(self, name: str, device: str):
        # Keep only one model in memory; reload only when name or device changes
        key = (name, device)
        if self._model_key != key:
            import whisper

            self._model = None
            self._model = whisper.load_model(name, device=device)
            self._model_key = key
        return self._model

    def transcribe(
        self,
        path: str,
        model_name: str = DEFAULT_MODEL,
        language: Optional[str] = None,
        device: Optional[str] = None,
        on_status: Optional[Callable[[str], None]] = None,
        on_progress: Optional[ProgressCallback] = None,
    ) -> dict:
        # Validate inputs before spending time on model loading
        if not os.path.isfile(path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
        ensure_ffmpeg()

        status = on_status or (lambda _msg: None)
        device = device or detect_device()

        # Serialize runs so the model and the tqdm patch are never shared concurrently
        with self._lock:
            status(f"모델 로딩 중... ({model_name}, {device.upper()})")
            model = self.load_model(model_name, device)

            status("음성 인식 중...")
            with _patched_progress(on_progress):
                # fp16 is only supported on GPU; passing False avoids the CPU warning
                return model.transcribe(
                    path,
                    language=language,
                    fp16=(device == "cuda"),
                    verbose=False if on_progress else None,
                )


def write_result(result: dict, out_path: str) -> str:
    # Pick the format from the file extension, e.g. .srt -> srt
    fmt = os.path.splitext(out_path)[1].lstrip(".").lower()
    if fmt not in OUTPUT_FORMATS:
        raise ValueError(f"지원하지 않는 형식입니다: .{fmt} (지원: {', '.join(OUTPUT_FORMATS)})")

    # Reuse Whisper's own writers so output matches the official CLI
    from whisper.utils import get_writer

    writer = get_writer(fmt, os.path.dirname(out_path) or ".")
    # Writers call options.get(), so a dict is required instead of None
    options = {"max_line_width": None, "max_line_count": None, "highlight_words": False}
    with open(out_path, "w", encoding="utf-8") as f:
        writer.write_result(result, file=f, options=options)
    return out_path
