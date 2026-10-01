"""Speech recognition engines: faster-whisper (CTranslate2) and openai-whisper (PyTorch)."""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import inspect
import os
import shutil
import sys
from typing import Callable, Iterator, Optional

ProgressCallback = Callable[[float], None]

# Silent gaps longer than this (seconds) are skipped when a hallucination is suspected
HALLUCINATION_SILENCE_SEC = 2.0


class FFmpegNotFoundError(RuntimeError):
    """Raised when the ffmpeg binary is not available on PATH."""


def _accepts_kwarg(func: Callable, name: str) -> bool:
    # True when func has a parameter with this name or accepts **kwargs
    try:
        params = inspect.signature(func).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == name or p.kind is p.VAR_KEYWORD for p in params)


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def cuda_arch_supported(capability: tuple, arch_list: list) -> bool:
    # A CUDA build ships kernels only for listed GPU generations, e.g. sm_86 or compute_90
    major, minor = capability
    cap = major * 10 + minor
    for arch in arch_list:
        kind, _, num = arch.partition("_")
        if not num.isdigit():
            continue
        n = int(num)
        # Binary kernels run on the same major generation with an equal or newer minor
        if kind == "sm" and n // 10 == major and n <= cap:
            return True
        # PTX code can be JIT-compiled for any newer GPU
        if kind == "compute" and n <= cap:
            return True
    return False


class Backend:
    """Common interface every engine implements."""

    name = ""
    module = ""

    def is_available(self) -> bool:
        return _module_available(self.module)

    def prepare(self) -> None:
        # Hook for environment checks before loading a model
        return None

    def detect_device(self) -> tuple:
        raise NotImplementedError

    def load(self, model_name: str, device: str):
        raise NotImplementedError

    def run(self, model, path: str, *, device: str, language: Optional[str], hint: str,
            suppress: bool, on_progress: Optional[ProgressCallback]) -> dict:
        raise NotImplementedError


# ---------------- faster-whisper ----------------


def _add_windows_cuda_dll_dirs() -> None:
    # CTranslate2 needs cuBLAS/cuDNN DLLs; reuse the ones shipped with torch or nvidia-* pip packages
    if os.name != "nt":
        return
    candidates = []
    spec = importlib.util.find_spec("torch") if _module_available("torch") else None
    if spec and spec.origin:
        candidates.append(os.path.join(os.path.dirname(spec.origin), "lib"))
    if _module_available("nvidia"):
        for root in importlib.util.find_spec("nvidia").submodule_search_locations or []:
            candidates += [os.path.join(root, sub, "bin") for sub in ("cublas", "cudnn", "cuda_runtime")]
    for path in candidates:
        if os.path.isdir(path) and path not in os.environ.get("PATH", ""):
            os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
            with contextlib.suppress(OSError, AttributeError):
                os.add_dll_directory(path)


class FasterWhisperBackend(Backend):
    """CTranslate2 port of Whisper: faster, lighter, built-in VAD, decodes audio via PyAV."""

    name = "faster-whisper"
    module = "faster_whisper"

    def prepare(self) -> None:
        _add_windows_cuda_dll_dirs()

    def detect_device(self) -> tuple:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", None
        return "cpu", None

    def load(self, model_name: str, device: str):
        from faster_whisper import WhisperModel

        # float16 on GPU, int8 on CPU keeps speed high with little accuracy loss
        compute_type = "float16" if device == "cuda" else "int8"
        return WhisperModel(model_name, device=device, compute_type=compute_type)

    def build_options(self, transcribe_func: Callable, hint: str, suppress: bool) -> dict:
        options = {}
        if hint:
            # hotwords is prepended to every 30 s window, unlike initial_prompt
            key = "hotwords" if _accepts_kwarg(transcribe_func, "hotwords") else "initial_prompt"
            options[key] = hint
        if suppress:
            # VAD drops non-speech audio where most hallucinations start
            options["vad_filter"] = True
            # Stop one wrong window from being copied into the next ones
            options["condition_on_previous_text"] = False
        return options

    def run(self, model, path, *, device, language, hint, suppress, on_progress):
        options = self.build_options(model.transcribe, hint, suppress)
        segments_iter, info = model.transcribe(path, language=language, **options)

        # Segments are produced lazily; consuming them drives the actual decoding
        segments = []
        total = float(getattr(info, "duration", 0) or 0)
        for seg in segments_iter:
            segments.append({"id": len(segments), "start": float(seg.start), "end": float(seg.end), "text": seg.text})
            if on_progress and total > 0:
                on_progress(min(seg.end / total, 1.0))
        if on_progress:
            on_progress(1.0)

        return {
            "text": "".join(s["text"] for s in segments),
            "segments": segments,
            "language": getattr(info, "language", language),
        }


# ---------------- openai-whisper ----------------


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


def _add_bundled_ffmpeg_to_path() -> None:
    # PyInstaller builds unpack bundled files under sys._MEIPASS; expose ffmpeg there via PATH
    base = getattr(sys, "_MEIPASS", None)
    if not base:
        return
    bundled_dir = os.path.join(base, "ffmpeg")
    if os.path.isdir(bundled_dir) and bundled_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = bundled_dir + os.pathsep + os.environ.get("PATH", "")


def ensure_ffmpeg() -> None:
    # openai-whisper shells out to ffmpeg for decoding, so fail early with a clear message
    _add_bundled_ffmpeg_to_path()
    if shutil.which("ffmpeg") is None:
        raise FFmpegNotFoundError(
            "ffmpeg를 찾을 수 없습니다. ffmpeg를 설치하고 PATH에 추가하거나 faster-whisper 엔진을 사용해 주세요."
        )


class OpenAIWhisperBackend(Backend):
    """Reference PyTorch implementation from OpenAI."""

    name = "openai-whisper"
    module = "whisper"

    def prepare(self) -> None:
        ensure_ffmpeg()

    def detect_device(self) -> tuple:
        # Lazy import keeps app startup fast since torch is heavy
        import torch

        if not torch.cuda.is_available():
            return "cpu", None
        capability = torch.cuda.get_device_capability(0)
        if not cuda_arch_supported(capability, torch.cuda.get_arch_list()):
            # Avoid "no kernel image is available" by running on CPU instead
            name = torch.cuda.get_device_name(0)
            return "cpu", f"{name}는 설치된 PyTorch가 지원하지 않아 CPU로 실행합니다"
        return "cuda", None

    def load(self, model_name: str, device: str):
        import whisper

        return whisper.load_model(model_name, device=device)

    def build_options(self, transcribe_func: Callable, hint: str, suppress: bool) -> dict:
        options = {}
        if hint:
            options["initial_prompt"] = hint
            # By default the prompt only affects the first 30 s window; carry it when supported
            if _accepts_kwarg(transcribe_func, "carry_initial_prompt"):
                options["carry_initial_prompt"] = True
        if suppress:
            options["condition_on_previous_text"] = False
            # Skipping silent stretches needs word-level timing
            if _accepts_kwarg(transcribe_func, "hallucination_silence_threshold"):
                options["word_timestamps"] = True
                options["hallucination_silence_threshold"] = HALLUCINATION_SILENCE_SEC
        return options

    def run(self, model, path, *, device, language, hint, suppress, on_progress):
        options = self.build_options(model.transcribe, hint, suppress)
        with _patched_progress(on_progress):
            # fp16 is only supported on GPU; passing False avoids the CPU warning
            return model.transcribe(
                path,
                language=language,
                fp16=(device == "cuda"),
                verbose=False if on_progress else None,
                **options,
            )


BACKENDS = {b.name: b for b in (FasterWhisperBackend(), OpenAIWhisperBackend())}
