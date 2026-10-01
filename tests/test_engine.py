"""Engine and backend tests that run without downloading any model."""

import importlib
from types import SimpleNamespace

import pytest

from whisper_transcriber import backends, engine

SAMPLE_RESULT = {
    "text": " 안녕하세요. 테스트입니다.",
    "language": "ko",
    "segments": [
        {"id": 0, "start": 0.0, "end": 1.5, "text": " 안녕하세요."},
        {"id": 1, "start": 1.5, "end": 3723.25, "text": " 테스트입니다."},
    ],
}


class FakeOpenAIModel:
    """Mimics whisper.Whisper.transcribe including its tqdm progress loop."""

    def __init__(self):
        self.kwargs = None

    def transcribe(self, path, **kwargs):
        self.kwargs = kwargs
        # Use the module attribute exactly like whisper/transcribe.py does
        tqdm = importlib.import_module("whisper.transcribe").tqdm
        with tqdm.tqdm(total=300, unit="frames", disable=True) as pbar:
            for _ in range(3):
                pbar.update(100)
        return SAMPLE_RESULT


class FakeFasterModel:
    """Mimics faster_whisper.WhisperModel.transcribe returning a lazy segment generator."""

    def __init__(self):
        self.kwargs = None

    def transcribe(self, path, language=None, hotwords=None, vad_filter=False, condition_on_previous_text=True,
                   initial_prompt=None):
        self.kwargs = dict(language=language, hotwords=hotwords, vad_filter=vad_filter,
                           condition_on_previous_text=condition_on_previous_text, initial_prompt=initial_prompt)
        segs = [SimpleNamespace(start=0.0, end=5.0, text=" 축전기"), SimpleNamespace(start=5.0, end=10.0, text=" 전하")]
        return iter(segs), SimpleNamespace(duration=10.0, language="ko")


class FakeBackend(backends.Backend):
    """Backend whose GPU run fails with a configurable error, for fallback tests."""

    name = "fake"

    def __init__(self, gpu_error=None):
        self.gpu_error = gpu_error
        self.loads = []
        self.runs = []

    def is_available(self):
        return True

    def detect_device(self):
        return "cuda", None

    def load(self, model_name, device):
        self.loads.append((model_name, device))
        return object()

    def run(self, model, path, *, device, language, hint, suppress, on_progress):
        self.runs.append(dict(device=device, hint=hint, suppress=suppress))
        if device == "cuda" and self.gpu_error:
            raise self.gpu_error
        return SAMPLE_RESULT


@pytest.fixture
def media_file(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"\x00")
    return str(path)


# ---------- formatting ----------


def test_format_timestamp():
    assert engine.format_timestamp(0) == "00:00.000"
    assert engine.format_timestamp(61.5) == "01:01.500"
    assert engine.format_timestamp(3723.25) == "01:02:03.250"


def test_format_segments():
    lines = engine.format_segments(SAMPLE_RESULT).splitlines()
    assert lines[0] == "[00:00.000 --> 00:01.500] 안녕하세요."
    assert lines[1].startswith("[00:01.500 --> 01:02:03.250]")


# ---------- transcriber orchestration ----------


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        engine.Transcriber().transcribe("/no/such/file.mp4")


def test_no_engine_installed(media_file):
    class Missing(FakeBackend):
        def is_available(self):
            return False

    with pytest.raises(RuntimeError, match="설치된 엔진이 없습니다"):
        engine.Transcriber({"fake": Missing()}).transcribe(media_file)


def test_default_backend_and_options_forwarded(media_file):
    fake = FakeBackend()
    statuses = []
    engine.Transcriber({"fake": fake}).transcribe(
        media_file, initial_prompt="  축전기 ", suppress_hallucination=True, on_status=statuses.append
    )
    assert fake.runs == [dict(device="cuda", hint="축전기", suppress=True)]
    assert "fake" in statuses[0]


def test_gpu_failure_retries_on_cpu(media_file):
    fake = FakeBackend(RuntimeError("Library cublas64_12.dll is not found or cannot be loaded"))
    statuses = []
    result = engine.Transcriber({"fake": fake}).transcribe(media_file, on_status=statuses.append)
    assert result is SAMPLE_RESULT
    assert [r["device"] for r in fake.runs] == ["cuda", "cpu"]
    assert fake.loads == [("turbo", "cuda"), ("turbo", "cpu")]
    assert "CPU로 다시 실행" in statuses[-1]


def test_non_gpu_error_is_not_retried(media_file):
    fake = FakeBackend(ValueError("bad audio"))
    with pytest.raises(ValueError):
        engine.Transcriber({"fake": fake}).transcribe(media_file)
    assert len(fake.runs) == 1


def test_model_is_cached(media_file):
    fake = FakeBackend()
    t = engine.Transcriber({"fake": fake})
    t.transcribe(media_file, model_name="tiny")
    t.transcribe(media_file, model_name="tiny")
    t.transcribe(media_file, model_name="base")
    assert fake.loads == [("tiny", "cuda"), ("base", "cuda")]


# ---------- faster-whisper backend ----------


def test_faster_options_use_hotwords_with_real_signature():
    from faster_whisper import WhisperModel

    opts = backends.FasterWhisperBackend().build_options(WhisperModel.transcribe, "축전기, 전하", True)
    assert opts == {"hotwords": "축전기, 전하", "vad_filter": True, "condition_on_previous_text": False}


def test_faster_options_empty():
    assert backends.FasterWhisperBackend().build_options(FakeFasterModel().transcribe, "", False) == {}


def test_faster_run_collects_segments_and_progress():
    model = FakeFasterModel()
    progress = []
    result = backends.FasterWhisperBackend().run(
        model, "x.mp4", device="cpu", language="ko", hint="축전기", suppress=True, on_progress=progress.append
    )
    assert result["text"] == " 축전기 전하"
    assert result["language"] == "ko"
    assert [s["end"] for s in result["segments"]] == [5.0, 10.0]
    assert progress == [0.5, 1.0, 1.0]
    assert model.kwargs["hotwords"] == "축전기"
    assert model.kwargs["vad_filter"] is True


# ---------- openai-whisper backend ----------


def test_openai_options_with_real_signature():
    import whisper

    opts = backends.OpenAIWhisperBackend().build_options(whisper.transcribe, "축전기", True)
    assert opts["initial_prompt"] == "축전기"
    assert opts["carry_initial_prompt"] is True
    assert opts["condition_on_previous_text"] is False
    assert opts["word_timestamps"] is True
    assert opts["hallucination_silence_threshold"] == backends.HALLUCINATION_SILENCE_SEC


def test_openai_options_with_old_signature():
    def old_transcribe(model, audio, *, initial_prompt=None, condition_on_previous_text=True):
        return {}

    opts = backends.OpenAIWhisperBackend().build_options(old_transcribe, "축전기", True)
    assert opts == {"initial_prompt": "축전기", "condition_on_previous_text": False}


def test_openai_run_reports_progress_and_restores_tqdm():
    module = importlib.import_module("whisper.transcribe")
    original = module.tqdm
    model = FakeOpenAIModel()
    progress = []

    result = backends.OpenAIWhisperBackend().run(
        model, "x.mp4", device="cpu", language="ko", hint="", suppress=False, on_progress=progress.append
    )

    assert result is SAMPLE_RESULT
    assert progress == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert module.tqdm is original
    assert model.kwargs["fp16"] is False


def test_openai_missing_ffmpeg(monkeypatch):
    monkeypatch.setattr(backends.shutil, "which", lambda _name: None)
    with pytest.raises(backends.FFmpegNotFoundError):
        backends.OpenAIWhisperBackend().prepare()


@pytest.mark.parametrize(
    "capability, arch_list, expected",
    [
        ((8, 6), ["sm_50", "sm_80", "sm_86", "sm_90"], True),
        ((8, 9), ["sm_80", "sm_86", "sm_90"], True),
        ((12, 0), ["sm_50", "sm_80", "sm_86", "sm_90"], False),
        ((12, 0), ["sm_90", "sm_120"], True),
        ((12, 0), ["sm_80", "compute_90"], True),
        ((3, 7), ["sm_50", "sm_60", "sm_90"], False),
    ],
)
def test_cuda_arch_supported(capability, arch_list, expected):
    assert backends.cuda_arch_supported(capability, arch_list) is expected


def test_available_backends_order():
    # Both engines are installed in the test environment; faster-whisper comes first
    assert engine.available_backends() == ["faster-whisper", "openai-whisper"]
