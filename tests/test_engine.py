"""Engine tests that run without downloading any Whisper model."""

import importlib
import json

import pytest

from whisper_transcriber import engine

SAMPLE_RESULT = {
    "text": " 안녕하세요. 테스트입니다.",
    "language": "ko",
    "segments": [
        {"id": 0, "start": 0.0, "end": 1.5, "text": " 안녕하세요."},
        {"id": 1, "start": 1.5, "end": 3723.25, "text": " 테스트입니다."},
    ],
}


class FakeModel:
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


@pytest.fixture
def media_file(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"\x00")
    return str(path)


def test_format_timestamp():
    assert engine.format_timestamp(0) == "00:00.000"
    assert engine.format_timestamp(61.5) == "01:01.500"
    assert engine.format_timestamp(3723.25) == "01:02:03.250"


def test_format_segments():
    lines = engine.format_segments(SAMPLE_RESULT).splitlines()
    assert lines[0] == "[00:00.000 --> 00:01.500] 안녕하세요."
    assert lines[1].startswith("[00:01.500 --> 01:02:03.250]")


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        engine.Transcriber().transcribe("/no/such/file.mp4")


def test_missing_ffmpeg_raises(monkeypatch, media_file):
    monkeypatch.setattr(engine.shutil, "which", lambda _name: None)
    with pytest.raises(engine.FFmpegNotFoundError):
        engine.Transcriber().transcribe(media_file, device="cpu")


def test_transcribe_reports_progress_and_restores_tqdm(monkeypatch, media_file):
    monkeypatch.setattr(engine.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    fake = FakeModel()
    transcriber = engine.Transcriber()
    monkeypatch.setattr(transcriber, "load_model", lambda name, device: fake)

    module = importlib.import_module("whisper.transcribe")
    original = module.tqdm
    progress, statuses = [], []

    result = transcriber.transcribe(
        media_file, model_name="tiny", language="ko", device="cpu",
        on_status=statuses.append, on_progress=progress.append,
    )

    assert result is SAMPLE_RESULT
    assert progress == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert module.tqdm is original
    assert fake.kwargs["fp16"] is False
    assert fake.kwargs["language"] == "ko"
    assert len(statuses) == 2


def test_model_is_cached(monkeypatch):
    calls = []
    import whisper

    monkeypatch.setattr(whisper, "load_model", lambda name, device: calls.append((name, device)) or object())
    t = engine.Transcriber()
    first = t.load_model("tiny", "cpu")
    assert t.load_model("tiny", "cpu") is first
    t.load_model("base", "cpu")
    assert calls == [("tiny", "cpu"), ("base", "cpu")]


@pytest.mark.parametrize("fmt", engine.OUTPUT_FORMATS)
def test_write_result_all_formats(tmp_path, fmt):
    out = engine.write_result(SAMPLE_RESULT, str(tmp_path / f"out.{fmt}"))
    content = open(out, encoding="utf-8").read()
    assert content
    if fmt == "srt":
        assert "00:00:00,000 --> 00:00:01,500" in content
    if fmt == "json":
        assert json.loads(content)["language"] == "ko"


def test_write_result_rejects_unknown_extension(tmp_path):
    with pytest.raises(ValueError):
        engine.write_result(SAMPLE_RESULT, str(tmp_path / "out.docx"))


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
    assert engine.cuda_arch_supported(capability, arch_list) is expected
