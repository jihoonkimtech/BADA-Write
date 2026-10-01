"""Subtitle and text writer tests."""

import json

import pytest

from whisper_transcriber.writers import OUTPUT_FORMATS, write_result

RESULT = {
    "text": " 안녕하세요. 테스트입니다.",
    "language": "ko",
    "segments": [
        {"start": 0.0, "end": 1.5, "text": " 안녕하세요."},
        {"start": 1.5, "end": 2.0, "text": "   "},
        {"start": 3661.25, "end": 3662.0, "text": " 테스트입니다."},
    ],
}


def _write(tmp_path, fmt):
    path = write_result(RESULT, str(tmp_path / f"out.{fmt}"))
    return open(path, encoding="utf-8").read()


def test_srt(tmp_path):
    assert _write(tmp_path, "srt") == (
        "1\n00:00:00,000 --> 00:00:01,500\n안녕하세요.\n\n"
        "2\n01:01:01,250 --> 01:01:02,000\n테스트입니다.\n"
    )


def test_vtt(tmp_path):
    content = _write(tmp_path, "vtt")
    assert content.startswith("WEBVTT\n\n00:00:00.000 --> 00:00:01.500\n안녕하세요.\n")


def test_tsv(tmp_path):
    lines = _write(tmp_path, "tsv").splitlines()
    assert lines == ["start\tend\ttext", "0\t1500\t안녕하세요.", "3661250\t3662000\t테스트입니다."]


def test_txt_and_json(tmp_path):
    assert _write(tmp_path, "txt") == "안녕하세요.\n테스트입니다.\n"
    assert json.loads(_write(tmp_path, "json"))["language"] == "ko"


def test_all_formats_write(tmp_path):
    for fmt in OUTPUT_FORMATS:
        assert _write(tmp_path, fmt)


def test_unknown_extension(tmp_path):
    with pytest.raises(ValueError):
        write_result(RESULT, str(tmp_path / "out.docx"))
