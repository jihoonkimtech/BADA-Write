"""Write transcription results to txt, srt, vtt, tsv and json without depending on any engine."""

from __future__ import annotations

import json
import os

OUTPUT_FORMATS = ["txt", "srt", "vtt", "tsv", "json"]


def _clock(seconds: float, decimal: str) -> str:
    # Subtitle timestamp HH:MM:SS<decimal>mmm
    millis = int(round(max(seconds, 0.0) * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal}{millis:03d}"


def _segments(result: dict) -> list:
    # Skip empty segments so subtitles never contain blank cues
    return [seg for seg in result.get("segments", []) if str(seg.get("text", "")).strip()]


def to_txt(result: dict) -> str:
    return "\n".join(seg["text"].strip() for seg in _segments(result)) + "\n"


def to_srt(result: dict) -> str:
    blocks = []
    for i, seg in enumerate(_segments(result), start=1):
        start = _clock(seg["start"], ",")
        end = _clock(seg["end"], ",")
        blocks.append(f"{i}\n{start} --> {end}\n{seg['text'].strip()}\n")
    return "\n".join(blocks)


def to_vtt(result: dict) -> str:
    cues = [
        f"{_clock(seg['start'], '.')} --> {_clock(seg['end'], '.')}\n{seg['text'].strip()}\n"
        for seg in _segments(result)
    ]
    return "WEBVTT\n\n" + "\n".join(cues)


def to_tsv(result: dict) -> str:
    # Same layout as the Whisper CLI: start and end in integer milliseconds
    rows = ["start\tend\ttext"]
    for seg in _segments(result):
        text = seg["text"].strip().replace("\t", " ")
        rows.append(f"{int(round(seg['start'] * 1000))}\t{int(round(seg['end'] * 1000))}\t{text}")
    return "\n".join(rows) + "\n"


def to_json(result: dict) -> str:
    return json.dumps(result, ensure_ascii=False, indent=2)


_RENDERERS = {"txt": to_txt, "srt": to_srt, "vtt": to_vtt, "tsv": to_tsv, "json": to_json}


def write_result(result: dict, out_path: str) -> str:
    # Pick the format from the file extension, e.g. .srt -> srt
    fmt = os.path.splitext(out_path)[1].lstrip(".").lower()
    if fmt not in _RENDERERS:
        raise ValueError(f"지원하지 않는 형식입니다: .{fmt} (지원: {', '.join(OUTPUT_FORMATS)})")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(_RENDERERS[fmt](result))
    return out_path
