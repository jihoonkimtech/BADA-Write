"""Tkinter front-end for the Whisper transcription engine."""

from __future__ import annotations

import os
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from . import __version__
from .engine import (
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    LANGUAGES,
    MEDIA_EXTENSIONS,
    MODEL_NAMES,
    OUTPUT_FORMATS,
    Transcriber,
    format_segments,
    write_result,
)

POLL_INTERVAL_MS = 100


class TranscriberApp:
    """Main window: file selection, options, progress and result view."""

    def __init__(self, root: tk.Tk, transcriber: Transcriber | None = None) -> None:
        self.root = root
        self.root.title(f"Whisper 받아쓰기 v{__version__}")
        self.root.geometry("720x620")
        self.root.minsize(560, 480)

        self.transcriber = transcriber or Transcriber()
        self.selected_file_path = ""
        self.last_result: dict | None = None
        self.worker: threading.Thread | None = None
        self.started_at = 0.0
        self._status_base = "준비 중..."

        # Worker thread posts events here; only the main thread touches widgets
        self.events: "queue.Queue[tuple[str, object]]" = queue.Queue()

        self._build_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(POLL_INTERVAL_MS, self._poll_events)

    # ---------- layout ----------

    def _build_widgets(self) -> None:
        pad = {"padx": 12, "pady": 6}

        # File selection row
        file_frame = ttk.LabelFrame(self.root, text="파일 선택", padding=8)
        file_frame.pack(fill="x", **pad)
        self.file_label = ttk.Label(file_frame, text="선택된 파일이 없습니다.", foreground="gray")
        self.file_label.pack(side="left", fill="x", expand=True)
        ttk.Button(file_frame, text="파일 찾기", command=self.select_file).pack(side="right")

        # Options row: model, language, timestamp toggle
        opt = ttk.LabelFrame(self.root, text="설정", padding=8)
        opt.pack(fill="x", **pad)

        ttk.Label(opt, text="모델").grid(row=0, column=0, sticky="w")
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        ttk.Combobox(opt, textvariable=self.model_var, values=MODEL_NAMES, state="readonly", width=10).grid(
            row=0, column=1, padx=(4, 16)
        )

        ttk.Label(opt, text="언어").grid(row=0, column=2, sticky="w")
        self.lang_var = tk.StringVar(value=DEFAULT_LANGUAGE)
        ttk.Combobox(opt, textvariable=self.lang_var, values=list(LANGUAGES), state="readonly", width=10).grid(
            row=0, column=3, padx=(4, 16)
        )

        # Toggling only re-renders the existing result, no re-transcription needed
        self.timestamp_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt, text="타임스탬프 표시", variable=self.timestamp_var, command=self._render_result).grid(
            row=0, column=4, sticky="w"
        )

        # Run button and progress bar
        self.run_button = ttk.Button(self.root, text="받아쓰기 시작", command=self.start_transcription)
        self.run_button.pack(fill="x", **pad)

        self.progress = ttk.Progressbar(self.root, mode="determinate", maximum=100)
        self.progress.pack(fill="x", padx=12)

        self.status_label = ttk.Label(self.root, text="대기 중")
        self.status_label.pack(anchor="w", **pad)

        # Result actions, packed at the bottom first so the expanding text area cannot hide them
        actions = ttk.Frame(self.root)
        actions.pack(side="bottom", fill="x", padx=12, pady=(0, 10))
        self.save_button = ttk.Button(actions, text="파일로 저장", command=self.save_result, state="disabled")
        self.save_button.pack(side="right")
        self.copy_button = ttk.Button(actions, text="복사", command=self.copy_result, state="disabled")
        self.copy_button.pack(side="right", padx=6)

        # Result text area with scrollbar
        result_frame = ttk.LabelFrame(self.root, text="변환 결과", padding=8)
        result_frame.pack(fill="both", expand=True, **pad)
        self.result_text = tk.Text(result_frame, wrap="word", undo=True)
        scrollbar = ttk.Scrollbar(result_frame, command=self.result_text.yview)
        self.result_text.config(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.result_text.pack(side="left", fill="both", expand=True)

    # ---------- user actions ----------

    def select_file(self) -> None:
        patterns = " ".join(f"*{ext}" for ext in MEDIA_EXTENSIONS)
        path = filedialog.askopenfilename(
            title="영상 또는 음성 파일 선택",
            filetypes=[("Media Files", patterns), ("All Files", "*.*")],
        )
        if path:
            self.selected_file_path = path
            self.file_label.config(text=os.path.basename(path), foreground="")

    def start_transcription(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        if not self.selected_file_path:
            messagebox.showwarning("경고", "먼저 변환할 영상 또는 음성 파일을 선택해 주세요.")
            return

        # Read Tk variables here: Tk objects must not be accessed from the worker thread
        params = {
            "path": self.selected_file_path,
            "model_name": self.model_var.get(),
            "language": LANGUAGES[self.lang_var.get()],
        }

        self._set_busy(True)
        self.last_result = None
        self.result_text.delete("1.0", tk.END)
        self.progress["value"] = 0
        self.started_at = time.monotonic()

        self.worker = threading.Thread(target=self._run_worker, kwargs=params, daemon=True)
        self.worker.start()

    def copy_result(self) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(self.result_text.get("1.0", "end-1c"))
        self.status_label.config(text="클립보드에 복사했습니다.")

    def save_result(self) -> None:
        if not self.last_result:
            return
        base = os.path.splitext(os.path.basename(self.selected_file_path))[0] or "transcript"
        out_path = filedialog.asksaveasfilename(
            title="결과 저장",
            initialfile=f"{base}.txt",
            defaultextension=".txt",
            filetypes=[(fmt.upper(), f"*.{fmt}") for fmt in OUTPUT_FORMATS],
        )
        if not out_path:
            return
        try:
            write_result(self.last_result, out_path)
        except Exception as exc:
            messagebox.showerror("저장 실패", str(exc))
            return
        self.status_label.config(text=f"저장 완료: {out_path}")

    # ---------- worker thread ----------

    def _run_worker(self, **params) -> None:
        # Runs off the main thread; communicates only through the event queue
        try:
            result = self.transcriber.transcribe(
                **params,
                on_status=lambda msg: self.events.put(("status", msg)),
                on_progress=lambda ratio: self.events.put(("progress", ratio)),
            )
            self.events.put(("done", result))
        except Exception as exc:
            self.events.put(("error", f"{type(exc).__name__}: {exc}"))

    # ---------- main-thread event handling ----------

    def _poll_events(self) -> None:
        # Drain all pending events, then reschedule
        try:
            while True:
                kind, payload = self.events.get_nowait()
                self._handle_event(kind, payload)
        except queue.Empty:
            pass
        if self.worker is not None and self.worker.is_alive():
            self._update_elapsed()
        self.root.after(POLL_INTERVAL_MS, self._poll_events)

    def _handle_event(self, kind: str, payload: object) -> None:
        if kind == "status":
            self._status_base = str(payload)
            self._update_elapsed()
        elif kind == "progress":
            self.progress["value"] = float(payload) * 100
        elif kind == "done":
            self.last_result = payload  # type: ignore[assignment]
            self.progress["value"] = 100
            self._render_result()
            elapsed = time.monotonic() - self.started_at
            lang = self.last_result.get("language", "?")  # type: ignore[union-attr]
            self.status_label.config(text=f"변환 완료 ({elapsed:.1f}초, 감지 언어: {lang})", foreground="green")
            self._set_busy(False)
        elif kind == "error":
            self.status_label.config(text="오류 발생", foreground="red")
            self._set_busy(False)
            messagebox.showerror("오류 발생", f"작업 도중 오류가 발생했습니다:\n{payload}")

    def _update_elapsed(self) -> None:
        elapsed = time.monotonic() - self.started_at
        self.status_label.config(text=f"{self._status_base}  {elapsed:.0f}초 경과", foreground="blue")

    def _render_result(self) -> None:
        # Show either plain text or timestamped segments from the cached result
        if not self.last_result:
            return
        if self.timestamp_var.get():
            text = format_segments(self.last_result)
        else:
            text = str(self.last_result.get("text", "")).strip()
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, text)

    def _set_busy(self, busy: bool) -> None:
        self.run_button.config(state="disabled" if busy else "normal")
        has_result = (not busy) and bool(self.last_result)
        self.copy_button.config(state="normal" if has_result else "disabled")
        self.save_button.config(state="normal" if has_result else "disabled")
        if busy:
            self._status_base = "준비 중..."

    def _on_close(self) -> None:
        # Whisper cannot be interrupted mid-run, so warn before discarding work
        if self.worker is not None and self.worker.is_alive():
            if not messagebox.askyesno("종료 확인", "변환이 진행 중입니다. 종료하시겠습니까?"):
                return
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    TranscriberApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
