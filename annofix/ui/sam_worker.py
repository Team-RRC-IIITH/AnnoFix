"""SAM off the GUI thread. The SamWorker lives in its own QThread: it builds the backend on
first use (importing torch there, not at startup), embeds an image only when asked, and
computes only the newest prediction request. It never touches the Session or widgets, and
every exception becomes a `failed` signal (an exception escaping a worker slot would reach
the GUI excepthook from the wrong thread). The SamController is the GUI-thread glue: it
listens to the Session, sends images/requests, and hands results back."""
from __future__ import annotations

import time
from typing import Callable

from PyQt6.QtCore import QObject, QThread, QTimer, pyqtSignal, pyqtSlot


class SamWorker(QObject):
    loaded = pyqtSignal(str)                                       # "<model> on <device>"
    result = pyqtSignal(int, int, object, object, object, float)  # token, seq, masks, scores, logits, s
    failed = pyqtSignal(str, bool)                                 # message, fatal (model never loaded)

    def __init__(self, factory: Callable):
        super().__init__()
        self._factory = factory
        self._backend = None
        self._image_token: int | None = None
        self._pending_image = None
        self._latest = None
        self._scheduled = False
        self._dead = False

    @pyqtSlot(int, object)
    def set_image(self, token: int, image) -> None:
        self._pending_image = (token, image)
        self._schedule()

    @pyqtSlot(object)
    def predict(self, request) -> None:
        self._latest = request  # a newer request replaces one still waiting
        self._schedule()

    def _schedule(self) -> None:
        if not self._scheduled:
            self._scheduled = True
            QTimer.singleShot(0, self._run)

    def _run(self) -> None:
        self._scheduled = False
        if self._dead:
            return
        try:
            if self._backend is None:
                self._backend = self._factory()
                self.loaded.emit(f"{self._backend.model_name} on {self._backend.device}")
            if self._pending_image is not None:
                token, image = self._pending_image
                self._pending_image = None
                self._backend.set_image(image)
                self._image_token = token
            if self._latest is not None:
                req, self._latest = self._latest, None
                if req.token != self._image_token:
                    return  # the image changed after this request was made
                t = time.perf_counter()
                masks, scores, logits = self._backend.predict(req.points, req.labels, req.box,
                                                              req.prev_logits, req.multimask)
                self.result.emit(req.token, req.seq, masks, scores, logits, time.perf_counter() - t)
        except Exception as e:  # noqa: BLE001 - must never escape the worker thread
            fatal = self._backend is None
            self._dead = fatal
            self.failed.emit(f"{type(e).__name__}: {e}", fatal)


class SamController(QObject):
    status_changed = pyqtSignal(str)
    _send_image = pyqtSignal(int, object)
    _send_request = pyqtSignal(object)

    def __init__(self, session, factory: Callable, parent=None):
        super().__init__(parent)
        self.session = session
        self.status = "SAM: not loaded (press R to use it)"
        self._sent_token: int | None = None
        self._thread = QThread()
        self._worker = SamWorker(factory)
        self._worker.moveToThread(self._thread)
        self._send_image.connect(self._worker.set_image)
        self._send_request.connect(self._worker.predict)
        self._worker.loaded.connect(lambda what: self._set_status(f"SAM ready: {what}"))
        self._worker.result.connect(self._on_result)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()
        session.sam_enabled = True
        session.subscribe(self._on_change)

    def _set_status(self, text: str) -> None:
        self.status = text
        self.status_changed.emit(text)

    def _image(self) -> None:
        s = self.session
        if self.status.startswith("SAM: not loaded"):
            self._set_status("SAM: loading the model (the first time can take a while)…")
        self._send_image.emit(s.image_token, s.image)
        self._sent_token = s.image_token

    def _on_change(self, kind: str) -> None:
        s = self.session
        if not s.sam_enabled:
            return
        if kind == "image":
            self._sent_token = None
            if s.tool == "sam":
                self._image()  # keep the embedding ready while working with SAM
        elif kind == "view" and s.tool == "sam" and self._sent_token != s.image_token:
            self._image()      # warm up as soon as the SAM tool is chosen
        elif kind == "sam_request":
            request = s.take_sam_request()
            if request is None:
                return
            if self._sent_token != request.token:
                self._image()
            if not self.status.startswith("SAM: loading"):
                self._set_status("SAM: thinking…")
            self._send_request.emit(request)

    def _on_result(self, token, seq, masks, scores, logits, seconds) -> None:
        if self.session.sam_receive(token, seq, masks, scores, logits):
            self._set_status(f"SAM ready ({seconds * 1000:.0f} ms)")

    def _on_failed(self, message: str, fatal: bool) -> None:
        s = self.session
        if fatal:
            s.sam_enabled = False
            s.sam_cancel()
            if s.tool == "sam":
                s.set_tool("select")
            self._set_status(f"SAM unavailable: {message}")
        else:
            self._set_status(f"SAM error: {message}")
        s.add_message(self.status)

    def shutdown(self) -> None:
        if not self._thread.isRunning():
            return
        self._thread.quit()
        if not self._thread.wait(3000):  # e.g. a first-time checkpoint download in progress
            self._thread.terminate()
            self._thread.wait()
