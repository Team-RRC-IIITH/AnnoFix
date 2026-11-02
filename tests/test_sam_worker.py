import time

import numpy as np
import pytest
from helpers import LABELS, make_dataset

from annofix.sam_backend import FakeBackend
from annofix.sam_edit import SamRequest
from annofix.session import Session
from annofix.ui.sam_worker import SamController


@pytest.fixture
def make(tmp_path):
    controllers = []

    def build(backend=None, factory=None):
        images, ann_dir = make_dataset(tmp_path)
        s = Session(images, ann_dir, LABELS)
        made = []

        def default_factory():
            made.append(1)
            return backend

        c = SamController(s, factory or default_factory)
        controllers.append(c)
        s.open(0)
        return s, c, made

    yield build
    for c in controllers:
        c.shutdown()


def predicts(backend):
    return [call for call in backend.calls if call[0] == "predict"]


def test_window_opens_without_loading_the_model(make, qtbot):
    backend = FakeBackend()
    s, c, made = make(backend)
    qtbot.wait(150)
    assert s.sam_enabled and made == [] and backend.calls == [] and c.status.startswith("SAM: not loaded")


def test_sam_round_trip_through_the_worker_thread(make, qtbot):
    backend = FakeBackend()
    s, c, made = make(backend)
    s.select_at(10, 10)
    s.set_tool("sam")
    qtbot.waitUntil(lambda: s.sam is not None and s.sam.mask is not None, timeout=3000)
    assert made == [1] and backend.calls[0] == ("set_image", (40, 60))
    assert c.status.startswith("SAM ready")


def test_only_the_newest_request_is_computed(make, qtbot):
    backend = FakeBackend(delay=0.3)
    s, c, _ = make(backend)
    s.select_at(10, 10)
    s.set_tool("sam")
    qtbot.waitUntil(lambda: len(predicts(backend)) == 1, timeout=3000)  # busy with the seed
    for x in (20, 21, 22):
        s.sam_click(x, 12, False)
    qtbot.waitUntil(lambda: s.sam.received_seq == 4, timeout=5000)
    assert len(predicts(backend)) == 2  # the seed, then only the newest refinement


def test_result_for_an_image_you_left_is_dropped(make, qtbot):
    backend = FakeBackend(delay=0.3)
    s, c, _ = make(backend)
    s.select_at(10, 10)
    s.set_tool("sam")
    qtbot.waitUntil(lambda: len(predicts(backend)) == 1, timeout=3000)
    s.step(1)
    qtbot.wait(600)
    assert s.stem == "b" and s.sam is None


def test_model_that_fails_to_load_disables_sam_but_not_the_editor(make, qtbot):
    def broken():
        raise RuntimeError("no network for the checkpoint download")

    s, c, _ = make(factory=broken)
    s.select_at(10, 10)
    s.set_tool("sam")
    qtbot.waitUntil(lambda: not s.sam_enabled, timeout=3000)
    assert s.tool == "select" and s.sam is None and "no network" in c.status
    assert s.new_part() is not None and s.tool == "brush"  # N falls back to the brush


def test_prediction_error_keeps_sam_enabled(make, qtbot):
    class Flaky(FakeBackend):
        def predict(self, *a, **k):
            raise RuntimeError("CUDA out of memory")

    s, c, _ = make(Flaky())
    s.select_at(10, 10)
    s.set_tool("sam")
    qtbot.waitUntil(lambda: "out of memory" in c.status, timeout=3000)
    assert s.sam_enabled and s.tool == "sam"


def test_shutdown_while_predicting_does_not_hang(make, qtbot):
    backend = FakeBackend(delay=0.5)
    s, c, _ = make(backend)
    s.select_at(10, 10)
    s.set_tool("sam")
    qtbot.waitUntil(lambda: len(predicts(backend)) == 1, timeout=3000)
    t = time.perf_counter()
    c.shutdown()
    assert time.perf_counter() - t < 2.5
    c.shutdown()  # idempotent


def test_answer_for_a_cancelled_fix_is_dropped_through_the_thread(make, qtbot):
    backend = FakeBackend(delay=0.4)
    s, c, _ = make(backend)
    s.select_at(10, 10)
    s.set_tool("sam")
    qtbot.waitUntil(lambda: len(predicts(backend)) == 1, timeout=3000)  # busy with fix A
    s.sam_cancel()
    s.sam_click(10, 25, True)                                            # fix B on part 5
    qtbot.waitUntil(lambda: s.sam.mask is not None, timeout=5000)
    assert s.sam.mask[25, 10] and not s.sam.mask[10, 15]                 # B's disc, not A's


def test_a_failed_embedding_is_retried(make, qtbot):
    class FlakyEmbed(FakeBackend):
        failures = 1

        def set_image(self, image):
            if self.failures:
                self.failures -= 1
                raise RuntimeError("CUDA out of memory (embedding)")
            super().set_image(image)

    s, c, _ = make(FlakyEmbed())
    s.select_at(10, 10)
    s.set_tool("sam")
    qtbot.waitUntil(lambda: "out of memory" in c.status, timeout=3000)
    s.sam_click(15, 12, False)  # the user simply tries again
    qtbot.waitUntil(lambda: s.sam.mask is not None, timeout=3000)


def test_closing_during_a_slow_model_load_does_not_hang(tmp_path):
    import os
    import subprocess
    import sys
    import textwrap

    script = tmp_path / "close_during_load.py"
    script.write_text(textwrap.dedent('''
        import os, sys, time
        sys.path.insert(0, sys.argv[1])
        from pathlib import Path
        from PyQt6.QtWidgets import QApplication
        from helpers import LABELS, make_dataset
        from annofix.sam_backend import FakeBackend
        from annofix.session import Session
        from annofix.ui.sam_worker import SamController

        app = QApplication([])
        images, ann_dir = make_dataset(Path(sys.argv[2]))
        s = Session(images, ann_dir, LABELS)

        def slow_pure_python_load():
            t = time.perf_counter()
            while time.perf_counter() - t < 5:
                pass
            return FakeBackend()

        c = SamController(s, slow_pure_python_load)
        s.open(0)
        s.select_at(10, 10)
        s.set_tool("sam")
        for _ in range(40):
            app.processEvents()
            time.sleep(0.01)
        t = time.perf_counter()
        stopped = c.shutdown()
        print(f"shutdown returned {stopped} after {time.perf_counter() - t:.1f}s", flush=True)
        os._exit(0)
    '''))
    tests_dir = str(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run([sys.executable, str(script), tests_dir, str(tmp_path)], capture_output=True,
                         text=True, timeout=30, env={**os.environ, "QT_QPA_PLATFORM": "offscreen"})
    assert out.returncode == 0 and "shutdown returned False" in out.stdout, out.stdout + out.stderr[-500:]


def test_sam_failure_reason_is_given_when_r_is_pressed_again(make, qtbot):
    def broken():
        raise RuntimeError("no network for the checkpoint download")

    s, c, _ = make(factory=broken)
    s.select_at(10, 10)
    s.set_tool("sam")
    qtbot.waitUntil(lambda: not s.sam_enabled, timeout=3000)
    s.set_tool("sam")
    assert "no network" in s.messages[-1]


def test_status_is_reset_when_leaving_an_image_mid_request(make, qtbot):
    backend = FakeBackend(delay=0.3)
    s, c, _ = make(backend)
    s.select_at(10, 10)
    s.set_tool("sam")
    qtbot.waitUntil(lambda: s.sam.mask is not None, timeout=3000)  # model loaded, first answer in
    s.sam_click(20, 12, False)                                       # a second request...
    assert c.status == "SAM: thinking…"
    qtbot.waitUntil(lambda: len(predicts(backend)) == 2, timeout=3000)
    s.step(1)                                                        # ...and leave while it runs
    qtbot.wait(600)
    assert "thinking" not in c.status


def test_auto_clean_batch_runs_every_job_through_the_worker(make, qtbot):
    backend = FakeBackend()
    s, c, made = make(backend)
    statuses = []
    c.status_changed.connect(statuses.append)
    s.start_auto_clean()
    qtbot.waitUntil(lambda: s.auto_clean is not None, timeout=5000)
    jobs = len([k for k in s.auto_clean.report if k[0] in ("object", "part")])
    assert len(predicts(backend)) >= 2 and not s.clean_pending
    assert any("refining" in t for t in statuses) and "auto-clean" in c.status
    assert jobs >= 2


def test_auto_clean_answer_for_a_left_image_is_dropped(make, qtbot):
    backend = FakeBackend(delay=0.3)
    s, c, made = make(backend)
    s.start_auto_clean()
    s.step(1)
    qtbot.wait(1500)
    assert s.auto_clean is None and not s.clean_pending


def test_a_stale_click_never_strands_a_queued_auto_clean(qtbot):
    from annofix import refine
    from annofix.ui.sam_worker import SamWorker

    w = SamWorker(FakeBackend)
    got = []
    w.batch_result.connect(lambda token, batch, results, t: got.append((token, batch, len(results))))
    image = np.full((40, 60, 3), 100, np.uint8)
    mask = np.zeros((40, 60), bool)
    mask[10:30, 10:40] = True
    w.set_image(2, image)
    w._run()
    w._latest = SamRequest(1, 5, np.array([[5, 5]], np.float32), np.array([1], np.int32), None, None, True)  # old image
    w._batch = (2, 7, [refine.make_job(("object", 1), mask, np.zeros_like(mask))])
    w._run()
    assert got == [(2, 7, 1)]


def test_a_failing_job_is_smoothed_not_lost(qtbot):
    from annofix import refine
    from annofix.ui.sam_worker import SamWorker

    class Broken(FakeBackend):
        def predict(self, *a, **k):
            raise RuntimeError("CUDA out of memory")

    w = SamWorker(Broken)
    got = []
    w.batch_result.connect(lambda token, batch, results, t: got.append(results))
    mask = np.zeros((40, 60), bool)
    mask[10:30, 10:40] = True
    w.set_image(1, np.full((40, 60, 3), 100, np.uint8))
    w._batch = (1, 1, [refine.make_job(("object", 1), mask, np.zeros_like(mask))])
    w._run()
    assert len(got) == 1 and got[0][0].mask is None


def test_status_says_why_sam_is_on_the_cpu(make, qtbot):
    backend = FakeBackend()
    backend.note = "the GPU is full (another program holds it), so SAM runs on the CPU"
    s, c, _ = make(backend)
    s.select_at(10, 10)
    s.set_tool("sam")
    qtbot.waitUntil(lambda: s.sam is not None and s.sam.mask is not None, timeout=3000)
    assert any("GPU is full" in m for m in s.messages)  # a message: the status line moves on
