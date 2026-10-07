"""python -m annofix --images DIR --annotations DIR [--labels labels.yaml] [--annotator NAME]"""
from __future__ import annotations

import annofix  # noqa: F401  (thread cap before numpy)

import argparse
import os
import sys
from pathlib import Path


def add_sam_arguments(ap: argparse.ArgumentParser) -> None:
    from .sam_backend import MODELS  # a plain dict; torch is only imported by Sam2Backend()

    ap.add_argument("--sam-model", choices=sorted(MODELS),
                    help="SAM 2.1 size (default: base_plus on a GPU, small on CPU)")
    ap.add_argument("--device", help="cuda or cpu (default: cuda if available)")
    ap.add_argument("--checkpoint", type=Path, help="local SAM 2.1 .pt instead of a Hugging Face download")
    ap.add_argument("--no-sam", action="store_true", help="start without SAM (brush and eraser only)")


def sam_factory_from_args(args):
    """None with --no-sam; otherwise a factory the SAM worker thread calls on first SAM use."""
    if args.no_sam:
        return None

    def sam_factory():
        from .sam_backend import Sam2Backend

        return Sam2Backend(args.sam_model, args.device, args.checkpoint)

    return sam_factory


def run_window(session, ann_dir: Path, sam_factory) -> int:
    """Open the editor on a ready Session and run until the window closes."""
    from PyQt6.QtWidgets import QApplication

    from .ui.main_window import MainWindow, install_excepthook

    app = QApplication.instance() or QApplication(sys.argv[:1])
    Path(ann_dir).mkdir(parents=True, exist_ok=True)
    install_excepthook(session, Path(ann_dir) / "annofix-errors.log")
    try:
        window = MainWindow(session, sam_factory=sam_factory)
    except ValueError as e:  # no image in the folder could be opened
        print(f"annofix: {e}", file=sys.stderr)
        return 1
    window.resize(1500, 900)
    window.show()
    code = app.exec()
    if window.sam_still_running:  # never wait on, or kill, a SAM thread still running Python
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    return code


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m annofix",
                                 description="Fix object/part/material segmentation annotations.")
    ap.add_argument("--images", required=True, type=Path, help="folder of colour images")
    ap.add_argument("--annotations", required=True, type=Path, help="annotations folder (created if missing)")
    ap.add_argument("--labels", type=Path, default=Path("labels.yaml"), help="materials and object classes")
    ap.add_argument("--annotator", help="your name, recorded in each annotation you save")
    add_sam_arguments(ap)
    args = ap.parse_args(argv)

    from .config import load_labels
    from .session import Session

    try:
        labels = load_labels(args.labels)
        session = Session(args.images, args.annotations, labels, annotator=args.annotator)
    except (OSError, ValueError) as e:
        print(f"annofix: {e}", file=sys.stderr)
        return 1
    return run_window(session, args.annotations, sam_factory_from_args(args))


if __name__ == "__main__":
    sys.exit(main())
