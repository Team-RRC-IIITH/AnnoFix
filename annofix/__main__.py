"""python -m annofix --images DIR --annotations DIR [--labels labels.yaml] [--annotator NAME]"""
from __future__ import annotations

import annofix  # noqa: F401  (thread cap before numpy)

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m annofix",
                                 description="Fix object/part/material segmentation annotations.")
    ap.add_argument("--images", required=True, type=Path, help="folder of colour images")
    ap.add_argument("--annotations", required=True, type=Path, help="annotations folder (created if missing)")
    ap.add_argument("--labels", type=Path, default=Path("labels.yaml"), help="materials and object classes")
    ap.add_argument("--annotator", help="your name, recorded in each annotation you save")
    args = ap.parse_args(argv)

    from .config import load_labels
    from .session import Session

    try:
        labels = load_labels(args.labels)
        session = Session(args.images, args.annotations, labels, annotator=args.annotator)
    except (OSError, ValueError) as e:
        print(f"annofix: {e}", file=sys.stderr)
        return 1

    from PyQt6.QtWidgets import QApplication

    from .ui.main_window import MainWindow, install_excepthook

    app = QApplication(sys.argv[:1])
    args.annotations.mkdir(parents=True, exist_ok=True)
    install_excepthook(session, args.annotations / "annofix-errors.log")
    try:
        window = MainWindow(session)
    except ValueError as e:  # no image in the folder could be opened
        print(f"annofix: {e}", file=sys.stderr)
        return 1
    window.resize(1500, 900)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
