"""Compatibility entry point for the training workflow."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from musica.workflows.training import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Musica training workflow.")
    parser.parse_args()
    run_pipeline()


if __name__ == "__main__":
    main()
