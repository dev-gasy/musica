"""Compatibility entry point for the training workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from musica.config import MusicaConfig
from musica.workflows.audio_bootstrap import ensure_training_audio_dataset
from musica.workflows.training import run_pipeline

FLUIDSYNTH_INSTALL_STEPS = """\
Execution order:
  1. Run reproducible setup: uv run musica setup-env
  2. Install Python deps: uv sync --extra dev
  3. Optional FluidSynth install:
     macOS:   brew install fluid-synth
     Linux:   sudo apt install fluidsynth
              sudo dnf install fluidsynth
              sudo pacman -S fluidsynth
     Windows: choco install fluidsynth
  4. Optional SoundFont: uv run musica download-soundfont
  5. Generate WAV if needed: uv run python main.py --audio-only
  6. Train/evaluate/predict: uv run python main.py

The default renderer is read from musica.toml. With renderer=auto, FluidSynth is
used only when both the binary and SoundFont are available; otherwise PrettyMIDI
is used so the project can still generate WAV files.
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare chord WAV files, then run the Musica training workflow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=FLUIDSYNTH_INSTALL_STEPS,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "musica.toml",
        help="Path to musica.toml.",
    )
    parser.add_argument(
        "--renderer",
        choices=("auto", "pretty-midi", "fluidsynth"),
        help="Override the audio renderer used when WAV files must be generated.",
    )
    parser.add_argument(
        "--max-audio-files",
        type=int,
        help="Limit generated WAV files, useful for smoke tests.",
    )
    parser.add_argument(
        "--force-audio-bootstrap",
        action="store_true",
        help="Regenerate clean chord WAV files even when the dataset already has WAV files.",
    )
    parser.add_argument(
        "--skip-audio-bootstrap",
        action="store_true",
        help="Do not generate WAV files before training.",
    )
    parser.add_argument(
        "--audio-only",
        action="store_true",
        help="Prepare/generate WAV files and stop before training.",
    )
    args = parser.parse_args()

    if args.max_audio_files is not None and args.max_audio_files < 1:
        parser.error("--max-audio-files must be at least 1")
    if args.skip_audio_bootstrap and args.audio_only:
        parser.error("--audio-only cannot be combined with --skip-audio-bootstrap")

    config = MusicaConfig.load(args.config)
    if not args.skip_audio_bootstrap:
        result = ensure_training_audio_dataset(
            config,
            PROJECT_ROOT,
            renderer=args.renderer,
            max_files=args.max_audio_files,
            force=args.force_audio_bootstrap,
        )
        if result.generated:
            print(
                f"Audio bootstrap: generated {result.generated_count} WAV files "
                f"in {result.output_dir.relative_to(PROJECT_ROOT)} "
                f"using {result.renderer}"
            )
            print(f"Audio manifest: {result.manifest_path.relative_to(PROJECT_ROOT)}")
        else:
            print(
                f"Audio bootstrap: found {result.wav_count} WAV files "
                f"under {result.dataset_dir.relative_to(PROJECT_ROOT)}"
            )

    if args.audio_only:
        return

    run_pipeline(project_root=PROJECT_ROOT, config_path=args.config, config=config)


if __name__ == "__main__":
    main()
