"""Generate a clean chord WAV dataset with the package generator."""

from __future__ import annotations

import argparse
from pathlib import Path

from musica.audio.chords import generate_chord_wav_files, write_generated_audio_manifest
from musica.config import MusicaConfig


def main() -> None:
    config = MusicaConfig.load()
    parser = argparse.ArgumentParser(description="Generate clean chord WAV files.")
    parser.add_argument("--output-dir", type=Path, default=config.clean_output_dir)
    parser.add_argument("--duration", type=float, default=config.chord_duration)
    parser.add_argument("--sample-rate", type=int, default=config.chord_sample_rate)
    parser.add_argument("--soundfont", type=Path, default=config.soundfont_path)
    parser.add_argument("--renderer", choices=("auto", "fluidsynth", "pretty-midi"), default=config.renderer)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--manifest-path", type=Path)
    args = parser.parse_args()

    generated = generate_chord_wav_files(
        args.output_dir,
        duration=args.duration,
        sample_rate=args.sample_rate,
        renderer=args.renderer,
        soundfont_path=args.soundfont,
        octave_offsets=config.octave_offsets,
        velocities=config.velocities,
        instrument_programs=config.instrument_programs,
        max_files=args.max_files,
    )
    manifest_path = args.manifest_path or args.output_dir / "manifest.csv"
    write_generated_audio_manifest(generated, manifest_path)
    renderer = generated[0].renderer if generated else args.renderer
    print(f"Generated {len(generated)} WAV files in {args.output_dir} using {renderer}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
