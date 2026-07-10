"""Command-line entry point for Musica audio dataset construction."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from musica.audio.chords import (
    HumanizationConfig,
    generate_chord_midi_files,
    generate_chord_wav_files,
    write_generated_audio_manifest,
)
from musica.audio.manifest import build_audio_manifest
from musica.augmentation.noise import (
    DEFAULT_INTERNET_NOISES,
    augment_wav_dataset_with_noise,
    download_url,
    download_noise_wavs,
    noise_sources_from_urls,
    parse_float_list,
)
from musica.augmentation.realism import augment_wav_dataset_with_realism
from musica.augmentation.transpose import augment_wav_dataset_with_transposition
from musica.config import MusicaConfig

CommandHandler = Callable[[argparse.Namespace, MusicaConfig], None]
DEFAULT_SOUNDFONT_URL = "https://musical-artifacts.com/artifacts/738/FluidR3_GM.sf2"


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def parse_durations(value: str) -> tuple[float, ...]:
    try:
        durations = parse_float_list(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if any(duration <= 0 for duration in durations):
        raise argparse.ArgumentTypeError("durations must be positive")
    return durations


def parse_snrs(value: str) -> tuple[float, ...]:
    try:
        return parse_float_list(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_sample_rates(value: str) -> tuple[int, ...]:
    try:
        sample_rates = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("sample rates must be integers") from error
    if not sample_rates:
        raise argparse.ArgumentTypeError("at least one sample rate is required")
    if any(sample_rate < 8000 for sample_rate in sample_rates):
        raise argparse.ArgumentTypeError("sample rates must be at least 8000 Hz")
    return sample_rates


def parse_int_list(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("values must be integers") from error
    if not values:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return values


def parse_str_list(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one value is required")
    return values


def add_dataset_variation_args(parser: argparse.ArgumentParser, config: MusicaConfig) -> None:
    parser.add_argument(
        "--duration",
        type=positive_float,
        default=config.chord_duration,
        help="Chord duration in seconds.",
    )
    parser.add_argument(
        "--durations",
        type=parse_durations,
        help=(
            "Comma-separated chord durations in seconds for dataset variation "
            "(example: 1.0,1.5,2.0). Overrides --duration."
        ),
    )
    parser.add_argument(
        "--repetitions",
        type=positive_int,
        default=config.repetitions,
        help="Number of reproducible humanized takes per chord variation.",
    )
    parser.add_argument(
        "--max-files",
        type=positive_int,
        help="Limit the number of generated files for smoke tests.",
    )


def add_humanization_args(parser: argparse.ArgumentParser, config: MusicaConfig) -> None:
    parser.add_argument(
        "--no-humanize",
        action="store_false",
        dest="humanize",
        default=config.humanize,
        help="Generate exact block chords without timing, velocity, or voicing variation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.seed,
        help="Seed used for reproducible humanized voicings.",
    )
    parser.add_argument(
        "--humanize-onset-ms",
        type=float,
        default=config.humanize_onset_ms,
        help="Maximum random note onset offset in milliseconds.",
    )
    parser.add_argument(
        "--velocity-jitter",
        type=int,
        default=config.velocity_jitter,
        help="Maximum per-note velocity variation around the requested velocity.",
    )


def humanization_config(args: argparse.Namespace) -> HumanizationConfig:
    return HumanizationConfig(
        enabled=args.humanize,
        onset_spread_seconds=max(0.0, args.humanize_onset_ms) / 1000.0,
        velocity_jitter=max(0, args.velocity_jitter),
    )


def build_parser(config: MusicaConfig | None = None) -> argparse.ArgumentParser:
    config = config or MusicaConfig.load()
    parser = argparse.ArgumentParser(prog="musica")
    parser.set_defaults(config=config)
    subparsers = parser.add_subparsers(dest="command", required=True)

    midi_parser = subparsers.add_parser(
        "generate-midi",
        help="Generate annotated chord MIDI files.",
    )
    midi_parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.midi_output_dir,
        help="Directory where generated .mid files are written.",
    )
    add_dataset_variation_args(midi_parser, config)
    add_humanization_args(midi_parser, config)

    wav_parser = subparsers.add_parser(
        "generate-wav",
        help="Generate annotated chord WAV files.",
    )
    wav_parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.clean_output_dir,
        help="Directory where generated .wav files are written by key.",
    )
    add_dataset_variation_args(wav_parser, config)
    wav_parser.add_argument(
        "--sample-rate",
        type=positive_int,
        default=config.chord_sample_rate,
        help="Output WAV sample rate.",
    )
    wav_parser.add_argument(
        "--renderer",
        choices=("auto", "pretty-midi", "fluidsynth"),
        default=config.renderer,
        help="Audio renderer. Auto uses fluidsynth when available.",
    )
    wav_parser.add_argument(
        "--soundfont",
        type=Path,
        default=config.soundfont_path,
        help="SoundFont path used by the fluidsynth renderer.",
    )
    wav_parser.add_argument(
        "--manifest-path",
        type=Path,
        help="CSV manifest path. Defaults to manifest.csv inside the output directory.",
    )
    wav_parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Skip writing the generated WAV annotation manifest.",
    )
    add_humanization_args(wav_parser, config)

    noise_download_parser = subparsers.add_parser(
        "download-noises",
        help="Download noise WAV files for augmentation.",
    )
    noise_download_parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.noise_download_dir,
        help="Directory where downloaded noise WAV files are written.",
    )
    noise_download_parser.add_argument(
        "--url",
        action="append",
        default=[],
        help="Noise WAV URL to download. Repeat for multiple files.",
    )
    noise_download_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload files that already exist.",
    )
    noise_download_parser.add_argument(
        "--manifest-path",
        type=Path,
        help="CSV manifest path. Defaults to manifest.csv inside the output directory.",
    )

    soundfont_download_parser = subparsers.add_parser(
        "download-soundfont",
        help="Download the FluidR3 GM SoundFont used by the fluidsynth renderer.",
    )
    soundfont_download_parser.add_argument(
        "--output-path",
        type=Path,
        default=config.soundfont_path,
        help="Path where the .sf2 SoundFont is written.",
    )
    soundfont_download_parser.add_argument(
        "--url",
        default=DEFAULT_SOUNDFONT_URL,
        help="SoundFont URL to download.",
    )
    soundfont_download_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload the SoundFont if the output file already exists.",
    )

    assets_download_parser = subparsers.add_parser(
        "download-assets",
        help="Download external assets: the FluidR3 GM SoundFont and noise WAV files.",
    )
    assets_download_parser.add_argument(
        "--soundfont-output-path",
        type=Path,
        default=config.soundfont_path,
        help="Path where the .sf2 SoundFont is written.",
    )
    assets_download_parser.add_argument(
        "--soundfont-url",
        default=DEFAULT_SOUNDFONT_URL,
        help="SoundFont URL to download.",
    )
    assets_download_parser.add_argument(
        "--noise-output-dir",
        type=Path,
        default=config.noise_download_dir,
        help="Directory where downloaded noise WAV files are written.",
    )
    assets_download_parser.add_argument(
        "--noise-url",
        action="append",
        default=[],
        help="Noise WAV URL to download. Repeat for multiple files.",
    )
    assets_download_parser.add_argument(
        "--noise-manifest-path",
        type=Path,
        help="CSV manifest path for downloaded noises. Defaults to manifest.csv inside the noise output directory.",
    )
    assets_download_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload assets that already exist.",
    )
    assets_download_parser.add_argument(
        "--skip-soundfont",
        action="store_true",
        help="Do not download the SoundFont.",
    )
    assets_download_parser.add_argument(
        "--skip-noises",
        action="store_true",
        help="Do not download noise WAV files.",
    )

    noise_augment_parser = subparsers.add_parser(
        "augment-noise",
        help="Mix generated chord WAV files with noise.",
    )
    noise_augment_parser.add_argument(
        "--input-dir",
        type=Path,
        default=config.clean_output_dir,
        help="Directory containing clean chord WAV files.",
    )
    noise_augment_parser.add_argument(
        "--noise-dir",
        type=Path,
        default=config.noise_download_dir,
        help="Directory containing noise WAV files.",
    )
    noise_augment_parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.noisy_output_dir,
        help="Directory where noisy chord WAV files are written.",
    )
    noise_augment_parser.add_argument(
        "--snrs-db",
        type=parse_snrs,
        default=config.noise_snrs_db,
        help="Comma-separated signal-to-noise ratios in dB.",
    )
    noise_augment_parser.add_argument(
        "--mode",
        choices=("cycle", "all"),
        default=config.noise_mode,
        help="Use one cycled noise per chord, or all noise files per chord.",
    )
    noise_augment_parser.add_argument(
        "--max-files",
        type=positive_int,
        help="Limit the number of clean chord files processed.",
    )
    noise_augment_parser.add_argument(
        "--seed",
        type=non_negative_int,
        default=config.seed,
        help="Seed used for deterministic noise segment offsets.",
    )
    noise_augment_parser.add_argument(
        "--manifest-path",
        type=Path,
        help="CSV manifest path. Defaults to manifest.csv inside the output directory.",
    )

    realism_augment_parser = subparsers.add_parser(
        "augment-realistic",
        help="Generate realistic variants of chord WAV files.",
    )
    realism_augment_parser.add_argument(
        "--input-dir",
        type=Path,
        default=config.clean_output_dir,
        help="Directory containing clean chord WAV files.",
    )
    realism_augment_parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.realistic_output_dir,
        help="Directory where realistic chord WAV files are written.",
    )
    realism_augment_parser.add_argument(
        "--variants",
        type=positive_int,
        default=config.realism_variants,
        help="Number of deterministic realistic variants generated per input WAV.",
    )
    realism_augment_parser.add_argument(
        "--sample-rates",
        type=parse_sample_rates,
        default=config.realism_sample_rates,
        help="Comma-separated output sample rates.",
    )
    realism_augment_parser.add_argument(
        "--max-files",
        type=positive_int,
        help="Limit the number of clean chord files processed.",
    )
    realism_augment_parser.add_argument(
        "--seed",
        type=non_negative_int,
        default=config.seed,
        help="Seed used for deterministic realism parameters.",
    )
    realism_augment_parser.add_argument(
        "--keep-duration",
        action="store_true",
        default=config.realism_keep_duration,
        help="Crop or pad augmented audio back to the original input duration.",
    )
    realism_augment_parser.add_argument(
        "--manifest-path",
        type=Path,
        help="CSV manifest path. Defaults to manifest.csv inside the output directory.",
    )

    transpose_augment_parser = subparsers.add_parser(
        "augment-transpose",
        help="Pitch-shift chord WAV files and relabel their roots.",
    )
    transpose_augment_parser.add_argument(
        "--input-dir",
        type=Path,
        default=config.clean_output_dir,
        help="Directory containing chord WAV files named like C_maj.wav.",
    )
    transpose_augment_parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.transposed_output_dir,
        help="Directory where transposed chord WAV files are written.",
    )
    transpose_augment_parser.add_argument(
        "--semitones",
        type=parse_int_list,
        default=config.transpose_semitones,
        help="Comma-separated non-zero semitone shifts, for example -5,7.",
    )
    transpose_augment_parser.add_argument(
        "--roots",
        type=parse_str_list,
        default=config.transpose_roots,
        help="Optional comma-separated roots to augment, for example C,F#,A#.",
    )
    transpose_augment_parser.add_argument(
        "--qualities",
        type=parse_str_list,
        default=config.transpose_qualities,
        help="Optional comma-separated qualities to augment: maj,min,dim.",
    )
    transpose_augment_parser.add_argument(
        "--max-files",
        type=positive_int,
        help="Limit the number of source chord files processed.",
    )
    transpose_augment_parser.add_argument(
        "--manifest-path",
        type=Path,
        help="CSV manifest path. Defaults to manifest.csv inside the output directory.",
    )

    manifest_parser = subparsers.add_parser(
        "build-manifest",
        help="Compile generated, derived, and recorded chord WAV manifests.",
    )
    manifest_parser.add_argument(
        "--output-path",
        type=Path,
        default=config.audio_manifest_path,
        help="Output unified audio manifest path.",
    )
    manifest_parser.add_argument(
        "--train-ratio",
        type=float,
        default=config.manifest_train_ratio,
        help="Fraction of rows assigned to the training split.",
    )
    manifest_parser.add_argument(
        "--val-ratio",
        type=float,
        default=config.manifest_val_ratio,
        help="Fraction of rows assigned to the validation split.",
    )
    manifest_parser.add_argument(
        "--test-ratio",
        type=float,
        default=config.manifest_test_ratio,
        help="Fraction of rows assigned to the test split.",
    )
    manifest_parser.add_argument(
        "--include-missing-audio",
        action="store_true",
        default=config.include_missing_audio,
        help="Keep manifest rows even when the referenced audio file is missing.",
    )

    return parser


def run_generate_midi(args: argparse.Namespace, config: MusicaConfig) -> None:
    generated = generate_chord_midi_files(
        args.output_dir,
        duration=args.duration,
        durations=args.durations,
        repetitions=args.repetitions,
        max_files=args.max_files,
        octave_offsets=config.octave_offsets,
        velocities=config.velocities,
        instrument_programs=config.instrument_programs,
        humanization=humanization_config(args),
        random_seed=args.seed,
    )
    print(f"Generated {len(generated)} MIDI files in {args.output_dir}")


def run_generate_wav(args: argparse.Namespace, config: MusicaConfig) -> None:
    generated = generate_chord_wav_files(
        args.output_dir,
        duration=args.duration,
        durations=args.durations,
        repetitions=args.repetitions,
        max_files=args.max_files,
        sample_rate=args.sample_rate,
        octave_offsets=config.octave_offsets,
        velocities=config.velocities,
        instrument_programs=config.instrument_programs,
        renderer=args.renderer,
        soundfont_path=args.soundfont,
        humanization=humanization_config(args),
        random_seed=args.seed,
    )
    renderer = generated[0].renderer if generated else args.renderer
    print(
        f"Generated {len(generated)} WAV files in {args.output_dir} "
        f"using {renderer}"
    )
    if not args.no_manifest:
        manifest_path = args.manifest_path or args.output_dir / "manifest.csv"
        write_generated_audio_manifest(generated, manifest_path)
        print(f"Wrote manifest to {manifest_path}")


def run_download_noises(args: argparse.Namespace, config: MusicaConfig) -> None:
    sources = noise_sources_from_urls(args.url) if args.url else DEFAULT_INTERNET_NOISES
    downloaded = download_noise_wavs(
        args.output_dir,
        sources=sources,
        overwrite=args.overwrite,
        manifest_path=args.manifest_path,
    )
    print(f"Downloaded {len(downloaded)} noise WAV files in {args.output_dir}")


def run_download_soundfont(args: argparse.Namespace, config: MusicaConfig) -> None:
    downloaded = download_soundfont(
        args.output_path,
        url=args.url,
        overwrite=args.overwrite,
    )
    if downloaded:
        print(f"Downloaded SoundFont to {args.output_path}")
    else:
        print(f"SoundFont already exists at {args.output_path}")


def download_soundfont(output_path: Path, *, url: str, overwrite: bool = False) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        return False

    download_url(url, output_path)
    return True


def run_download_assets(args: argparse.Namespace, config: MusicaConfig) -> None:
    if args.skip_soundfont and args.skip_noises:
        raise ValueError("At least one asset type must be enabled")

    if not args.skip_soundfont:
        downloaded_soundfont = download_soundfont(
            args.soundfont_output_path,
            url=args.soundfont_url,
            overwrite=args.overwrite,
        )
        if downloaded_soundfont:
            print(f"Downloaded SoundFont to {args.soundfont_output_path}")
        else:
            print(f"SoundFont already exists at {args.soundfont_output_path}")

    if not args.skip_noises:
        sources = (
            noise_sources_from_urls(args.noise_url)
            if args.noise_url
            else DEFAULT_INTERNET_NOISES
        )
        downloaded_noises = download_noise_wavs(
            args.noise_output_dir,
            sources=sources,
            overwrite=args.overwrite,
            manifest_path=args.noise_manifest_path,
        )
        print(
            f"Downloaded {len(downloaded_noises)} noise WAV files "
            f"in {args.noise_output_dir}"
        )


def run_augment_noise(args: argparse.Namespace, config: MusicaConfig) -> None:
    generated = augment_wav_dataset_with_noise(
        args.input_dir,
        args.noise_dir,
        args.output_dir,
        snrs_db=args.snrs_db,
        mode=args.mode,
        max_files=args.max_files,
        random_seed=args.seed,
        manifest_path=args.manifest_path,
    )
    print(f"Generated {len(generated)} noisy WAV files in {args.output_dir}")


def run_augment_realistic(args: argparse.Namespace, config: MusicaConfig) -> None:
    generated = augment_wav_dataset_with_realism(
        args.input_dir,
        args.output_dir,
        variants=args.variants,
        sample_rates=args.sample_rates,
        max_files=args.max_files,
        random_seed=args.seed,
        keep_duration=args.keep_duration,
        manifest_path=args.manifest_path,
    )
    print(f"Generated {len(generated)} realistic WAV files in {args.output_dir}")


def run_augment_transpose(args: argparse.Namespace, config: MusicaConfig) -> None:
    generated = augment_wav_dataset_with_transposition(
        args.input_dir,
        args.output_dir,
        semitones=args.semitones,
        roots=args.roots,
        qualities=args.qualities,
        max_files=args.max_files,
        manifest_path=args.manifest_path,
    )
    print(f"Generated {len(generated)} transposed WAV files in {args.output_dir}")


def run_build_manifest(args: argparse.Namespace, config: MusicaConfig) -> None:
    summary = build_audio_manifest(
        args.output_path,
        config=config,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        include_missing_audio=args.include_missing_audio,
    )
    print(f"Wrote {summary.rows_written} rows to {summary.output_path}")
    if summary.rows_skipped:
        print(f"Skipped {summary.rows_skipped} unavailable or invalid rows")
    print(f"Datasets: {summary.dataset_counts}")
    print(f"Splits: {summary.split_counts}")


COMMAND_HANDLERS: dict[str, CommandHandler] = {
    "generate-midi": run_generate_midi,
    "generate-wav": run_generate_wav,
    "download-noises": run_download_noises,
    "download-soundfont": run_download_soundfont,
    "download-assets": run_download_assets,
    "augment-noise": run_augment_noise,
    "augment-realistic": run_augment_realistic,
    "augment-transpose": run_augment_transpose,
    "build-manifest": run_build_manifest,
}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = args.config
    handler = COMMAND_HANDLERS[args.command]

    try:
        handler(args, config)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
