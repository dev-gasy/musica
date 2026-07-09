"""Pitch-shift augmentation for chord WAV datasets."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from musica.audio.chords import safe_key_name
from musica.audio.manifest import parse_chord_filename
from musica.modeling.config import MusicaConfig

ROOTS: tuple[str, ...] = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
DEFAULT_CONFIG = MusicaConfig()


@dataclass(frozen=True)
class TransposedAudio:
    path: Path
    source_path: Path
    original_root_note: str
    root_note: str
    quality: str
    semitones: int
    sample_rate: int


def augment_wav_dataset_with_transposition(
        input_dir: Path,
        output_dir: Path,
        *,
        semitones: tuple[int, ...] = DEFAULT_CONFIG.transpose_semitones,
        roots: tuple[str, ...] = DEFAULT_CONFIG.transpose_roots,
        qualities: tuple[str, ...] = DEFAULT_CONFIG.transpose_qualities,
        max_files: int | None = None,
        manifest_path: Path | None = None,
) -> list[TransposedAudio]:
    validate_transposition_args(semitones, roots, qualities)
    source_files = sorted(path for path in input_dir.rglob("*.wav") if path.is_file())
    selected_files = [
        path
        for path in source_files
        if chord_matches_filters(path, roots=roots, qualities=qualities)
    ]
    if max_files is not None:
        selected_files = selected_files[:max_files]
    if not selected_files:
        raise FileNotFoundError(f"No matching chord WAV files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[TransposedAudio] = []
    for source_path in selected_files:
        parsed = parse_chord_filename(source_path)
        if parsed is None:
            continue
        original_root, quality = parsed
        audio, sample_rate = read_audio(source_path)
        for shift in semitones:
            root_note = transpose_root(original_root, shift)
            shifted = pitch_shift_audio(audio, sample_rate, shift)
            output_path = transposed_output_path(
                output_dir,
                source_path,
                root_note=root_note,
                quality=quality,
                semitones=shift,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            write_audio(output_path, shifted, sample_rate)
            generated.append(
                TransposedAudio(
                    path=output_path,
                    source_path=source_path,
                    original_root_note=original_root,
                    root_note=root_note,
                    quality=quality,
                    semitones=shift,
                    sample_rate=sample_rate,
                )
            )

    write_transposed_audio_manifest(
        generated,
        manifest_path or output_dir / "manifest.csv",
    )
    return generated


def validate_transposition_args(
        semitones: tuple[int, ...],
        roots: tuple[str, ...],
        qualities: tuple[str, ...],
) -> None:
    if not semitones:
        raise ValueError("at least one semitone shift is required")
    if any(shift == 0 for shift in semitones):
        raise ValueError("semitone shifts must not include 0")
    if any(abs(shift) > 24 for shift in semitones):
        raise ValueError("semitone shifts must be between -24 and 24")
    unknown_roots = sorted(set(roots) - set(ROOTS))
    if unknown_roots:
        raise ValueError(f"unsupported roots: {', '.join(unknown_roots)}")
    unknown_qualities = sorted(set(qualities) - {"maj", "min", "dim"})
    if unknown_qualities:
        raise ValueError(f"unsupported qualities: {', '.join(unknown_qualities)}")


def chord_matches_filters(
        path: Path,
        *,
        roots: tuple[str, ...],
        qualities: tuple[str, ...],
) -> bool:
    parsed = parse_chord_filename(path)
    if parsed is None:
        return False
    root_note, quality = parsed
    return (not roots or root_note in roots) and (not qualities or quality in qualities)


def transpose_root(root_note: str, semitones: int) -> str:
    root_index = ROOTS.index(root_note)
    return ROOTS[(root_index + semitones) % len(ROOTS)]


def pitch_shift_audio(audio: np.ndarray, sample_rate: int, semitones: int) -> np.ndarray:
    import librosa

    shifted_channels = [
        librosa.effects.pitch_shift(
            y=audio[:, channel],
            sr=sample_rate,
            n_steps=semitones,
        )
        for channel in range(audio.shape[1])
    ]
    shifted = np.stack(shifted_channels, axis=1).astype(np.float32)
    return fit_audio_length(shifted, len(audio))


def fit_audio_length(audio: np.ndarray, target_samples: int) -> np.ndarray:
    if len(audio) > target_samples:
        return audio[:target_samples]
    if len(audio) < target_samples:
        padding = np.zeros((target_samples - len(audio), audio.shape[1]), dtype=np.float32)
        return np.vstack((audio, padding))
    return audio


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    return audio, sample_rate


def write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    import soundfile as sf

    peak = float(np.max(np.abs(audio)))
    if peak > 0.99:
        audio = audio * (0.99 / peak)
    sf.write(path, audio, sample_rate, subtype="PCM_16")


def transposed_output_path(
        output_dir: Path,
        source_path: Path,
        *,
        root_note: str,
        quality: str,
        semitones: int,
) -> Path:
    shift_label = f"p{semitones}" if semitones > 0 else f"m{abs(semitones)}"
    filename = (
        f"{safe_key_name(root_note)}_{quality}_from_{source_path.stem}"
        f"_transpose_{shift_label}.wav"
    )
    return output_dir / safe_key_name(root_note) / filename


def write_transposed_audio_manifest(
        generated: list[TransposedAudio],
        manifest_path: Path,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "path",
                "source_path",
                "root_note",
                "quality",
                "label",
                "original_root_note",
                "original_quality",
                "semitones",
                "sample_rate",
            ],
        )
        writer.writeheader()
        for item in generated:
            writer.writerow(
                {
                    "path": str(item.path),
                    "source_path": str(item.source_path),
                    "root_note": item.root_note,
                    "quality": item.quality,
                    "label": f"{item.root_note}_{item.quality}",
                    "original_root_note": item.original_root_note,
                    "original_quality": item.quality,
                    "semitones": item.semitones,
                    "sample_rate": item.sample_rate,
                }
            )
