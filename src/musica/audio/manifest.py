"""Audio manifest compilation utilities for generated and recorded chord WAVs."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import soundfile as sf

from musica.config import MusicaConfig

MANIFEST_FIELDS: tuple[str, ...] = (
    "path",
    "source_path",
    "start_time",
    "duration",
    "root_note",
    "quality",
    "label",
    "dataset",
    "split",
    "is_segment",
    "track_id",
    "source_manifest",
    "sample_rate",
)

ROOT_ALIASES: dict[str, str] = {
    "C": "C",
    "B#": "C",
    "C#": "C#",
    "Csharp": "C#",
    "Db": "C#",
    "D": "D",
    "D#": "D#",
    "Dsharp": "D#",
    "Eb": "D#",
    "E": "E",
    "Fb": "E",
    "E#": "F",
    "F": "F",
    "F#": "F#",
    "Fsharp": "F#",
    "Gb": "F#",
    "G": "G",
    "G#": "G#",
    "Gsharp": "G#",
    "Ab": "G#",
    "A": "A",
    "A#": "A#",
    "Asharp": "A#",
    "Bb": "A#",
    "B": "B",
    "Cb": "B",
}

SUPPORTED_QUALITIES = {"maj", "min", "dim"}


@dataclass(frozen=True)
class ManifestSource:
    dataset: str
    manifest_path: Path | None = None
    directory: Path | None = None
    source_type: str = "generated"


@dataclass(frozen=True)
class CompiledManifestRow:
    path: Path
    source_path: Path
    start_time: float
    duration: float
    root_note: str
    quality: str
    label: str
    dataset: str
    split: str
    is_segment: bool
    track_id: str
    source_manifest: Path | None
    sample_rate: int


@dataclass(frozen=True)
class ManifestBuildSummary:
    output_path: Path
    rows_written: int
    rows_skipped: int
    dataset_counts: dict[str, int] = field(default_factory=dict)
    split_counts: dict[str, int] = field(default_factory=dict)


def default_manifest_sources(config: MusicaConfig | None = None) -> tuple[ManifestSource, ...]:
    config = config or MusicaConfig()
    return (
        ManifestSource(
            dataset="synthetic_clean",
            manifest_path=config.clean_output_dir / "manifest.csv",
            source_type="generated",
        ),
        ManifestSource(
            dataset="synthetic_noisy",
            manifest_path=config.noisy_output_dir / "manifest.csv",
            source_type="derived",
        ),
        ManifestSource(
            dataset="synthetic_realistic",
            manifest_path=config.realistic_output_dir / "manifest.csv",
            source_type="derived",
        ),
        ManifestSource(
            dataset="synthetic_transposed",
            manifest_path=config.transposed_output_dir / "manifest.csv",
            source_type="derived",
        ),
        ManifestSource(
            dataset="raw",
            directory=config.recorded_audio_dir,
            source_type="raw",
        ),
    )


def build_audio_manifest(
        output_path: Path,
        *,
        sources: Iterable[ManifestSource] | None = None,
        config: MusicaConfig | None = None,
        train_ratio: float = 0.80,
        val_ratio: float = 0.10,
        test_ratio: float = 0.10,
        include_missing_audio: bool = False,
) -> ManifestBuildSummary:
    validate_split_ratios(train_ratio, val_ratio, test_ratio)
    resolved_sources = tuple(sources or default_manifest_sources(config))
    rows: list[CompiledManifestRow] = []
    skipped = 0

    for source in resolved_sources:
        if source.source_type == "raw":
            source_rows, source_skipped = rows_from_raw_directory(
                source,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                include_missing_audio=include_missing_audio,
            )
        elif source.source_type in {"generated", "derived"}:
            source_rows, source_skipped = rows_from_generated_manifest(
                source,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                include_missing_audio=include_missing_audio,
            )
        else:
            raise ValueError(f"Unsupported manifest source type: {source.source_type}")
        rows.extend(source_rows)
        skipped += source_skipped

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row_to_dict(row))

    return ManifestBuildSummary(
        output_path=output_path,
        rows_written=len(rows),
        rows_skipped=skipped,
        dataset_counts=count_by(rows, "dataset"),
        split_counts=count_by(rows, "split"),
    )


def rows_from_raw_directory(
        source: ManifestSource,
        *,
        train_ratio: float,
        val_ratio: float,
        include_missing_audio: bool,
) -> tuple[list[CompiledManifestRow], int]:
    if source.directory is None or not source.directory.exists():
        return [], 1

    rows: list[CompiledManifestRow] = []
    skipped = 0
    for path in sorted(source.directory.glob("*.wav")):
        parsed = parse_chord_filename(path)
        if parsed is None:
            skipped += 1
            continue
        audio_info = read_audio_info(path, include_missing_audio=include_missing_audio)
        if audio_info is None:
            skipped += 1
            continue
        root_note, quality = parsed
        duration, sample_rate = audio_info
        rows.append(
            compiled_manifest_row(
                path=path,
                source_path=path,
                duration=duration,
                root_note=root_note,
                quality=quality,
                dataset=source.dataset,
                split_key=str(path),
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                source_manifest=None,
                sample_rate=sample_rate,
            )
        )
    return rows, skipped


def rows_from_generated_manifest(
        source: ManifestSource,
        *,
        train_ratio: float,
        val_ratio: float,
        include_missing_audio: bool,
) -> tuple[list[CompiledManifestRow], int]:
    if source.manifest_path is None or not source.manifest_path.exists():
        return [], 1

    rows: list[CompiledManifestRow] = []
    skipped = 0
    for raw_row in read_csv_rows(source.manifest_path):
        path_value = raw_row.get("path", "")
        if not path_value:
            skipped += 1
            continue
        path = Path(path_value)
        parsed = extract_chord_label(raw_row, path)
        if parsed is None:
            skipped += 1
            continue
        audio_info = read_audio_info(path, include_missing_audio=include_missing_audio)
        if audio_info is None:
            skipped += 1
            continue
        root_note, quality = parsed
        duration, sample_rate = audio_info
        source_path = Path(raw_row.get("source_path") or path_value)
        rows.append(
            compiled_manifest_row(
                path=path,
                source_path=source_path,
                duration=duration,
                root_note=root_note,
                quality=quality,
                dataset=source.dataset,
                split_key=str(source_path),
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                source_manifest=source.manifest_path,
                sample_rate=sample_rate,
            )
        )
    return rows, skipped


def compiled_manifest_row(
        *,
        path: Path,
        source_path: Path,
        duration: float,
        root_note: str,
        quality: str,
        dataset: str,
        split_key: str,
        train_ratio: float,
        val_ratio: float,
        source_manifest: Path | None,
        sample_rate: int,
) -> CompiledManifestRow:
    return CompiledManifestRow(
        path=path,
        source_path=source_path,
        start_time=0.0,
        duration=duration,
        root_note=root_note,
        quality=quality,
        label=f"{root_note}_{quality}",
        dataset=dataset,
        split=stable_split(split_key, train_ratio, val_ratio),
        is_segment=False,
        track_id=path.stem,
        source_manifest=source_manifest,
        sample_rate=sample_rate,
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def extract_chord_label(row: dict[str, str], path: Path) -> tuple[str, str] | None:
    root_note = normalize_root(row.get("root_note", ""))
    quality = row.get("quality", "")
    if root_note is not None and quality in SUPPORTED_QUALITIES:
        return root_note, quality
    return parse_chord_filename(path)


def parse_chord_filename(path: Path) -> tuple[str, str] | None:
    parts = path.stem.split("_")
    if len(parts) < 2:
        return None
    root_note = normalize_root(parts[0])
    quality = parts[1]
    if root_note is None or quality not in SUPPORTED_QUALITIES:
        return None
    return root_note, quality


def normalize_root(value: str) -> str | None:
    return ROOT_ALIASES.get(value.strip())


def read_audio_info(path: Path, *, include_missing_audio: bool) -> tuple[float, int] | None:
    if not path.exists():
        if include_missing_audio:
            return 0.0, 0
        return None
    info = sf.info(path)
    return float(info.frames) / float(info.samplerate), int(info.samplerate)


def stable_split(key: str, train_ratio: float, val_ratio: float) -> str:
    bucket = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + val_ratio:
        return "val"
    return "test"


def validate_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("split ratios must be non-negative")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("train, validation, and test ratios must sum to 1.0")


def count_by(rows: list[CompiledManifestRow], field_name: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(getattr(row, field_name))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def row_to_dict(row: CompiledManifestRow) -> dict[str, object]:
    return {
        "path": str(row.path),
        "source_path": str(row.source_path),
        "start_time": format_float(row.start_time),
        "duration": format_float(row.duration),
        "root_note": row.root_note,
        "quality": row.quality,
        "label": row.label,
        "dataset": row.dataset,
        "split": row.split,
        "is_segment": str(row.is_segment).lower(),
        "track_id": row.track_id,
        "source_manifest": str(row.source_manifest) if row.source_manifest else "",
        "sample_rate": row.sample_rate,
    }


def format_float(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")
