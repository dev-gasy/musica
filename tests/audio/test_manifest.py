import csv
from pathlib import Path

import numpy as np
import soundfile as sf

from musica.audio.manifest import ManifestSource, build_audio_manifest, default_manifest_sources
from musica.config import MusicaConfig


def write_tone(path: Path, *, sample_rate: int = 8000, duration: float = 0.1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = int(sample_rate * duration)
    audio = np.full(samples, 0.1, dtype=np.float32)
    sf.write(path, audio, sample_rate)


def test_build_audio_manifest_from_generated_source(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio" / "C_maj.wav"
    manifest_path = tmp_path / "generated_manifest.csv"
    output_path = tmp_path / "audio_manifest.csv"
    write_tone(audio_path)

    with manifest_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["path", "root_note", "quality", "duration"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "path": audio_path,
                "root_note": "C",
                "quality": "maj",
                "duration": 0.1,
            }
        )

    summary = build_audio_manifest(
        output_path,
        sources=(
            ManifestSource(
                dataset="synthetic_clean",
                manifest_path=manifest_path,
                source_type="generated",
            ),
        ),
    )

    assert summary.rows_written == 1
    assert summary.rows_skipped == 0
    assert summary.dataset_counts == {"synthetic_clean": 1}

    rows = list(csv.DictReader(output_path.open()))
    assert rows[0]["label"] == "C_maj"
    assert rows[0]["sample_rate"] == "8000"


def test_default_manifest_sources_use_feature_first_config_paths() -> None:
    config = MusicaConfig(
        clean_output_dir=Path("custom/clean"),
        noisy_output_dir=Path("custom/noisy"),
        realistic_output_dir=Path("custom/realistic"),
        transposed_output_dir=Path("custom/transposed"),
        recorded_audio_dir=Path("custom/recorded"),
    )

    sources = default_manifest_sources(config)

    assert sources[0].manifest_path == Path("custom/clean/manifest.csv")
    assert sources[1].manifest_path == Path("custom/noisy/manifest.csv")
    assert sources[2].manifest_path == Path("custom/realistic/manifest.csv")
    assert sources[3].manifest_path == Path("custom/transposed/manifest.csv")
    assert sources[4].directory == Path("custom/recorded")
