import csv
from pathlib import Path

from musica.audio.chords import (
    HumanizationConfig,
    generate_chord_midi_files,
    generate_chord_wav_files,
    write_generated_audio_manifest,
)


def test_generate_midi_respects_max_files(tmp_path: Path) -> None:
    generated = generate_chord_midi_files(
        tmp_path / "midi",
        max_files=2,
        humanization=HumanizationConfig(enabled=False),
    )

    assert len(generated) == 2
    assert all(item.path.exists() for item in generated)


def test_generate_wav_writes_limited_dataset_and_manifest(tmp_path: Path) -> None:
    output_dir = tmp_path / "wav"
    generated = generate_chord_wav_files(
        output_dir,
        duration=0.1,
        max_files=3,
        sample_rate=8000,
        renderer="pretty-midi",
        humanization=HumanizationConfig(enabled=False),
    )
    manifest_path = output_dir / "manifest.csv"
    write_generated_audio_manifest(generated, manifest_path)

    assert len(generated) == 3
    assert all(item.path.exists() for item in generated)

    rows = list(csv.DictReader(manifest_path.open()))
    assert len(rows) == 3
    assert rows[0]["root_note"] == "C"
    assert rows[0]["quality"] == "maj"
