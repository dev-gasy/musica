import csv
from pathlib import Path

import numpy as np
import soundfile as sf

from musica.augmentation.noise import augment_wav_dataset_with_noise
from musica.augmentation.realism import augment_wav_dataset_with_realism
from musica.augmentation.transpose import augment_wav_dataset_with_transposition


def write_wav(path: Path, *, frequency: float = 440.0, sample_rate: int = 8000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    duration = 0.2
    samples = int(sample_rate * duration)
    time = np.arange(samples, dtype=np.float32) / sample_rate
    audio = 0.2 * np.sin(2.0 * np.pi * frequency * time)
    sf.write(path, audio, sample_rate)


def test_augment_noise_writes_audio_and_manifest(tmp_path: Path) -> None:
    clean_path = tmp_path / "clean" / "C" / "C_maj.wav"
    noise_path = tmp_path / "noise" / "room.wav"
    output_dir = tmp_path / "noisy"
    write_wav(clean_path)
    write_wav(noise_path, frequency=120.0)

    generated = augment_wav_dataset_with_noise(
        tmp_path / "clean",
        tmp_path / "noise",
        output_dir,
        snrs_db=(12.0,),
        max_files=1,
    )

    assert len(generated) == 1
    assert generated[0].path.exists()
    rows = list(csv.DictReader((output_dir / "manifest.csv").open()))
    assert rows[0]["source_path"] == str(clean_path)
    assert rows[0]["noise_path"] == str(noise_path)


def test_augment_realistic_writes_audio_and_manifest(tmp_path: Path) -> None:
    clean_path = tmp_path / "clean" / "C" / "C_maj.wav"
    output_dir = tmp_path / "realistic"
    write_wav(clean_path)

    generated = augment_wav_dataset_with_realism(
        tmp_path / "clean",
        output_dir,
        variants=1,
        sample_rates=(8000,),
        max_files=1,
        keep_duration=True,
    )

    assert len(generated) == 1
    assert generated[0].path.exists()
    rows = list(csv.DictReader((output_dir / "manifest.csv").open()))
    assert rows[0]["source_path"] == str(clean_path)
    assert rows[0]["keep_duration"] == "True"


def test_augment_transpose_writes_relabelled_audio_and_manifest(tmp_path: Path) -> None:
    clean_path = tmp_path / "clean" / "C" / "C_maj.wav"
    output_dir = tmp_path / "transposed"
    write_wav(clean_path)

    generated = augment_wav_dataset_with_transposition(
        tmp_path / "clean",
        output_dir,
        semitones=(2,),
        qualities=("maj",),
        max_files=1,
    )

    assert len(generated) == 1
    assert generated[0].path.exists()
    assert generated[0].root_note == "D"
    rows = list(csv.DictReader((output_dir / "manifest.csv").open()))
    assert rows[0]["source_path"] == str(clean_path)
    assert rows[0]["root_note"] == "D"
    assert rows[0]["quality"] == "maj"
    assert rows[0]["label"] == "D_maj"
    assert rows[0]["semitones"] == "2"
