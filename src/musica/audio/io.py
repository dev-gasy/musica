"""Shared WAV audio I/O helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    return audio, sample_rate


def write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    import soundfile as sf

    sf.write(path, audio, sample_rate, subtype="PCM_16")


def list_wav_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*.wav") if path.is_file())


def match_sample_rate(audio: np.ndarray, current_rate: int, target_rate: int) -> np.ndarray:
    if current_rate == target_rate:
        return audio

    import librosa

    channels = [
        librosa.resample(
            audio[:, channel],
            orig_sr=current_rate,
            target_sr=target_rate,
        )
        for channel in range(audio.shape[1])
    ]
    return np.stack(channels, axis=1).astype(np.float32)


def fit_audio_length(audio: np.ndarray, target_samples: int) -> np.ndarray:
    if len(audio) > target_samples:
        return audio[:target_samples]
    if len(audio) < target_samples:
        pad_width = [(0, target_samples - len(audio))]
        pad_width.extend((0, 0) for _ in audio.shape[1:])
        return np.pad(audio, pad_width)
    return audio


def peak_limit(audio: np.ndarray, peak: float = 0.99) -> np.ndarray:
    current_peak = float(np.max(np.abs(audio)))
    if current_peak > peak:
        return (audio * (peak / current_peak)).astype(np.float32)
    return audio
