"""Realistic audio augmentation for generated chord WAV datasets."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from musica.modeling.config import MusicaConfig

DEFAULT_CONFIG = MusicaConfig()


@dataclass(frozen=True)
class RealismProfile:
    gain_db: float
    pre_silence_ms: float
    post_silence_ms: float
    highpass_hz: float
    lowpass_hz: float
    reverb_wet: float
    reverb_decay: float
    compression_threshold_db: float
    compression_ratio: float
    saturation_drive: float
    stereo_width: float
    target_sample_rate: int
    output_channels: int


@dataclass(frozen=True)
class RealisticAudio:
    path: Path
    source_path: Path
    variant: int
    profile: RealismProfile


def augment_wav_dataset_with_realism(
        input_dir: Path,
        output_dir: Path,
        *,
        variants: int = DEFAULT_CONFIG.realism_variants,
        sample_rates: tuple[int, ...] = DEFAULT_CONFIG.realism_sample_rates,
        max_files: int | None = None,
        random_seed: int = DEFAULT_CONFIG.seed,
        keep_duration: bool = DEFAULT_CONFIG.realism_keep_duration,
        manifest_path: Path | None = None,
) -> list[RealisticAudio]:
    if variants < 1:
        raise ValueError("variants must be at least 1")
    if not sample_rates:
        raise ValueError("at least one sample rate is required")
    if any(sample_rate < 8000 for sample_rate in sample_rates):
        raise ValueError("sample rates must be at least 8000 Hz")

    source_files = sorted(path for path in input_dir.rglob("*.wav") if path.is_file())
    if max_files is not None:
        source_files = source_files[:max_files]
    if not source_files:
        raise FileNotFoundError(f"No WAV files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[RealisticAudio] = []

    for source_path in source_files:
        signal, sample_rate = read_audio(source_path)
        for variant in range(1, variants + 1):
            rng = rng_for_realism(random_seed, source_path, variant)
            profile = random_realism_profile(rng, sample_rates)
            augmented = apply_realism_profile(
                signal,
                sample_rate,
                profile,
                rng,
                keep_duration=keep_duration,
            )
            output_path = realistic_output_path(input_dir, output_dir, source_path, variant)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            write_audio(output_path, augmented, profile.target_sample_rate)
            generated.append(
                RealisticAudio(
                    path=output_path,
                    source_path=source_path,
                    variant=variant,
                    profile=profile,
                )
            )

    write_realistic_audio_manifest(
        generated,
        manifest_path or output_dir / "manifest.csv",
        keep_duration=keep_duration,
    )
    return generated


def random_realism_profile(
        rng: np.random.Generator,
        sample_rates: tuple[int, ...],
) -> RealismProfile:
    return RealismProfile(
        gain_db=float(rng.uniform(-9.0, 3.0)),
        pre_silence_ms=float(rng.uniform(0.0, 80.0)),
        post_silence_ms=float(rng.uniform(0.0, 120.0)),
        highpass_hz=float(rng.uniform(35.0, 160.0)),
        lowpass_hz=float(rng.uniform(6500.0, 18000.0)),
        reverb_wet=float(rng.uniform(0.02, 0.16)),
        reverb_decay=float(rng.uniform(0.10, 0.32)),
        compression_threshold_db=float(rng.uniform(-22.0, -10.0)),
        compression_ratio=float(rng.uniform(1.2, 3.0)),
        saturation_drive=float(rng.uniform(0.0, 0.28)),
        stereo_width=float(rng.uniform(0.75, 1.25)),
        target_sample_rate=int(rng.choice(sample_rates)),
        output_channels=int(rng.choice((1, 2), p=(0.25, 0.75))),
    )


def apply_realism_profile(
        signal: np.ndarray,
        sample_rate: int,
        profile: RealismProfile,
        rng: np.random.Generator,
        *,
        keep_duration: bool,
) -> np.ndarray:
    original_samples = len(signal)
    audio = signal.astype(np.float32, copy=True)
    audio = apply_gain(audio, profile.gain_db)
    audio = spectral_filter(audio, sample_rate, profile.highpass_hz, profile.lowpass_hz)
    audio = apply_static_compression(
        audio,
        threshold_db=profile.compression_threshold_db,
        ratio=profile.compression_ratio,
    )
    audio = apply_saturation(audio, profile.saturation_drive)
    audio = apply_synthetic_room_reverb(
        audio,
        sample_rate,
        wet=profile.reverb_wet,
        decay_seconds=profile.reverb_decay,
        rng=rng,
    )
    audio = apply_stereo_width(audio, profile.stereo_width)
    audio = convert_channels(audio, profile.output_channels)
    audio = add_silence(
        audio,
        sample_rate,
        pre_silence_ms=profile.pre_silence_ms,
        post_silence_ms=profile.post_silence_ms,
    )
    if keep_duration:
        audio = fit_audio_length(audio, original_samples)
    audio = match_sample_rate(audio, sample_rate, profile.target_sample_rate)
    return peak_limit(audio)


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    return audio, sample_rate


def write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    import soundfile as sf

    sf.write(path, audio, sample_rate, subtype="PCM_16")


def apply_gain(audio: np.ndarray, gain_db: float) -> np.ndarray:
    return audio * float(10 ** (gain_db / 20.0))


def spectral_filter(
        audio: np.ndarray,
        sample_rate: int,
        highpass_hz: float,
        lowpass_hz: float,
) -> np.ndarray:
    filtered_channels = []
    frequencies = np.fft.rfftfreq(len(audio), d=1.0 / sample_rate)
    response = np.ones_like(frequencies, dtype=np.float64)

    highpass_floor = max(20.0, highpass_hz)
    lowpass_ceiling = min(float(sample_rate) / 2.0, lowpass_hz)
    response *= smooth_highpass_response(frequencies, highpass_floor)
    response *= smooth_lowpass_response(frequencies, lowpass_ceiling)

    for channel in range(audio.shape[1]):
        spectrum = np.fft.rfft(audio[:, channel])
        filtered = np.fft.irfft(spectrum * response, n=len(audio))
        filtered_channels.append(filtered)
    return np.stack(filtered_channels, axis=1).astype(np.float32)


def smooth_highpass_response(frequencies: np.ndarray, cutoff_hz: float) -> np.ndarray:
    response = np.ones_like(frequencies, dtype=np.float64)
    transition_start = cutoff_hz * 0.5
    transition_end = cutoff_hz
    response[frequencies <= transition_start] = 0.0
    mask = (frequencies > transition_start) & (frequencies < transition_end)
    response[mask] = raised_cosine(
        (frequencies[mask] - transition_start) / (transition_end - transition_start)
    )
    return response


def smooth_lowpass_response(frequencies: np.ndarray, cutoff_hz: float) -> np.ndarray:
    response = np.ones_like(frequencies, dtype=np.float64)
    transition_start = cutoff_hz
    transition_end = cutoff_hz * 1.25
    response[frequencies >= transition_end] = 0.0
    mask = (frequencies > transition_start) & (frequencies < transition_end)
    response[mask] = 1.0 - raised_cosine(
        (frequencies[mask] - transition_start) / (transition_end - transition_start)
    )
    return response


def raised_cosine(position: np.ndarray) -> np.ndarray:
    return 0.5 - (0.5 * np.cos(np.pi * position))


def apply_static_compression(
        audio: np.ndarray,
        *,
        threshold_db: float,
        ratio: float,
) -> np.ndarray:
    threshold = float(10 ** (threshold_db / 20.0))
    magnitude = np.abs(audio)
    compressed = audio.copy()
    mask = magnitude > threshold
    if np.any(mask):
        compressed_magnitude = threshold + ((magnitude[mask] - threshold) / ratio)
        compressed[mask] = np.sign(audio[mask]) * compressed_magnitude
    return compressed.astype(np.float32)


def apply_saturation(audio: np.ndarray, drive: float) -> np.ndarray:
    if drive <= 0:
        return audio
    shaped = np.tanh(audio * (1.0 + drive * 6.0)) / np.tanh(1.0 + drive * 6.0)
    return ((audio * (1.0 - drive)) + (shaped * drive)).astype(np.float32)


def apply_synthetic_room_reverb(
        audio: np.ndarray,
        sample_rate: int,
        *,
        wet: float,
        decay_seconds: float,
        rng: np.random.Generator,
) -> np.ndarray:
    if wet <= 0:
        return audio

    impulse = synthetic_room_impulse(sample_rate, decay_seconds, rng)
    reverbed_channels = []
    for channel in range(audio.shape[1]):
        reverbed_channels.append(fft_convolve(audio[:, channel], impulse)[: len(audio)])
    reverbed = np.stack(reverbed_channels, axis=1).astype(np.float32)
    return ((audio * (1.0 - wet)) + (reverbed * wet)).astype(np.float32)


def synthetic_room_impulse(
        sample_rate: int,
        decay_seconds: float,
        rng: np.random.Generator,
) -> np.ndarray:
    impulse_samples = max(1, int(sample_rate * decay_seconds))
    times = np.arange(impulse_samples) / sample_rate
    decay = np.exp(-5.0 * times / decay_seconds)
    impulse = rng.normal(0.0, 0.08, impulse_samples) * decay
    impulse[0] += 1.0

    for delay_ms, gain in ((7.0, 0.18), (17.0, 0.11), (31.0, 0.08)):
        index = int(sample_rate * delay_ms / 1000.0)
        if index < impulse_samples:
            impulse[index] += gain * float(rng.uniform(0.7, 1.2))
    return impulse.astype(np.float32)


def fft_convolve(signal: np.ndarray, impulse: np.ndarray) -> np.ndarray:
    output_len = len(signal) + len(impulse) - 1
    fft_len = 1 << (output_len - 1).bit_length()
    spectrum = np.fft.rfft(signal, n=fft_len)
    impulse_spectrum = np.fft.rfft(impulse, n=fft_len)
    return np.fft.irfft(spectrum * impulse_spectrum, n=fft_len)[:output_len]


def apply_stereo_width(audio: np.ndarray, width: float) -> np.ndarray:
    if audio.shape[1] == 1:
        return np.repeat(audio, 2, axis=1)
    left = audio[:, 0]
    right = audio[:, 1]
    mid = (left + right) * 0.5
    side = (left - right) * 0.5 * width
    return np.stack((mid + side, mid - side), axis=1).astype(np.float32)


def convert_channels(audio: np.ndarray, output_channels: int) -> np.ndarray:
    if output_channels == audio.shape[1]:
        return audio
    if output_channels == 1:
        return np.mean(audio, axis=1, keepdims=True).astype(np.float32)
    if audio.shape[1] == 1:
        return np.repeat(audio, 2, axis=1).astype(np.float32)
    return audio[:, :output_channels].astype(np.float32)


def add_silence(
        audio: np.ndarray,
        sample_rate: int,
        *,
        pre_silence_ms: float,
        post_silence_ms: float,
) -> np.ndarray:
    pre_samples = int(round(sample_rate * pre_silence_ms / 1000.0))
    post_samples = int(round(sample_rate * post_silence_ms / 1000.0))
    if pre_samples == 0 and post_samples == 0:
        return audio
    pre = np.zeros((pre_samples, audio.shape[1]), dtype=np.float32)
    post = np.zeros((post_samples, audio.shape[1]), dtype=np.float32)
    return np.vstack((pre, audio, post))


def fit_audio_length(audio: np.ndarray, target_samples: int) -> np.ndarray:
    if len(audio) > target_samples:
        return audio[:target_samples]
    if len(audio) < target_samples:
        padding = np.zeros((target_samples - len(audio), audio.shape[1]), dtype=np.float32)
        return np.vstack((audio, padding))
    return audio


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


def peak_limit(audio: np.ndarray, peak: float = 0.99) -> np.ndarray:
    current_peak = float(np.max(np.abs(audio)))
    if current_peak > peak:
        return (audio * (peak / current_peak)).astype(np.float32)
    return audio.astype(np.float32)


def rng_for_realism(
        random_seed: int,
        source_path: Path,
        variant: int,
) -> np.random.Generator:
    seed_material = f"{random_seed}:{source_path}:{variant}"
    digest = hashlib.sha256(seed_material.encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def realistic_output_path(
        input_dir: Path,
        output_dir: Path,
        source_path: Path,
        variant: int,
) -> Path:
    relative_source = source_path.relative_to(input_dir)
    filename = f"{source_path.stem}_real_v{variant:03d}.wav"
    return output_dir / relative_source.parent / filename


def write_realistic_audio_manifest(
        generated: list[RealisticAudio],
        manifest_path: Path,
        *,
        keep_duration: bool,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "path",
                "source_path",
                "variant",
                "gain_db",
                "pre_silence_ms",
                "post_silence_ms",
                "highpass_hz",
                "lowpass_hz",
                "reverb_wet",
                "reverb_decay",
                "compression_threshold_db",
                "compression_ratio",
                "saturation_drive",
                "stereo_width",
                "target_sample_rate",
                "output_channels",
                "keep_duration",
            ],
        )
        writer.writeheader()
        for item in generated:
            profile = item.profile
            writer.writerow(
                {
                    "path": str(item.path),
                    "source_path": str(item.source_path),
                    "variant": item.variant,
                    "gain_db": profile.gain_db,
                    "pre_silence_ms": profile.pre_silence_ms,
                    "post_silence_ms": profile.post_silence_ms,
                    "highpass_hz": profile.highpass_hz,
                    "lowpass_hz": profile.lowpass_hz,
                    "reverb_wet": profile.reverb_wet,
                    "reverb_decay": profile.reverb_decay,
                    "compression_threshold_db": profile.compression_threshold_db,
                    "compression_ratio": profile.compression_ratio,
                    "saturation_drive": profile.saturation_drive,
                    "stereo_width": profile.stereo_width,
                    "target_sample_rate": profile.target_sample_rate,
                    "output_channels": profile.output_channels,
                    "keep_duration": keep_duration,
                }
            )
