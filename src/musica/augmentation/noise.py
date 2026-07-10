"""Noise download and audio augmentation utilities."""

from __future__ import annotations

import csv
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

import numpy as np

from musica.audio.io import (
    list_wav_files,
    match_sample_rate,
    peak_limit,
    read_audio,
    write_audio,
)
from musica.modeling.config import MusicaConfig

DEFAULT_CONFIG = MusicaConfig()


@dataclass(frozen=True)
class NoiseSource:
    name: str
    url: str
    category: str = ""
    license: str = ""
    attribution: str = ""


@dataclass(frozen=True)
class DownloadedNoise:
    path: Path
    name: str
    url: str
    category: str
    license: str
    attribution: str


@dataclass(frozen=True)
class NoisyAudio:
    path: Path
    source_path: Path
    noise_path: Path
    snr_db: float
    sample_rate: int
    noise_start_sample: int


DEFAULT_INTERNET_NOISES: tuple[NoiseSource, ...] = (
    NoiseSource(
        name="esc10_rain_1-17367-A-10",
        url=(
            "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/"
            "audio/1-17367-A-10.wav"
        ),
        category="rain",
        license="CC BY 3.0 via ESC-10 subset; original clip CC0",
        attribution=(
            "ESC-50 clip 1-17367-A; derived from 'unlocki door to mod rain.wav' "
            "by cognito perceptu on Freesound"
        ),
    ),
    NoiseSource(
        name="esc10_sea_waves_1-28135-A-11",
        url=(
            "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/"
            "audio/1-28135-A-11.wav"
        ),
        category="sea_waves",
        license="CC BY 3.0 via ESC-10 subset; original clip CC-BY",
        attribution=(
            "ESC-50 clip 1-28135-A; derived from 'Branding_kort.wav' "
            "by HerbertBoland on Freesound"
        ),
    ),
    NoiseSource(
        name="esc10_clock_tick_1-21934-A-38",
        url=(
            "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/"
            "audio/1-21934-A-38.wav"
        ),
        category="clock_tick",
        license="CC BY 3.0 via ESC-10 subset; original clip CC-BY",
        attribution=(
            "ESC-50 clip 1-21934-A; derived from '20060810.peters.clock.01.flac' "
            "by dobroide on Freesound"
        ),
    ),
)


def parse_float_list(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("at least one numeric value is required")
    return values


def noise_sources_from_urls(urls: list[str]) -> list[NoiseSource]:
    return [
        NoiseSource(
            name=safe_stem_from_url(url),
            url=url,
            category="custom",
            license="unknown",
            attribution="custom URL supplied by user",
        )
        for url in urls
    ]


def download_noise_wavs(
    output_dir: Path,
    *,
    sources: list[NoiseSource] | tuple[NoiseSource, ...] = DEFAULT_INTERNET_NOISES,
    overwrite: bool = False,
    manifest_path: Path | None = None,
) -> list[DownloadedNoise]:
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[DownloadedNoise] = []

    for source in sources:
        filename = f"{safe_filename(source.name)}.wav"
        path = output_dir / filename
        if overwrite or not path.exists():
            download_url(source.url, path)
        downloaded.append(
            DownloadedNoise(
                path=path,
                name=source.name,
                url=source.url,
                category=source.category,
                license=source.license,
                attribution=source.attribution,
            )
        )

    write_noise_download_manifest(
        downloaded,
        manifest_path or output_dir / "manifest.csv",
    )
    return downloaded


def download_url(url: str, output_path: Path) -> None:
    request = Request(url, headers={"User-Agent": "musica-dataset-builder/0.1"})
    with urlopen(request) as response, output_path.open("wb") as file:
        shutil.copyfileobj(response, file)


def augment_wav_dataset_with_noise(
    input_dir: Path,
    noise_dir: Path,
    output_dir: Path,
    *,
    snrs_db: tuple[float, ...] = DEFAULT_CONFIG.noise_snrs_db,
    mode: str = DEFAULT_CONFIG.noise_mode,
    max_files: int | None = None,
    random_seed: int = DEFAULT_CONFIG.seed,
    manifest_path: Path | None = None,
) -> list[NoisyAudio]:
    if mode not in {"cycle", "all"}:
        raise ValueError("mode must be 'cycle' or 'all'")

    source_files = list_wav_files(input_dir)
    noise_files = list_wav_files(noise_dir)
    if max_files is not None:
        source_files = source_files[:max_files]
    if not source_files:
        raise FileNotFoundError(f"No WAV files found in {input_dir}")
    if not noise_files:
        raise FileNotFoundError(f"No noise WAV files found in {noise_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[NoisyAudio] = []

    for source_index, source_path in enumerate(source_files):
        signal, sample_rate = read_audio(source_path)
        for noise_path in noise_files_for_source(noise_files, mode, source_index):
            noise = prepare_noise_for_signal(noise_path, signal, sample_rate)

            for snr_db in snrs_db:
                rng = rng_for_mix(random_seed, source_path, noise_path, snr_db)
                noise_segment, start_sample = select_noise_segment(
                    noise,
                    len(signal),
                    rng,
                )
                mixed = mix_at_snr(signal, noise_segment, snr_db)
                output_path = noisy_output_path(
                    input_dir,
                    output_dir,
                    source_path,
                    noise_path,
                    snr_db,
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                write_audio(output_path, mixed, sample_rate)
                generated.append(
                    NoisyAudio(
                        path=output_path,
                        source_path=source_path,
                        noise_path=noise_path,
                        snr_db=snr_db,
                        sample_rate=sample_rate,
                        noise_start_sample=start_sample,
                    )
                )

    write_noisy_audio_manifest(
        generated,
        manifest_path or output_dir / "manifest.csv",
    )
    return generated


def noise_files_for_source(
    noise_files: list[Path],
    mode: str,
    source_index: int,
) -> list[Path]:
    if mode == "all":
        return noise_files
    return [noise_files[source_index % len(noise_files)]]


def prepare_noise_for_signal(
    noise_path: Path,
    signal: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    noise, noise_sample_rate = read_audio(noise_path)
    noise = match_sample_rate(noise, noise_sample_rate, sample_rate)
    return match_channels(noise, signal)


def match_channels(noise: np.ndarray, signal: np.ndarray) -> np.ndarray:
    signal_channels = signal.shape[1]
    noise_channels = noise.shape[1]
    if noise_channels == signal_channels:
        return noise
    if signal_channels == 1:
        return np.mean(noise, axis=1, keepdims=True)
    if noise_channels == 1:
        return np.repeat(noise, signal_channels, axis=1)
    mono_noise = np.mean(noise, axis=1, keepdims=True)
    return np.repeat(mono_noise, signal_channels, axis=1)


def select_noise_segment(
    noise: np.ndarray,
    target_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    if len(noise) < target_samples:
        repeats = int(np.ceil(target_samples / len(noise)))
        noise = np.tile(noise, (repeats, 1))

    max_start = len(noise) - target_samples
    start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
    return noise[start: start + target_samples], start


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    signal_rms = rms(signal)
    noise_rms = rms(noise)
    if signal_rms == 0 or noise_rms == 0:
        return signal

    target_noise_rms = signal_rms / (10 ** (snr_db / 20.0))
    mixed = signal + (noise * (target_noise_rms / noise_rms))
    return peak_limit(mixed).astype(np.float32)


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))


def rng_for_mix(
    random_seed: int,
    source_path: Path,
    noise_path: Path,
    snr_db: float,
) -> np.random.Generator:
    seed_material = f"{random_seed}:{source_path}:{noise_path}:{snr_db}"
    digest = hashlib.sha256(seed_material.encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def noisy_output_path(
    input_dir: Path,
    output_dir: Path,
    source_path: Path,
    noise_path: Path,
    snr_db: float,
) -> Path:
    relative_source = source_path.relative_to(input_dir)
    snr_label = format_snr_label(snr_db)
    filename = f"{source_path.stem}_noise-{noise_path.stem}_snr{snr_label}.wav"
    return output_dir / relative_source.parent / filename


def format_snr_label(snr_db: float) -> str:
    return str(snr_db).replace("-", "m").replace(".", "p")


def safe_stem_from_url(url: str) -> str:
    path = unquote(urlparse(url).path)
    stem = Path(path).stem
    return stem or hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]


def safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def write_noise_download_manifest(
    downloaded: list[DownloadedNoise],
    manifest_path: Path,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["path", "name", "category", "license", "attribution", "url"],
        )
        writer.writeheader()
        for item in downloaded:
            writer.writerow(
                {
                    "path": str(item.path),
                    "name": item.name,
                    "category": item.category,
                    "license": item.license,
                    "attribution": item.attribution,
                    "url": item.url,
                }
            )


def write_noisy_audio_manifest(
    generated: list[NoisyAudio],
    manifest_path: Path,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "path",
                "source_path",
                "noise_path",
                "snr_db",
                "sample_rate",
                "noise_start_sample",
            ],
        )
        writer.writeheader()
        for item in generated:
            writer.writerow(
                {
                    "path": str(item.path),
                    "source_path": str(item.source_path),
                    "noise_path": str(item.noise_path),
                    "snr_db": item.snr_db,
                    "sample_rate": item.sample_rate,
                    "noise_start_sample": item.noise_start_sample,
                }
            )
