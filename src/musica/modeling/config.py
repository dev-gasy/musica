"""Configuration loading and validation for model training."""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from musica.modeling.constants import ROOTS

LOGGER = logging.getLogger(__name__)

PATH_FIELDS = {
    "dataset_dir",
    "logs_dir",
    "legacy_model_path",
    "legacy_params_path",
    "example_audio_path",
    "midi_output_dir",
    "clean_output_dir",
    "noise_download_dir",
    "noisy_output_dir",
    "realistic_output_dir",
    "transposed_output_dir",
    "recorded_audio_dir",
    "audio_manifest_path",
    "soundfont_path",
}
FLOAT_TUPLE_FIELDS = {"noise_snrs_db"}
INT_TUPLE_FIELDS = {"octave_offsets", "velocities", "realism_sample_rates", "transpose_semitones"}
STR_TUPLE_FIELDS = {"transpose_roots", "transpose_qualities"}
CONFIG_SECTIONS = {
    "general": {"seed"},
    "training": {
        "epochs",
        "batch_size",
        "learning_rate",
        "force_retrain",
        "val_ratio",
        "test_ratio",
    },
    "features": {
        "sample_rate",
        "target_duration",
        "hop_length",
        "bins_per_octave",
        "n_chroma",
    },
    "prediction": {"top_k", "example_audio_path"},
    "paths": {
        "dataset_dir",
        "logs_dir",
        "legacy_model_path",
        "legacy_params_path",
        "midi_output_dir",
        "clean_output_dir",
        "noise_download_dir",
        "noisy_output_dir",
        "realistic_output_dir",
        "transposed_output_dir",
        "recorded_audio_dir",
        "audio_manifest_path",
    },
    "model": {"model_architecture"},
    "audio": {
        "chord_duration",
        "chord_sample_rate",
        "renderer",
        "soundfont_path",
        "instrument_programs",
        "repetitions",
        "octave_offsets",
        "velocities",
    },
    "humanization": {"humanize", "humanize_onset_ms", "velocity_jitter"},
    "noise": {"noise_snrs_db", "noise_mode"},
    "realism": {"realism_variants", "realism_sample_rates", "realism_keep_duration"},
    "transpose": {"transpose_semitones", "transpose_roots", "transpose_qualities"},
    "manifest": {
        "manifest_train_ratio",
        "manifest_val_ratio",
        "manifest_test_ratio",
        "include_missing_audio",
    },
    "callbacks": {
        "early_stopping_patience",
        "early_stopping_min_delta",
        "reduce_lr_factor",
        "reduce_lr_patience",
        "min_lr",
        "tensorboard_histogram_freq",
    },
}


@dataclass(frozen=True)
class MusicaConfig:
    seed: int = 0
    epochs: int = 60
    batch_size: int = 32
    learning_rate: float = 0.001
    force_retrain: bool = False
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    sample_rate: int = 22050
    target_duration: float = 1.5
    hop_length: int = 512
    bins_per_octave: int = 12
    n_chroma: int = 12
    top_k: int = 3
    dataset_dir: Path = Path("audio/chords")
    logs_dir: Path = Path("logs")
    legacy_model_path: Path = Path("logs/best_model.keras")
    legacy_params_path: Path = Path("logs/best_model.params.json")
    example_audio_path: Path = Path("audio/chords/clean/C/C_maj_piano_oct0_vel100.wav")
    model_architecture: str = "cnn_chords_chroma_cqt_v1"
    midi_output_dir: Path = Path("audio/chords/midi")
    clean_output_dir: Path = Path("audio/chords/clean")
    noise_download_dir: Path = Path("assets/noises/internet")
    noisy_output_dir: Path = Path("audio/chords/noisy")
    realistic_output_dir: Path = Path("audio/chords/realistic")
    transposed_output_dir: Path = Path("audio/chords/transposed")
    recorded_audio_dir: Path = Path("audio/chords/recorded")
    audio_manifest_path: Path = Path("audio/manifest.csv")
    chord_duration: float = 1.5
    chord_sample_rate: int = 44100
    renderer: str = "auto"
    soundfont_path: Path = Path("assets/soundfonts/FluidR3_GM.sf2")
    instrument_programs: dict[str, int] = field(default_factory=lambda: {
        "piano": 0,
        "guitar": 24,
        "synth_pad": 88,
    })
    repetitions: int = 1
    octave_offsets: tuple[int, ...] = (-1, 0, 1)
    velocities: tuple[int, ...] = (80, 100, 120)
    humanize: bool = True
    humanize_onset_ms: float = 35.0
    velocity_jitter: int = 8
    noise_snrs_db: tuple[float, ...] = (15.0,)
    noise_mode: str = "cycle"
    realism_variants: int = 2
    realism_sample_rates: tuple[int, ...] = (22050, 44100, 48000)
    realism_keep_duration: bool = False
    transpose_semitones: tuple[int, ...] = (-5, 7)
    transpose_roots: tuple[str, ...] = ()
    transpose_qualities: tuple[str, ...] = ()
    manifest_train_ratio: float = 0.80
    manifest_val_ratio: float = 0.10
    manifest_test_ratio: float = 0.10
    include_missing_audio: bool = False
    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 1e-4
    reduce_lr_factor: float = 0.3
    reduce_lr_patience: int = 4
    min_lr: float = 1e-6
    tensorboard_histogram_freq: int = 0

    @classmethod
    def load(cls, path: Path = Path("musica.toml")) -> "MusicaConfig":
        LOGGER.info("Chargement de la configuration: %s", path)
        data = tomllib.loads(path.read_text()) if path.exists() else {}
        known_fields = {field.name for field in fields(cls)}
        data = flatten_config_data(data, known_fields)
        unknown = sorted(set(data) - known_fields)
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {', '.join(unknown)}")
        config = cls(**coerce_config_data(data))
        config.validate()
        LOGGER.info(
            "Configuration chargee: epochs=%s, batch_size=%s, lr=%s, force_retrain=%s",
            config.epochs,
            config.batch_size,
            config.learning_rate,
            config.force_retrain,
        )
        return config

    def validate(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.sample_rate < 8000:
            raise ValueError("sample_rate must be at least 8000")
        if self.target_duration <= 0:
            raise ValueError("target_duration must be positive")
        if self.hop_length < 1:
            raise ValueError("hop_length must be at least 1")
        if self.bins_per_octave < 1:
            raise ValueError("bins_per_octave must be at least 1")
        if self.n_chroma != len(ROOTS):
            raise ValueError("n_chroma must stay at 12 for the current chord labels")
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if self.val_ratio <= 0 or self.test_ratio <= 0:
            raise ValueError("val_ratio and test_ratio must be positive")
        if self.val_ratio + self.test_ratio >= 1:
            raise ValueError("val_ratio + test_ratio must be below 1")
        if self.chord_duration <= 0:
            raise ValueError("chord_duration must be positive")
        if self.chord_sample_rate < 8000:
            raise ValueError("chord_sample_rate must be at least 8000")
        if self.renderer not in {"auto", "pretty-midi", "fluidsynth"}:
            raise ValueError("renderer must be one of: auto, pretty-midi, fluidsynth")
        if not self.instrument_programs:
            raise ValueError("instrument_programs must not be empty")
        if any(program < 0 or program > 127 for program in self.instrument_programs.values()):
            raise ValueError("instrument_programs values must be between 0 and 127")
        if self.repetitions < 1:
            raise ValueError("repetitions must be at least 1")
        if not self.octave_offsets:
            raise ValueError("octave_offsets must not be empty")
        if not self.velocities:
            raise ValueError("velocities must not be empty")
        if any(velocity < 1 or velocity > 127 for velocity in self.velocities):
            raise ValueError("velocities must be between 1 and 127")
        if self.humanize_onset_ms < 0:
            raise ValueError("humanize_onset_ms must be non-negative")
        if self.velocity_jitter < 0:
            raise ValueError("velocity_jitter must be non-negative")
        if not self.noise_snrs_db:
            raise ValueError("noise_snrs_db must not be empty")
        if self.noise_mode not in {"cycle", "all"}:
            raise ValueError("noise_mode must be one of: cycle, all")
        if self.realism_variants < 1:
            raise ValueError("realism_variants must be at least 1")
        if not self.realism_sample_rates:
            raise ValueError("realism_sample_rates must not be empty")
        if any(sample_rate < 8000 for sample_rate in self.realism_sample_rates):
            raise ValueError("realism_sample_rates must be at least 8000")
        if not self.transpose_semitones:
            raise ValueError("transpose_semitones must not be empty")
        if any(semitone == 0 for semitone in self.transpose_semitones):
            raise ValueError("transpose_semitones must not include 0")
        if any(ratio < 0 for ratio in (
            self.manifest_train_ratio,
            self.manifest_val_ratio,
            self.manifest_test_ratio,
        )):
            raise ValueError("manifest split ratios must be non-negative")
        if abs(
            self.manifest_train_ratio + self.manifest_val_ratio + self.manifest_test_ratio - 1.0
        ) > 1e-6:
            raise ValueError("manifest split ratios must sum to 1.0")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be at least 1")
        if self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta must be non-negative")
        if self.reduce_lr_factor <= 0 or self.reduce_lr_factor >= 1:
            raise ValueError("reduce_lr_factor must be between 0 and 1")
        if self.reduce_lr_patience < 1:
            raise ValueError("reduce_lr_patience must be at least 1")
        if self.min_lr <= 0:
            raise ValueError("min_lr must be positive")
        if self.tensorboard_histogram_freq < 0:
            raise ValueError("tensorboard_histogram_freq must be non-negative")

    def resolve_path(self, project_root: Path, path: Path) -> Path:
        return path if path.is_absolute() else project_root / path


def flatten_config_data(data: dict[str, Any], known_fields: set[str]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    unknown: list[str] = []

    for key, value in data.items():
        if key in known_fields:
            if key in flattened:
                raise ValueError(f"Duplicate config key: {key}")
            flattened[key] = value
            continue

        if isinstance(value, dict) and key in CONFIG_SECTIONS:
            for nested_key, nested_value in value.items():
                if nested_key not in CONFIG_SECTIONS[key] or nested_key not in known_fields:
                    unknown.append(f"{key}.{nested_key}")
                    continue
                if nested_key in flattened:
                    raise ValueError(f"Duplicate config key: {nested_key}")
                flattened[nested_key] = nested_value
            continue

        unknown.append(key)

    if unknown:
        raise ValueError(f"Unknown config keys: {', '.join(sorted(unknown))}")

    return flattened


def coerce_config_data(data: dict[str, Any]) -> dict[str, Any]:
    coerced: dict[str, Any] = {}
    for key, value in data.items():
        if key in PATH_FIELDS:
            coerced[key] = Path(value)
        elif key in FLOAT_TUPLE_FIELDS:
            coerced[key] = tuple(float(item) for item in value)
        elif key in INT_TUPLE_FIELDS:
            coerced[key] = tuple(int(item) for item in value)
        elif key in STR_TUPLE_FIELDS:
            coerced[key] = tuple(str(item) for item in value)
        elif key == "instrument_programs":
            coerced[key] = {str(name): int(program) for name, program in value.items()}
        else:
            coerced[key] = value
    return coerced
