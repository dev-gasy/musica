"""Configuration loading and validation for model training."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from musica.logging import logger

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

AtLeastOneInt = Annotated[int, Field(ge=1)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveFloat = Annotated[float, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0)]
SampleRate = Annotated[int, Field(ge=8000)]
MidiProgram = Annotated[int, Field(ge=0, le=127)]
MidiVelocity = Annotated[int, Field(ge=1, le=127)]
LearningRateFactor = Annotated[float, Field(gt=0, lt=1)]


class MusicaConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int = 0
    epochs: AtLeastOneInt = 60
    batch_size: AtLeastOneInt = 32
    learning_rate: PositiveFloat = 0.001
    force_retrain: bool = False
    val_ratio: PositiveFloat = 0.15
    test_ratio: PositiveFloat = 0.15
    sample_rate: SampleRate = 22050
    target_duration: PositiveFloat = 1.5
    hop_length: AtLeastOneInt = 512
    bins_per_octave: AtLeastOneInt = 12
    n_chroma: Literal[12] = 12
    top_k: AtLeastOneInt = 3
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
    chord_duration: PositiveFloat = 1.5
    chord_sample_rate: SampleRate = 44100
    renderer: Literal["auto", "pretty-midi", "fluidsynth"] = "auto"
    soundfont_path: Path = Path("assets/soundfonts/FluidR3_GM.sf2")
    instrument_programs: Annotated[dict[str, MidiProgram], Field(min_length=1)] = Field(
        default_factory=lambda: {
            "piano": 0,
            "guitar": 24,
            "synth_pad": 88,
        }
    )
    repetitions: AtLeastOneInt = 1
    octave_offsets: Annotated[tuple[int, ...], Field(min_length=1)] = (-1, 0, 1)
    velocities: Annotated[tuple[MidiVelocity, ...], Field(min_length=1)] = (80, 100, 120)
    humanize: bool = True
    humanize_onset_ms: NonNegativeFloat = 35.0
    velocity_jitter: NonNegativeInt = 8
    noise_snrs_db: Annotated[tuple[float, ...], Field(min_length=1)] = (15.0,)
    noise_mode: Literal["cycle", "all"] = "cycle"
    realism_variants: AtLeastOneInt = 2
    realism_sample_rates: Annotated[tuple[SampleRate, ...], Field(min_length=1)] = (
        22050,
        44100,
        48000,
    )
    realism_keep_duration: bool = False
    transpose_semitones: Annotated[tuple[int, ...], Field(min_length=1)] = (-5, 7)
    transpose_roots: tuple[str, ...] = ()
    transpose_qualities: tuple[str, ...] = ()
    manifest_train_ratio: NonNegativeFloat = 0.80
    manifest_val_ratio: NonNegativeFloat = 0.10
    manifest_test_ratio: NonNegativeFloat = 0.10
    include_missing_audio: bool = False
    early_stopping_patience: AtLeastOneInt = 8
    early_stopping_min_delta: NonNegativeFloat = 1e-4
    reduce_lr_factor: LearningRateFactor = 0.3
    reduce_lr_patience: AtLeastOneInt = 4
    min_lr: PositiveFloat = 1e-6
    tensorboard_histogram_freq: NonNegativeInt = 0

    @classmethod
    def load(cls, path: Path = Path("musica.toml")) -> "MusicaConfig":
        logger.info("Chargement de la configuration: {}", path)
        data = tomllib.loads(path.read_text()) if path.exists() else {}
        known_fields = set(cls.model_fields)
        data = flatten_config_data(data, known_fields)
        unknown = sorted(set(data) - known_fields)
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {', '.join(unknown)}")
        config = cls(**data)
        logger.info(
            "Configuration chargee: epochs={}, batch_size={}, lr={}, force_retrain={}",
            config.epochs,
            config.batch_size,
            config.learning_rate,
            config.force_retrain,
        )
        return config

    @field_validator("transpose_semitones")
    @classmethod
    def validate_transpose_semitones(cls, semitones: tuple[int, ...]) -> tuple[int, ...]:
        if any(semitone == 0 for semitone in semitones):
            raise ValueError("transpose_semitones must not include 0")
        return semitones

    @model_validator(mode="after")
    def validate_ratios(self) -> Self:
        if self.val_ratio + self.test_ratio >= 1:
            raise ValueError("val_ratio + test_ratio must be below 1")

        manifest_ratios = (
            self.manifest_train_ratio,
            self.manifest_val_ratio,
            self.manifest_test_ratio,
        )
        if abs(sum(manifest_ratios) - 1.0) > 1e-6:
            raise ValueError("manifest split ratios must sum to 1.0")
        return self

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
