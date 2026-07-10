"""Configuration loading and validation for model training."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from musica.logging import logger

AtLeastOneInt = Annotated[int, Field(ge=1)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveFloat = Annotated[float, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0)]
SampleRate = Annotated[int, Field(ge=8000)]
MidiProgram = Annotated[int, Field(ge=0, le=127)]
MidiVelocity = Annotated[int, Field(ge=1, le=127)]
LearningRateFactor = Annotated[float, Field(gt=0, lt=1)]


class ConfigSection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class GeneralConfig(ConfigSection):
    seed: int = 0


class TrainingConfig(ConfigSection):
    epochs: AtLeastOneInt = 60
    batch_size: AtLeastOneInt = 32
    learning_rate: PositiveFloat = 0.001
    force_retrain: bool = False
    val_ratio: PositiveFloat = 0.15
    test_ratio: PositiveFloat = 0.15
    logs_dir: Path = Path("logs")

    @model_validator(mode="after")
    def validate_ratios(self) -> Self:
        if self.val_ratio + self.test_ratio >= 1:
            raise ValueError("val_ratio + test_ratio must be below 1")
        return self


class FeatureConfig(ConfigSection):
    sample_rate: SampleRate = 22050
    target_duration: PositiveFloat = 1.5
    hop_length: AtLeastOneInt = 512
    bins_per_octave: AtLeastOneInt = 12
    n_chroma: Literal[12] = 12
    cache_features: bool = True
    feature_cache_dir: Path = Path("logs/features")


class PredictionConfig(ConfigSection):
    top_k: AtLeastOneInt = 3


class DatasetConfig(ConfigSection):
    dataset_dir: Path = Path("audio/chords")
    recorded_audio_dir: Path = Path("audio/chords/recorded")


class ExampleAudioConfig(ConfigSection):
    directory: Path = Path("examples")
    extensions: Annotated[tuple[str, ...], Field(min_length=1)] = (
        ".wav",
        ".mp3",
        ".flac",
        ".ogg",
        ".m4a",
        ".aiff",
        ".aif",
    )

    def audio_paths(self, project_root: Path) -> list[Path]:
        directory = self.resolve_directory(project_root)
        if not directory.exists() or not directory.is_dir():
            raise FileNotFoundError(f"No examples directory found: {directory}")
        extensions = {extension.lower() for extension in self.extensions}
        return sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in extensions
        )

    def resolve_directory(self, project_root: Path) -> Path:
        return self.directory if self.directory.is_absolute() else project_root / self.directory


class ModelConfig(ConfigSection):
    architecture: str = "cnn_chords_chroma_cqt_v1"


class AudioGenerationConfig(ConfigSection):
    chord_duration: PositiveFloat = 1.5
    chord_sample_rate: SampleRate = 44100
    renderer: Literal["auto", "pretty-midi", "fluidsynth"] = "auto"
    midi_output_dir: Path = Path("audio/chords/midi")
    clean_output_dir: Path = Path("audio/chords/clean")
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


class HumanizationConfig(ConfigSection):
    humanize: bool = True
    humanize_onset_ms: NonNegativeFloat = 35.0
    velocity_jitter: NonNegativeInt = 8


class NoiseConfig(ConfigSection):
    noise_snrs_db: Annotated[tuple[float, ...], Field(min_length=1)] = (15.0,)
    noise_mode: Literal["cycle", "all"] = "cycle"
    noise_download_dir: Path = Path("assets/noises/internet")
    noisy_output_dir: Path = Path("audio/chords/noisy")


class RealismConfig(ConfigSection):
    realism_variants: AtLeastOneInt = 2
    realism_sample_rates: Annotated[tuple[SampleRate, ...], Field(min_length=1)] = (
        22050,
        44100,
        48000,
    )
    realism_keep_duration: bool = False
    realistic_output_dir: Path = Path("audio/chords/realistic")


class TranspositionConfig(ConfigSection):
    transpose_semitones: Annotated[tuple[int, ...], Field(min_length=1)] = (-5, 7)
    transpose_roots: tuple[str, ...] = ()
    transpose_qualities: tuple[str, ...] = ()
    transposed_output_dir: Path = Path("audio/chords/transposed")

    @field_validator("transpose_semitones")
    @classmethod
    def validate_transpose_semitones(cls, semitones: tuple[int, ...]) -> tuple[int, ...]:
        if any(semitone == 0 for semitone in semitones):
            raise ValueError("transpose_semitones must not include 0")
        return semitones


class ManifestConfig(ConfigSection):
    manifest_train_ratio: NonNegativeFloat = 0.80
    manifest_val_ratio: NonNegativeFloat = 0.10
    manifest_test_ratio: NonNegativeFloat = 0.10
    include_missing_audio: bool = False
    audio_manifest_path: Path = Path("audio/manifest.csv")

    @model_validator(mode="after")
    def validate_ratios(self) -> Self:
        ratios = (
            self.manifest_train_ratio,
            self.manifest_val_ratio,
            self.manifest_test_ratio,
        )
        if abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError("manifest split ratios must sum to 1.0")
        return self


class CallbackConfig(ConfigSection):
    early_stopping_patience: AtLeastOneInt = 8
    early_stopping_min_delta: NonNegativeFloat = 1e-4
    reduce_lr_factor: LearningRateFactor = 0.3
    reduce_lr_patience: AtLeastOneInt = 4
    min_lr: PositiveFloat = 1e-6
    tensorboard_histogram_freq: NonNegativeInt = 0


CONFIG_SECTIONS: dict[str, set[str]] = {
    "general": set(GeneralConfig.model_fields),
    "training": set(TrainingConfig.model_fields),
    "features": set(FeatureConfig.model_fields),
    "prediction": set(PredictionConfig.model_fields),
    "dataset": set(DatasetConfig.model_fields),
    "examples": set(ExampleAudioConfig.model_fields),
    "model": {"architecture", "model_architecture"},
    "audio": set(AudioGenerationConfig.model_fields),
    "humanization": set(HumanizationConfig.model_fields),
    "noise": set(NoiseConfig.model_fields),
    "realism": set(RealismConfig.model_fields),
    "transpose": set(TranspositionConfig.model_fields),
    "manifest": set(ManifestConfig.model_fields),
    "callbacks": set(CallbackConfig.model_fields),
}
FIELD_TO_SECTION = {
    field_name: section
    for section, field_names in CONFIG_SECTIONS.items()
    for field_name in field_names
}


class MusicaConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    general: GeneralConfig = Field(default_factory=GeneralConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    prediction: PredictionConfig = Field(default_factory=PredictionConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    examples: ExampleAudioConfig = Field(default_factory=ExampleAudioConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    audio: AudioGenerationConfig = Field(default_factory=AudioGenerationConfig)
    humanization: HumanizationConfig = Field(default_factory=HumanizationConfig)
    noise: NoiseConfig = Field(default_factory=NoiseConfig)
    realism: RealismConfig = Field(default_factory=RealismConfig)
    transpose: TranspositionConfig = Field(default_factory=TranspositionConfig)
    manifest: ManifestConfig = Field(default_factory=ManifestConfig)
    callbacks: CallbackConfig = Field(default_factory=CallbackConfig)

    @model_validator(mode="before")
    @classmethod
    def normalize_sections(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return normalize_config_data(data)
        return data

    @classmethod
    def load(cls, path: Path = Path("musica.toml")) -> "MusicaConfig":
        logger.info("Chargement de la configuration: {}", path)
        data = load_toml_file(path)
        config = cls(**data)
        logger.info(
            "Configuration chargee: epochs={}, batch_size={}, lr={}, force_retrain={}",
            config.epochs,
            config.batch_size,
            config.learning_rate,
            config.force_retrain,
        )
        return config

    def __getattr__(self, name: str) -> Any:
        section = FIELD_TO_SECTION.get(name)
        if section is not None:
            return getattr(getattr(self, section), normalized_field_name(name))
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def resolve_path(self, project_root: Path, path: Path) -> Path:
        return path if path.is_absolute() else project_root / path


def load_toml_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    with path.open("rb") as toml_file:
        return tomllib.load(toml_file)


def normalize_config_data(data: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, dict[str, Any]] = {}
    unknown: list[str] = []

    for key, value in data.items():
        if key in CONFIG_SECTIONS:
            if not isinstance(value, dict):
                unknown.append(key)
                continue
            merge_section(normalized, unknown, key, value)
            continue

        section = FIELD_TO_SECTION.get(key)
        if section is None:
            unknown.append(key)
            continue
        merge_section(normalized, unknown, section, {key: value})

    if unknown:
        raise ValueError(f"Unknown config keys: {', '.join(sorted(unknown))}")
    return normalized


def merge_section(
        normalized: dict[str, dict[str, Any]],
        unknown: list[str],
        section: str,
        values: dict[str, Any],
) -> None:
    section_data = normalized.setdefault(section, {})
    section_fields = CONFIG_SECTIONS[section]
    for key, value in values.items():
        if key not in section_fields:
            unknown.append(f"{section}.{key}")
            continue
        target_key = normalized_field_name(key)
        if target_key in section_data:
            raise ValueError(f"Duplicate config key: {key}")
        section_data[target_key] = value


def normalized_field_name(field_name: str) -> str:
    if field_name == "model_architecture":
        return "architecture"
    return field_name
