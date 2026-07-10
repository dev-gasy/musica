"""Public API for the chord recognition pipeline."""

from __future__ import annotations

from musica.config import (
    AudioGenerationConfig,
    CallbackConfig,
    DatasetConfig,
    ExampleAudioConfig,
    FeatureConfig,
    GeneralConfig,
    HumanizationConfig,
    ManifestConfig,
    ModelConfig,
    MusicaConfig,
    NoiseConfig,
    PredictionConfig,
    RealismConfig,
    TrainingConfig,
    TranspositionConfig,
)
from musica.modeling.augmentation import TranspositionAugmenter
from musica.modeling.constants import NOTE_ALIASES, QUALITIES, ROOTS
from musica.modeling.dataset import ChordDataset, DatasetSplit
from musica.modeling.evaluation import ChordEvaluator, EvaluationResult
from musica.modeling.features import FeatureExtractor
from musica.modeling.prediction import ChordPredictor
from musica.modeling.preparation import prepare_data
from musica.modeling.training import ChordTrainer, PreparedData, TrainingResult
from musica.modeling.utils import label_sort_key, stable_digest

__all__ = [
    "NOTE_ALIASES",
    "QUALITIES",
    "ROOTS",
    "AudioGenerationConfig",
    "CallbackConfig",
    "ChordDataset",
    "ChordEvaluator",
    "ChordPredictor",
    "ChordTrainer",
    "DatasetConfig",
    "DatasetSplit",
    "EvaluationResult",
    "ExampleAudioConfig",
    "FeatureConfig",
    "FeatureExtractor",
    "GeneralConfig",
    "HumanizationConfig",
    "ManifestConfig",
    "ModelConfig",
    "MusicaConfig",
    "NoiseConfig",
    "PreparedData",
    "PredictionConfig",
    "RealismConfig",
    "TrainingConfig",
    "TrainingResult",
    "TranspositionConfig",
    "TranspositionAugmenter",
    "label_sort_key",
    "prepare_data",
    "stable_digest",
]
