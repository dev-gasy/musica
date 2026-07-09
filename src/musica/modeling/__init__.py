"""Public API for the chord recognition pipeline."""

from __future__ import annotations

from musica.modeling.augmentation import TranspositionAugmenter
from musica.modeling.config import MusicaConfig
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
    "ChordDataset",
    "ChordEvaluator",
    "ChordPredictor",
    "ChordTrainer",
    "DatasetSplit",
    "EvaluationResult",
    "FeatureExtractor",
    "MusicaConfig",
    "PreparedData",
    "TrainingResult",
    "TranspositionAugmenter",
    "label_sort_key",
    "prepare_data",
    "stable_digest",
]
