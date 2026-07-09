"""Single-file chord prediction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from musica.modeling.config import MusicaConfig
from musica.modeling.features import FeatureExtractor

LOGGER = logging.getLogger(__name__)


class ChordPredictor:
    def __init__(
        self,
        config: MusicaConfig,
        extractor: FeatureExtractor,
        labels: list[str],
    ) -> None:
        self.config = config
        self.extractor = extractor
        self.labels = labels

    def predict(self, model: Any, audio_path: Path | None = None) -> list[tuple[str, float]]:
        requested_path = audio_path or self.config.example_audio_path
        path = self.config.resolve_path(self.extractor.dataset.project_root, requested_path)
        LOGGER.info("Prediction de l'audio: %s", path)
        features = self.extractor.audio_features(path)
        probabilities = model.predict(features[np.newaxis, ...], verbose=0)[0]
        top_indices = np.argsort(probabilities)[-self.config.top_k:][::-1]
        return [
            (self.labels[int(index)], float(probabilities[int(index)]))
            for index in top_indices
        ]
