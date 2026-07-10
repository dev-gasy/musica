"""Single-file chord prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from musica.config import MusicaConfig
from musica.logging import logger
from musica.modeling.features import FeatureExtractor


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
        requested_path = audio_path or self.default_audio_path()
        path = self.config.resolve_path(self.extractor.dataset.project_root, requested_path)
        logger.info("Prediction de l'audio: {}", path)
        features = self.extractor.audio_features(path)
        probabilities = model.predict(features[np.newaxis, ...], verbose=0)[0]
        top_indices = np.argsort(probabilities)[-self.config.top_k:][::-1]
        return [
            (self.labels[int(index)], float(probabilities[int(index)]))
            for index in top_indices
        ]

    def default_audio_path(self) -> Path:
        audio_paths = self.config.examples.audio_paths(self.extractor.dataset.project_root)
        if not audio_paths:
            raise FileNotFoundError(
                f"No example audio files found in {self.config.examples.directory}"
            )
        return audio_paths[0]
