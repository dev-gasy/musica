"""Audio feature extraction for chord recognition."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from musica.logging import logger
from musica.modeling.config import MusicaConfig
from musica.modeling.dataset import ChordDataset


class FeatureExtractor:
    def __init__(self, config: MusicaConfig, dataset: ChordDataset) -> None:
        self.config = config
        self.dataset = dataset

    def audio_features(self, path: Path) -> np.ndarray:
        import librosa

        audio, _ = librosa.load(path, sr=self.config.sample_rate, mono=True)
        target_samples = int(self.config.target_duration * self.config.sample_rate)
        audio = audio[:target_samples]
        audio = np.pad(audio, (0, max(0, target_samples - len(audio))))

        chroma = librosa.feature.chroma_cqt(
            y=audio,
            sr=self.config.sample_rate,
            hop_length=self.config.hop_length,
            bins_per_octave=self.config.bins_per_octave,
            n_chroma=self.config.n_chroma,
        )
        return chroma.T[..., np.newaxis].astype(np.float32)

    def load_features(self, paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
        if not paths:
            raise ValueError("Cannot load features for an empty path list")

        logger.info("Extraction des features: {} fichiers", len(paths))
        first = self.audio_features(paths[0])
        x = np.empty((len(paths), *first.shape), dtype=np.float32)
        y = np.empty(len(paths), dtype=np.int32)
        x[0] = first
        y[0] = self.dataset.label_to_index[self.dataset.label_from_path(paths[0])]

        for index, path in enumerate(paths[1:], start=1):
            x[index] = self.audio_features(path)
            y[index] = self.dataset.label_to_index[self.dataset.label_from_path(path)]
            if (index + 1) % 250 == 0 or index + 1 == len(paths):
                logger.info("Features extraites: {}/{}", index + 1, len(paths))

        logger.info("Features pretes: shape={}", x.shape)
        return x, y
