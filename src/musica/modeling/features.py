"""Audio feature extraction for chord recognition."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from musica.config import MusicaConfig
from musica.logging import logger
from musica.modeling.dataset import ChordDataset
from musica.modeling.utils import stable_digest


FEATURE_CACHE_VERSION = 1


class FeatureExtractor:
    def __init__(
        self,
        config: MusicaConfig,
        dataset: ChordDataset,
        project_root: Path | None = None,
    ) -> None:
        self.config = config
        self.dataset = dataset
        self.project_root = Path.cwd() if project_root is None else project_root
        self.cache_dir = config.resolve_path(self.project_root, config.feature_cache_dir)

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

    def load_features(
        self,
        paths: list[Path],
        split_name: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not paths:
            raise ValueError("Cannot load features for an empty path list")

        cache_path = self.feature_cache_path(paths, split_name)
        if self.config.cache_features:
            cached = self.load_cached_features(cache_path)
            if cached is not None:
                return cached

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
        if self.config.cache_features:
            self.write_cached_features(cache_path, x, y)
        return x, y

    def feature_cache_path(self, paths: list[Path], split_name: str | None = None) -> Path:
        signature = stable_digest(self.feature_cache_params(paths, split_name))[:12]
        prefix = f"{split_name}-" if split_name else ""
        return self.cache_dir / f"{prefix}{signature}.npz"

    def feature_cache_params(
        self,
        paths: list[Path],
        split_name: str | None = None,
    ) -> dict[str, Any]:
        return {
            "version": FEATURE_CACHE_VERSION,
            "split": split_name,
            "sample_rate": self.config.sample_rate,
            "target_duration": self.config.target_duration,
            "hop_length": self.config.hop_length,
            "bins_per_octave": self.config.bins_per_octave,
            "n_chroma": self.config.n_chroma,
            "labels": self.dataset.labels,
            "files": [self.file_cache_params(path) for path in paths],
        }

    def file_cache_params(self, path: Path) -> dict[str, Any]:
        stat = path.stat()
        try:
            cache_path = path.relative_to(self.project_root)
        except ValueError:
            cache_path = path
        return {
            "path": str(cache_path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "label": self.dataset.label_from_path(path),
        }

    def load_cached_features(self, cache_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
        if not cache_path.exists():
            logger.info("Aucun cache de features: {}", cache_path)
            return None

        try:
            with np.load(cache_path) as cached:
                x = cached["x"].astype(np.float32, copy=False)
                y = cached["y"].astype(np.int32, copy=False)
        except (OSError, ValueError, KeyError) as exc:
            logger.warning("Cache de features ignore ({}): {}", cache_path, exc)
            return None

        logger.info("Cache hit features: {} shape={}", cache_path, x.shape)
        return x, y

    def write_cached_features(
        self,
        cache_path: Path,
        x: np.ndarray,
        y: np.ndarray,
    ) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, x=x, y=y)
        logger.info("Cache features ecrit: {}", cache_path)
