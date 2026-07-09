"""Dataset discovery, labels, and deterministic splits."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from musica.modeling.config import MusicaConfig
from musica.modeling.constants import NOTE_ALIASES
from musica.modeling.utils import label_sort_key, stable_digest

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetSplit:
    train_paths: list[Path]
    val_paths: list[Path]
    test_paths: list[Path]


class ChordDataset:
    def __init__(self, config: MusicaConfig, project_root: Path | None = None) -> None:
        self.config = config
        self.project_root = Path.cwd() if project_root is None else project_root
        self.dataset_dir = config.resolve_path(self.project_root, config.dataset_dir)
        self.audio_paths: list[Path] = []
        self.labels: list[str] = []
        self.label_to_index: dict[str, int] = {}

    def discover(self) -> "ChordDataset":
        LOGGER.info("Recherche des fichiers WAV dans %s", self.dataset_dir)
        self.audio_paths = sorted(self.dataset_dir.glob("**/*.wav"))
        if not self.audio_paths:
            raise FileNotFoundError(f"No WAV files found under {self.dataset_dir}")
        self.labels = sorted(
            {self.label_from_path(path) for path in self.audio_paths},
            key=label_sort_key,
        )
        self.label_to_index = {
            label: index for index, label in enumerate(self.labels)
        }
        LOGGER.info(
            "Dataset decouvert: %s fichiers audio, %s classes",
            len(self.audio_paths),
            len(self.labels),
        )
        return self

    def split(self) -> DatasetSplit:
        if not self.audio_paths:
            self.discover()

        LOGGER.info(
            "Creation du split stratifie: val_ratio=%s, test_ratio=%s, seed=%s",
            self.config.val_ratio,
            self.config.test_ratio,
            self.config.seed,
        )
        rng = np.random.default_rng(self.config.seed)
        train_paths: list[Path] = []
        val_paths: list[Path] = []
        test_paths: list[Path] = []

        for label in self.labels:
            class_paths = [
                path for path in self.audio_paths if self.label_from_path(path) == label
            ]
            indices = rng.permutation(len(class_paths))
            n_test = max(1, int(round(len(class_paths) * self.config.test_ratio)))
            n_val = max(1, int(round(len(class_paths) * self.config.val_ratio)))

            test_paths.extend(class_paths[int(index)] for index in indices[:n_test])
            val_paths.extend(
                class_paths[int(index)] for index in indices[n_test:n_test + n_val]
            )
            train_paths.extend(
                class_paths[int(index)] for index in indices[n_test + n_val:]
            )

        rng.shuffle(train_paths)
        rng.shuffle(val_paths)
        rng.shuffle(test_paths)
        LOGGER.info(
            "Split pret: train=%s, validation=%s, test=%s",
            len(train_paths),
            len(val_paths),
            len(test_paths),
        )
        return DatasetSplit(train_paths, val_paths, test_paths)

    @staticmethod
    def label_from_path(path: Path) -> str:
        note, quality = path.stem.split("_")[:2]
        note = NOTE_ALIASES.get(note, note)
        return f"{note}_{quality}"

    def digest_paths(self, paths: Iterable[Path]) -> str:
        payload = []
        for path in sorted(paths):
            stat = path.stat()
            payload.append({
                "path": str(path.relative_to(self.project_root)),
                "size": stat.st_size,
            })
        return stable_digest(payload)
