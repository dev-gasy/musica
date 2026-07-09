"""Training-data augmentation for chord transposition."""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np

from musica.modeling.constants import QUALITIES, ROOTS

LOGGER = logging.getLogger(__name__)


class TranspositionAugmenter:
    def __init__(self, n_notes: int = len(ROOTS), n_qualities: int = len(QUALITIES)) -> None:
        self.n_notes = n_notes
        self.n_qualities = n_qualities

    def augment(
        self,
        x: np.ndarray,
        y: np.ndarray,
        shifts: Iterable[int] = range(12),
    ) -> tuple[np.ndarray, np.ndarray]:
        x_aug = []
        y_aug = []
        shifts = tuple(shifts)
        LOGGER.info("Augmentation par transposition: shifts=%s", shifts)

        for shift in shifts:
            x_shifted = np.roll(x, shift=shift, axis=2)
            root = y // self.n_qualities
            quality = y % self.n_qualities
            shifted_root = (root + shift) % self.n_notes
            y_shifted = shifted_root * self.n_qualities + quality
            x_aug.append(x_shifted)
            y_aug.append(y_shifted)

        augmented_x = np.concatenate(x_aug, axis=0)
        augmented_y = np.concatenate(y_aug, axis=0)
        LOGGER.info(
            "Augmentation terminee: %s -> %s exemples",
            x.shape[0],
            augmented_x.shape[0],
        )
        return augmented_x, augmented_y
