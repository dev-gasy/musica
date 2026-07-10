"""End-to-end data preparation for training and evaluation."""

from __future__ import annotations

from pathlib import Path

from musica.logging import logger
from musica.modeling.config import MusicaConfig
from musica.modeling.dataset import ChordDataset
from musica.modeling.features import FeatureExtractor
from musica.modeling.training import PreparedData


def prepare_data(config: MusicaConfig, project_root: Path | None = None) -> PreparedData:
    logger.info("Preparation des donnees")
    dataset = ChordDataset(config, project_root=project_root).discover()
    split = dataset.split()
    extractor = FeatureExtractor(config, dataset)
    logger.info("Extraction du train")
    x_train, y_train = extractor.load_features(split.train_paths)
    logger.info("Extraction de la validation")
    x_val, y_val = extractor.load_features(split.val_paths)
    logger.info("Extraction du test")
    x_test, y_test = extractor.load_features(split.test_paths)
    logger.info("Preparation des donnees terminee")
    return PreparedData(
        dataset=dataset,
        split=split,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
    )
