"""Model building, cache handling, and training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from musica.logging import logger
from musica.modeling.augmentation import TranspositionAugmenter
from musica.modeling.config import MusicaConfig
from musica.modeling.constants import QUALITIES, ROOTS
from musica.modeling.dataset import ChordDataset, DatasetSplit
from musica.modeling.utils import stable_digest


@dataclass(frozen=True)
class PreparedData:
    dataset: ChordDataset
    split: DatasetSplit
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


@dataclass(frozen=True)
class TrainingResult:
    model: Any
    signature: str
    model_path: Path
    history_log_path: Path
    cache_hit: bool


@dataclass(frozen=True)
class RunArtifacts:
    run_dir: Path
    model_path: Path
    params_path: Path
    history_log_path: Path


class ChordTrainer:
    def __init__(
        self,
        config: MusicaConfig,
        dataset: ChordDataset,
        project_root: Path | None = None,
    ) -> None:
        self.config = config
        self.dataset = dataset
        self.project_root = Path.cwd() if project_root is None else project_root
        self.logs_dir = config.resolve_path(self.project_root, config.logs_dir)

    def build_model(
        self,
        input_shape: tuple[int, ...],
        label_count: int,
        normalizer: Any,
    ) -> Any:
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import (
            Activation,
            BatchNormalization,
            Conv2D,
            Dense,
            Dropout,
            GlobalAveragePooling2D,
            MaxPool2D,
        )
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.regularizers import l2

        model = Sequential(name="cnn_chords")
        model.add(tf.keras.Input(shape=input_shape))
        model.add(normalizer)

        model.add(Conv2D(32, (3, 12), padding="same", kernel_initializer="he_uniform"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPool2D((2, 1)))
        model.add(Dropout(0.10))

        model.add(Conv2D(64, (3, 3), padding="same", kernel_initializer="he_uniform"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPool2D((2, 1)))
        model.add(Dropout(0.10))

        model.add(Conv2D(64, (3, 3), padding="same", kernel_initializer="he_uniform"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPool2D((2, 1)))
        model.add(Dropout(0.15))

        model.add(GlobalAveragePooling2D())
        model.add(
            Dense(
                128,
                activation="relu",
                kernel_initializer="he_uniform",
                kernel_regularizer=l2(1e-4),
            )
        )
        model.add(Dropout(0.25))
        model.add(Dense(label_count, activation="softmax"))

        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        logger.info("Modele CNN construit: input_shape={}, classes={}", input_shape, label_count)
        return model

    def training_params(self, split: DatasetSplit) -> dict[str, Any]:
        return {
            "seed": self.config.seed,
            "epochs": self.config.epochs,
            "batch_size": self.config.batch_size,
            "learning_rate": self.config.learning_rate,
            "val_ratio": self.config.val_ratio,
            "test_ratio": self.config.test_ratio,
            "sample_rate": self.config.sample_rate,
            "target_duration": self.config.target_duration,
            "hop_length": self.config.hop_length,
            "bins_per_octave": self.config.bins_per_octave,
            "n_chroma": self.config.n_chroma,
            "n_notes": len(ROOTS),
            "n_qualites": len(QUALITIES),
            "augmentation": "all_transpositions_0_to_11",
            "model_architecture": self.config.model_architecture,
            "labels": self.dataset.labels,
            "dataset_digest": self.dataset.digest_paths(self.dataset.audio_paths),
            "train_digest": self.dataset.digest_paths(split.train_paths),
            "validation_digest": self.dataset.digest_paths(split.val_paths),
            "test_digest": self.dataset.digest_paths(split.test_paths),
            "audio_file_count": len(self.dataset.audio_paths),
            "train_file_count": len(split.train_paths),
            "validation_file_count": len(split.val_paths),
            "test_file_count": len(split.test_paths),
            "early_stopping_patience": self.config.early_stopping_patience,
            "early_stopping_min_delta": self.config.early_stopping_min_delta,
            "reduce_lr_factor": self.config.reduce_lr_factor,
            "reduce_lr_patience": self.config.reduce_lr_patience,
            "min_lr": self.config.min_lr,
            "tensorboard_histogram_freq": self.config.tensorboard_histogram_freq,
        }

    def signature(self, split: DatasetSplit) -> str:
        signature = stable_digest(self.training_params(split))[:12]
        logger.info("Signature du run: {}", signature)
        return signature

    def load_cached_model(self, signature: str) -> tuple[Any, Path, Path] | None:
        import tensorflow as tf

        model_path, history_log_path = self.cached_model_paths(signature)
        logger.info("Verification du cache modele: {}", model_path)
        if not self.config.force_retrain and model_path.exists():
            logger.info("Cache hit: chargement de {}", model_path)
            return tf.keras.models.load_model(model_path), model_path, history_log_path

        legacy_model_path = self.config.resolve_path(
            self.project_root,
            self.config.legacy_model_path,
        )
        legacy_params_path = self.config.resolve_path(
            self.project_root,
            self.config.legacy_params_path,
        )
        if self.can_use_legacy_cache(legacy_model_path, legacy_params_path):
            legacy_signature = json.loads(legacy_params_path.read_text()).get("signature")
            if legacy_signature == signature:
                logger.info("Cache legacy hit: chargement de {}", legacy_model_path)
                return (
                    tf.keras.models.load_model(legacy_model_path),
                    legacy_model_path,
                    self.logs_dir / "training_log.csv",
                )

        if self.config.force_retrain:
            logger.info("Cache ignore car force_retrain=true")
        else:
            logger.info("Aucun modele en cache pour la signature {}", signature)
        return None

    def cached_model_paths(self, signature: str) -> tuple[Path, Path]:
        run_dir = self.logs_dir / "models" / signature
        return run_dir / "best_model.keras", run_dir / "training_log.csv"

    def can_use_legacy_cache(self, model_path: Path, params_path: Path) -> bool:
        return (
            not self.config.force_retrain
            and model_path.exists()
            and params_path.exists()
        )

    def build_augmented_model(
        self,
        prepared: PreparedData,
    ) -> tuple[Any, np.ndarray, np.ndarray]:
        import tensorflow as tf
        from tensorflow.keras.layers import Normalization

        tf.keras.utils.set_random_seed(self.config.seed)

        augmenter = TranspositionAugmenter()
        x_train_aug, y_train_aug = augmenter.augment(prepared.x_train, prepared.y_train)

        logger.info("Adaptation de la normalisation sur le train augmente")
        normalizer = Normalization(axis=-1)
        normalizer.adapt(x_train_aug)
        model = self.build_model(
            prepared.x_train.shape[1:],
            len(prepared.dataset.labels),
            normalizer,
        )
        return model, x_train_aug, y_train_aug

    def run_artifacts(self, signature: str) -> RunArtifacts:
        run_dir = self.logs_dir / "models" / signature
        return RunArtifacts(
            run_dir=run_dir,
            model_path=run_dir / "best_model.keras",
            params_path=run_dir / "params.json",
            history_log_path=run_dir / "training_log.csv",
        )

    def write_training_metadata(
        self,
        signature: str,
        split: DatasetSplit,
        params_path: Path,
    ) -> None:
        logger.info("Ecriture des metadonnees du run: {}", params_path)
        params_path.write_text(
            json.dumps(
                {"signature": signature, "params": self.training_params(split)},
                indent=2,
                sort_keys=True,
            )
        )

    def training_callbacks(self, artifacts: RunArtifacts) -> list[Any]:
        from tensorflow.keras.callbacks import (
            CSVLogger,
            EarlyStopping,
            ModelCheckpoint,
            ReduceLROnPlateau,
            TensorBoard,
        )

        return [
            EarlyStopping(
                monitor="val_loss",
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                min_delta=self.config.early_stopping_min_delta,
                mode="min",
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=self.config.reduce_lr_factor,
                patience=self.config.reduce_lr_patience,
                min_lr=self.config.min_lr,
                mode="min",
                verbose=1,
            ),
            ModelCheckpoint(
                filepath=str(artifacts.model_path),
                monitor="val_loss",
                save_best_only=True,
                mode="min",
                verbose=1,
            ),
            CSVLogger(str(artifacts.history_log_path), append=False),
            TensorBoard(
                log_dir=str(artifacts.run_dir / "tensorboard"),
                histogram_freq=self.config.tensorboard_histogram_freq,
            ),
        ]

    def train_or_load(
        self,
        prepared: PreparedData,
        fit_kwargs: dict[str, Any] | None = None,
    ) -> TrainingResult:
        signature = self.signature(prepared.split)
        cached = self.load_cached_model(signature)
        if cached is not None:
            model, model_path, history_log_path = cached
            return TrainingResult(
                model=model,
                signature=signature,
                model_path=model_path,
                history_log_path=history_log_path,
                cache_hit=True,
            )

        model, x_train_aug, y_train_aug = self.build_augmented_model(prepared)
        artifacts = self.run_artifacts(signature)
        artifacts.run_dir.mkdir(parents=True, exist_ok=True)
        self.write_training_metadata(signature, prepared.split, artifacts.params_path)

        kwargs = fit_kwargs or {}
        logger.info(
            "Debut entrainement: epochs={}, batch_size={}, run_dir={}",
            self.config.epochs,
            self.config.batch_size,
            artifacts.run_dir,
        )
        model.fit(
            x_train_aug,
            y_train_aug,
            validation_data=(prepared.x_val, prepared.y_val),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=self.training_callbacks(artifacts),
            **kwargs,
        )
        logger.info("Entrainement termine: meilleur modele={}", artifacts.model_path)
        return TrainingResult(
            model=model,
            signature=signature,
            model_path=artifacts.model_path,
            history_log_path=artifacts.history_log_path,
            cache_hit=False,
        )
