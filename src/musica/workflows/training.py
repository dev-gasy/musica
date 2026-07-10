"""Run the Musica chord recognition workflow from musica.toml."""

from __future__ import annotations

from pathlib import Path

from musica.logging import configure_logging as configure_loguru
from musica.logging import logger
from musica.modeling import (
    ChordEvaluator,
    ChordPredictor,
    ChordTrainer,
    FeatureExtractor,
    MusicaConfig,
    prepare_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def run_pipeline(project_root: Path | None = None) -> None:
    configure_logging()
    root = PROJECT_ROOT if project_root is None else project_root
    logger.info("Demarrage du pipeline Musica")
    config = MusicaConfig.load(root / "musica.toml")
    prepared = prepare_data(config, project_root=root)
    trainer = ChordTrainer(config, prepared.dataset, project_root=root)
    training = trainer.train_or_load(prepared)

    print(f"Fichiers audio: {len(prepared.dataset.audio_paths)}")
    print(f"Classes: {len(prepared.dataset.labels)}")
    print(f"Train: {len(prepared.split.train_paths)}")
    print(f"Validation: {len(prepared.split.val_paths)}")
    print(f"Test: {len(prepared.split.test_paths)}")
    print(f"Signature: {training.signature}")
    if training.cache_hit:
        print(f"Modele reutilise: {training.model_path.relative_to(root)}")
    else:
        print(f"Modele entraine: {training.model_path.relative_to(root)}")

    evaluator = ChordEvaluator()
    evaluation = evaluator.evaluate(
        training.model,
        prepared.x_test,
        prepared.y_test,
        prepared.dataset.labels,
    )
    print(f"Test accuracy : {evaluation.test_accuracy:.4f}")
    print(f"Test loss     : {evaluation.test_loss:.4f}")
    print(f"F1 macro      : {evaluation.f1_macro:.4f}")
    print(evaluation.report)

    predictor = ChordPredictor(
        config,
        FeatureExtractor(config, prepared.dataset),
        prepared.dataset.labels,
    )
    predictions = predictor.predict(training.model)
    print(f"Audio exemple : {config.example_audio_path}")
    for label, probability in predictions:
        print(f"{label:8s} {probability:.3f}")
    logger.info("Pipeline Musica termine")


def configure_logging() -> None:
    configure_loguru()
