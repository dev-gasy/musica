from pathlib import Path
from types import SimpleNamespace

import numpy as np

from musica.workflows import training


def test_main_pipeline_orchestrates_without_cli_args(monkeypatch, tmp_path: Path, capsys) -> None:
    example_paths = (
        tmp_path / "examples" / "C_maj.wav",
        tmp_path / "examples" / "D_min.wav",
    )

    class FakeExamples:
        directory = Path("examples")

        def audio_paths(self, project_root: Path):
            assert project_root == tmp_path
            return list(example_paths)

    config = SimpleNamespace(
        top_k=2,
        examples=FakeExamples(),
    )
    dataset = SimpleNamespace(
        audio_paths=[Path("a.wav"), Path("b.wav")],
        labels=["C_maj", "D_min"],
    )
    split = SimpleNamespace(
        train_paths=[Path("train.wav")],
        val_paths=[Path("val.wav")],
        test_paths=[Path("E_maj.wav")],
    )
    prepared = SimpleNamespace(
        dataset=dataset,
        split=split,
        x_test=np.zeros((1, 2, 12, 1), dtype=np.float32),
        y_test=np.array([0], dtype=np.int32),
    )
    model = object()

    class FakeTrainer:
        def __init__(self, received_config, received_dataset, project_root=None):
            assert received_config is config
            assert received_dataset is dataset
            assert project_root == tmp_path

        def train_or_load(self, received_prepared):
            assert received_prepared is prepared
            return SimpleNamespace(
                model=model,
                signature="abc123",
                model_path=tmp_path / "logs" / "best_model.keras",
                cache_hit=True,
            )

    class FakeEvaluator:
        def evaluate(self, received_model, x_test, y_test, labels):
            assert received_model is model
            assert labels == ["C_maj", "D_min"]
            return SimpleNamespace(
                test_accuracy=0.9,
                test_loss=0.1,
                f1_macro=0.8,
                report="classification report",
            )

    class FakePredictor:
        def __init__(self, received_config, _extractor, labels):
            assert received_config is config
            assert labels == ["C_maj", "D_min"]

        def predict(self, received_model, audio_path):
            assert received_model is model
            assert audio_path in example_paths
            return [("C_maj", 0.75), ("D_min", 0.25)]

    monkeypatch.setattr(training.MusicaConfig, "load", lambda path: config)
    monkeypatch.setattr(
        training,
        "prepare_data",
        lambda received_config, project_root=None: prepared,
    )
    monkeypatch.setattr(training, "ChordTrainer", FakeTrainer)
    monkeypatch.setattr(training, "ChordEvaluator", FakeEvaluator)
    monkeypatch.setattr(training, "ChordPredictor", FakePredictor)
    monkeypatch.setattr(training, "FeatureExtractor", lambda *_args, **_kwargs: object())

    training.run_pipeline(project_root=tmp_path)

    output = capsys.readouterr().out
    assert "Signature: abc123" in output
    assert "Modele reutilise: logs/best_model.keras" in output
    assert "Test accuracy : 0.9000" in output
    assert f"Audio exemple : {example_paths[0]}" in output
    assert f"Audio exemple : {example_paths[1]}" in output
    assert "C_maj    0.750" in output
