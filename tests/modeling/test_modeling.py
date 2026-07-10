from pathlib import Path

import numpy as np
import pytest

from musica.modeling import (
    ChordDataset,
    ChordTrainer,
    DatasetSplit,
    MusicaConfig,
    PreparedData,
    TranspositionAugmenter,
    label_sort_key,
)


def write_fake_dataset(root: Path, labels: tuple[str, ...] = ("C_maj", "D_min")) -> None:
    for label in labels:
        note = label.split("_")[0]
        for index in range(6):
            path = root / "audio" / "chords" / note / f"{label}_{index}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{label}-{index}".encode("utf-8"))


def test_config_loads_simple_toml(tmp_path: Path) -> None:
    config_path = tmp_path / "musica.toml"
    config_path.write_text(
        "\n".join(
            [
                "seed = 7",
                "epochs = 3",
                "batch_size = 4",
                "learning_rate = 0.01",
                "force_retrain = true",
                "val_ratio = 0.2",
                "test_ratio = 0.2",
                "sample_rate = 16000",
                "target_duration = 1.0",
                "hop_length = 256",
                "bins_per_octave = 12",
                "n_chroma = 12",
                "top_k = 2",
                "dataset_dir = 'custom/chords'",
                "noise_snrs_db = [12.0, 18.0]",
                "instrument_programs = { piano = 0, organ = 19 }",
            ]
        )
    )

    config = MusicaConfig.load(config_path)

    assert config.seed == 7
    assert config.epochs == 3
    assert config.batch_size == 4
    assert config.force_retrain is True
    assert config.top_k == 2
    assert config.dataset_dir == Path("custom/chords")
    assert config.noise_snrs_db == (12.0, 18.0)
    assert config.instrument_programs == {"piano": 0, "organ": 19}


def test_config_loads_sectioned_toml(tmp_path: Path) -> None:
    config_path = tmp_path / "musica.toml"
    config_path.write_text(
        "\n".join(
            [
                "[general]",
                "seed = 11",
                "[training]",
                "epochs = 5",
                "batch_size = 8",
                "[features]",
                "sample_rate = 16000",
                "[paths]",
                "dataset_dir = 'sectioned/chords'",
                "[audio]",
                "instrument_programs = { piano = 0, organ = 19 }",
                "octave_offsets = [-1, 0]",
                "velocities = [90, 110]",
                "[noise]",
                "noise_snrs_db = [9.0, 12.0]",
                "[prediction]",
                "top_k = 4",
            ]
        )
    )

    config = MusicaConfig.load(config_path)

    assert config.seed == 11
    assert config.epochs == 5
    assert config.batch_size == 8
    assert config.sample_rate == 16000
    assert config.dataset_dir == Path("sectioned/chords")
    assert config.instrument_programs == {"piano": 0, "organ": 19}
    assert config.octave_offsets == (-1, 0)
    assert config.velocities == (90, 110)
    assert config.noise_snrs_db == (9.0, 12.0)
    assert config.top_k == 4


def test_config_coerces_path_values_with_pydantic() -> None:
    config = MusicaConfig(dataset_dir="custom/chords")

    assert config.dataset_dir == Path("custom/chords")


def test_config_validates_direct_instantiation() -> None:
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        MusicaConfig(epochs=0)


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "musica.toml"
    config_path.write_text("seed = 0\nnot_a_config = true\n")

    with pytest.raises(ValueError, match="Unknown config keys"):
        MusicaConfig.load(config_path)


def test_config_rejects_unknown_section_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "musica.toml"
    config_path.write_text("[training]\nepochs = 2\nnot_a_config = true\n")

    with pytest.raises(ValueError, match="training.not_a_config"):
        MusicaConfig.load(config_path)


def test_labels_sort_in_musical_order() -> None:
    labels = ["D_min", "C#_dim", "C_maj", "C_min"]

    assert sorted(labels, key=label_sort_key) == [
        "C_maj",
        "C_min",
        "C#_dim",
        "D_min",
    ]


def test_stratified_split_is_reproducible(tmp_path: Path) -> None:
    write_fake_dataset(tmp_path)
    config = MusicaConfig(seed=42, val_ratio=0.2, test_ratio=0.2)

    first = ChordDataset(config, project_root=tmp_path).discover().split()
    second = ChordDataset(config, project_root=tmp_path).discover().split()

    assert first.train_paths == second.train_paths
    assert first.val_paths == second.val_paths
    assert first.test_paths == second.test_paths
    assert len(first.train_paths) == 8
    assert len(first.val_paths) == 2
    assert len(first.test_paths) == 2


def test_training_signature_is_stable_and_config_sensitive(tmp_path: Path) -> None:
    write_fake_dataset(tmp_path)
    config = MusicaConfig(seed=0, epochs=2)
    dataset = ChordDataset(config, project_root=tmp_path).discover()
    split = dataset.split()

    signature = ChordTrainer(config, dataset, project_root=tmp_path).signature(split)
    same_signature = ChordTrainer(config, dataset, project_root=tmp_path).signature(split)
    changed_signature = ChordTrainer(
        MusicaConfig(seed=0, epochs=3),
        dataset,
        project_root=tmp_path,
    ).signature(split)

    assert signature == same_signature
    assert signature != changed_signature


def test_training_signature_changes_with_callback_strategy(tmp_path: Path) -> None:
    write_fake_dataset(tmp_path)
    config = MusicaConfig(seed=0, epochs=60, early_stopping_patience=8)
    dataset = ChordDataset(config, project_root=tmp_path).discover()
    split = dataset.split()

    signature = ChordTrainer(config, dataset, project_root=tmp_path).signature(split)
    changed_signature = ChordTrainer(
        MusicaConfig(seed=0, epochs=60, early_stopping_patience=15),
        dataset,
        project_root=tmp_path,
    ).signature(split)

    assert signature != changed_signature


def test_train_or_load_cache_hit_does_not_fit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = MusicaConfig()
    dataset = ChordDataset(config, project_root=tmp_path)
    dataset.audio_paths = []
    dataset.labels = ["C_maj"]
    split = DatasetSplit([], [], [])
    prepared = PreparedData(
        dataset=dataset,
        split=split,
        x_train=np.zeros((1, 2, 12, 1), dtype=np.float32),
        y_train=np.array([0], dtype=np.int32),
        x_val=np.zeros((1, 2, 12, 1), dtype=np.float32),
        y_val=np.array([0], dtype=np.int32),
        x_test=np.zeros((1, 2, 12, 1), dtype=np.float32),
        y_test=np.array([0], dtype=np.int32),
    )
    trainer = ChordTrainer(config, dataset, project_root=tmp_path)
    fake_model = object()

    monkeypatch.setattr(trainer, "signature", lambda _split: "abc123")
    monkeypatch.setattr(
        trainer,
        "load_cached_model",
        lambda _signature: (
            fake_model,
            tmp_path / "logs" / "models" / "abc123" / "best_model.keras",
            tmp_path / "logs" / "models" / "abc123" / "training_log.csv",
        ),
    )
    monkeypatch.setattr(
        trainer,
        "build_model",
        lambda *_args, **_kwargs: pytest.fail("fit path should not be used on cache hit"),
    )

    result = trainer.train_or_load(prepared)

    assert result.cache_hit is True
    assert result.model is fake_model
    assert result.signature == "abc123"


def test_transposition_augmenter_relabels_roots() -> None:
    x = np.zeros((1, 2, 12, 1), dtype=np.float32)
    y = np.array([0], dtype=np.int32)

    x_aug, y_aug = TranspositionAugmenter().augment(x, y, shifts=(0, 2))

    assert x_aug.shape == (2, 2, 12, 1)
    assert y_aug.tolist() == [0, 6]
