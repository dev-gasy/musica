from pathlib import Path

import numpy as np
import pytest

from musica.modeling import (
    ChordDataset,
    ChordTrainer,
    DatasetSplit,
    ExampleAudioConfig,
    FeatureExtractor,
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
                "logs_dir = 'logs/custom'",
                "[features]",
                "sample_rate = 16000",
                "cache_features = false",
                "feature_cache_dir = 'cache/features'",
                "[dataset]",
                "dataset_dir = 'sectioned/chords'",
                "recorded_audio_dir = 'sectioned/recorded'",
                "[examples]",
                "directory = 'examples'",
                "[audio]",
                "midi_output_dir = 'sectioned/midi'",
                "clean_output_dir = 'sectioned/clean'",
                "instrument_programs = { piano = 0, organ = 19 }",
                "octave_offsets = [-1, 0]",
                "velocities = [90, 110]",
                "[noise]",
                "noise_snrs_db = [9.0, 12.0]",
                "noise_download_dir = 'sectioned/noises'",
                "noisy_output_dir = 'sectioned/noisy'",
                "[realism]",
                "realistic_output_dir = 'sectioned/realistic'",
                "[transpose]",
                "transposed_output_dir = 'sectioned/transposed'",
                "[manifest]",
                "audio_manifest_path = 'sectioned/manifest.csv'",
                "[prediction]",
                "top_k = 4",
            ]
        )
    )

    config = MusicaConfig.load(config_path)

    assert config.seed == 11
    assert config.epochs == 5
    assert config.batch_size == 8
    assert config.logs_dir == Path("logs/custom")
    assert config.sample_rate == 16000
    assert config.cache_features is False
    assert config.dataset_dir == Path("sectioned/chords")
    assert config.recorded_audio_dir == Path("sectioned/recorded")
    assert config.feature_cache_dir == Path("cache/features")
    assert config.examples.directory == Path("examples")
    assert config.midi_output_dir == Path("sectioned/midi")
    assert config.clean_output_dir == Path("sectioned/clean")
    assert config.instrument_programs == {"piano": 0, "organ": 19}
    assert config.octave_offsets == (-1, 0)
    assert config.velocities == (90, 110)
    assert config.noise_snrs_db == (9.0, 12.0)
    assert config.noise_download_dir == Path("sectioned/noises")
    assert config.noisy_output_dir == Path("sectioned/noisy")
    assert config.realistic_output_dir == Path("sectioned/realistic")
    assert config.transposed_output_dir == Path("sectioned/transposed")
    assert config.audio_manifest_path == Path("sectioned/manifest.csv")
    assert config.top_k == 4


def test_config_coerces_path_values_with_pydantic() -> None:
    config = MusicaConfig(
        dataset_dir="custom/chords",
        examples={"directory": "examples"},
    )

    assert config.dataset_dir == Path("custom/chords")
    assert config.examples.directory == Path("examples")


def test_example_audio_config_lists_audio_files(tmp_path: Path) -> None:
    examples_dir = tmp_path / "examples"
    examples_dir.mkdir()
    (examples_dir / "C_maj.wav").write_bytes(b"audio")
    (examples_dir / "D_min.mp3").write_bytes(b"audio")
    (examples_dir / "notes.txt").write_text("ignore")

    examples = ExampleAudioConfig(directory=Path("examples"))

    assert examples.audio_paths(tmp_path) == [
        examples_dir / "C_maj.wav",
        examples_dir / "D_min.mp3",
    ]


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


def test_config_rejects_legacy_paths_section(tmp_path: Path) -> None:
    config_path = tmp_path / "musica.toml"
    config_path.write_text("[paths]\ndataset_dir = 'legacy/chords'\n")

    with pytest.raises(ValueError, match="Unknown config keys: paths"):
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


def test_feature_extractor_reuses_cached_features(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    write_fake_dataset(tmp_path, labels=("C_maj",))
    config = MusicaConfig(feature_cache_dir=Path("cache/features"))
    dataset = ChordDataset(config, project_root=tmp_path).discover()
    paths = dataset.audio_paths[:2]
    calls = 0

    def fake_audio_features(_path: Path) -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.full((3, 12, 1), calls, dtype=np.float32)

    extractor = FeatureExtractor(config, dataset, project_root=tmp_path)
    monkeypatch.setattr(extractor, "audio_features", fake_audio_features)

    x_first, y_first = extractor.load_features(paths, split_name="train")
    cache_files = sorted((tmp_path / "cache" / "features").glob("train-*.npz"))

    assert calls == 2
    assert len(cache_files) == 1

    monkeypatch.setattr(
        extractor,
        "audio_features",
        lambda _path: pytest.fail("features should be loaded from cache"),
    )

    x_second, y_second = extractor.load_features(paths, split_name="train")

    np.testing.assert_array_equal(x_second, x_first)
    np.testing.assert_array_equal(y_second, y_first)


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
