from pathlib import Path
from types import SimpleNamespace

import pytest

from musica.config import MusicaConfig
from musica.workflows import audio_bootstrap


def test_audio_bootstrap_skips_generation_when_dataset_has_wavs(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
) -> None:
    wav_path = tmp_path / "audio" / "chords" / "clean" / "C" / "C_maj.wav"
    wav_path.parent.mkdir(parents=True)
    wav_path.write_bytes(b"wav")
    config = MusicaConfig(
        dataset_dir=Path("audio/chords"),
        clean_output_dir=Path("audio/chords/clean"),
    )

    def fail_generate(*_args, **_kwargs):
        raise AssertionError("generation should not run")

    monkeypatch.setattr(audio_bootstrap, "generate_chord_wav_files", fail_generate)

    result = audio_bootstrap.ensure_training_audio_dataset(config, tmp_path)

    assert result.generated is False
    assert result.wav_count == 1
    assert result.dataset_dir == tmp_path / "audio" / "chords"


def test_audio_bootstrap_generates_clean_wavs_when_dataset_is_empty(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
) -> None:
    calls = []
    config = MusicaConfig(
        dataset_dir=Path("audio/chords"),
        clean_output_dir=Path("audio/chords/clean"),
        soundfont_path=Path("assets/soundfonts/FluidR3_GM.sf2"),
    )

    def fake_generate(output_dir, **kwargs):
        calls.append((output_dir, kwargs))
        wav_path = output_dir / "C" / "C_maj.wav"
        wav_path.parent.mkdir(parents=True)
        wav_path.write_bytes(b"wav")
        return [
            SimpleNamespace(
                path=wav_path,
                root="C",
                quality="maj",
                instrument="piano",
                octave_offset=0,
                velocity=100,
                duration=1.5,
                repetition=1,
                renderer="pretty-midi",
            )
        ]

    monkeypatch.setattr(audio_bootstrap, "generate_chord_wav_files", fake_generate)

    result = audio_bootstrap.ensure_training_audio_dataset(
        config,
        tmp_path,
        renderer="pretty-midi",
    )

    assert result.generated is True
    assert result.generated_count == 1
    assert result.wav_count == 1
    assert result.renderer == "pretty-midi"
    assert result.manifest_path == tmp_path / "audio" / "chords" / "clean" / "manifest.csv"
    assert result.manifest_path.exists()
    assert calls[0][0] == tmp_path / "audio" / "chords" / "clean"
    assert calls[0][1]["soundfont_path"] == tmp_path / "assets" / "soundfonts" / "FluidR3_GM.sf2"


def test_audio_bootstrap_can_generate_default_derived_layout(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
) -> None:
    calls = []
    config = MusicaConfig(
        dataset_dir=Path("audio/chords"),
        clean_output_dir=Path("audio/chords/clean"),
        noisy_output_dir=Path("audio/chords/noisy"),
        realistic_output_dir=Path("audio/chords/realistic"),
        recorded_audio_dir=Path("audio/chords/recorded"),
    )

    def fake_generate(output_dir, **_kwargs):
        wav_path = output_dir / "C" / "C_maj.wav"
        wav_path.parent.mkdir(parents=True)
        wav_path.write_bytes(b"clean")
        return [
            SimpleNamespace(
                path=wav_path,
                root="C",
                quality="maj",
                instrument="piano",
                octave_offset=0,
                velocity=100,
                duration=1.5,
                repetition=1,
                renderer="pretty-midi",
            )
        ]

    def fake_noise(input_dir, noise_dir, output_dir, **kwargs):
        calls.append(("noise", input_dir, noise_dir, output_dir, kwargs))
        wav_path = output_dir / "C" / "C_maj_noise-room.wav"
        wav_path.parent.mkdir(parents=True)
        wav_path.write_bytes(b"noisy")
        return [object()]

    def fake_realism(input_dir, output_dir, **kwargs):
        calls.append(("realism", input_dir, output_dir, kwargs))
        wav_path = output_dir / "C" / "C_maj_real_v001.wav"
        wav_path.parent.mkdir(parents=True)
        wav_path.write_bytes(b"realistic")
        return [object(), object()]

    monkeypatch.setattr(audio_bootstrap, "generate_chord_wav_files", fake_generate)
    monkeypatch.setattr(audio_bootstrap, "augment_wav_dataset_with_noise", fake_noise)
    monkeypatch.setattr(audio_bootstrap, "augment_wav_dataset_with_realism", fake_realism)

    result = audio_bootstrap.ensure_training_audio_dataset(
        config,
        tmp_path,
        renderer="pretty-midi",
        include_derived=True,
    )

    assert result.generated_count == 1
    assert result.noisy_generated_count == 1
    assert result.realistic_generated_count == 2
    assert result.recorded_dir == tmp_path / "audio" / "chords" / "recorded"
    assert result.recorded_dir.is_dir()
    assert calls[0][0] == "noise"
    assert calls[1][0] == "realism"


def test_audio_bootstrap_copies_recorded_assets_to_audio_dataset(
        tmp_path: Path,
) -> None:
    source_path = tmp_path / "assets" / "recorded" / "Ab_min_take001.wav"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"recorded")
    config = MusicaConfig(
        dataset_dir=Path("audio/chords"),
        clean_output_dir=Path("audio/chords/clean"),
        recorded_audio_dir=Path("audio/chords/recorded"),
    )

    result = audio_bootstrap.ensure_training_audio_dataset(
        config,
        tmp_path,
        max_files=1,
        include_derived=False,
    )

    target_path = tmp_path / "audio" / "chords" / "recorded" / "Ab_min_take001.wav"
    assert result.recorded_source_dir == tmp_path / "assets" / "recorded"
    assert result.recorded_dir == tmp_path / "audio" / "chords" / "recorded"
    assert result.recorded_copied_count == 1
    assert target_path.read_bytes() == b"recorded"

    second_result = audio_bootstrap.ensure_training_audio_dataset(
        config,
        tmp_path,
        max_files=1,
        include_derived=False,
    )

    assert second_result.recorded_copied_count == 0
