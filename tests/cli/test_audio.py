import argparse
from pathlib import Path

import pytest

from musica.cli import build_parser, run_download_assets, run_download_soundfont
from musica.config import MusicaConfig


def test_audio_commands_are_registered() -> None:
    parser = build_parser()

    assert parser.parse_args(["generate-midi"]).command == "generate-midi"
    assert parser.parse_args(["generate-wav"]).command == "generate-wav"
    assert parser.parse_args(["download-noises"]).command == "download-noises"
    assert parser.parse_args(["download-soundfont"]).command == "download-soundfont"
    assert parser.parse_args(["download-assets"]).command == "download-assets"
    assert parser.parse_args(["augment-noise"]).command == "augment-noise"
    assert parser.parse_args(["augment-realistic"]).command == "augment-realistic"
    assert parser.parse_args(["augment-transpose"]).command == "augment-transpose"
    assert parser.parse_args(["build-manifest"]).command == "build-manifest"


@pytest.mark.parametrize(
    "command",
    [
        "prepare-data",
        "report-data",
        "train",
        "evaluate",
        "predict",
        "predict-batch",
        "optimize-ensemble",
        "download-guitarset",
        "build-guitarset-manifest",
        "download-musicnet",
        "build-musicnet-manifest",
    ],
)
def test_model_and_external_dataset_commands_are_removed(command: str) -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([command])


def test_generate_wav_accepts_audio_dataset_args() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "generate-wav",
            "--durations",
            "1.0,1.5",
            "--repetitions",
            "2",
            "--renderer",
            "pretty-midi",
            "--no-humanize",
            "--no-manifest",
        ]
    )

    assert args.durations == (1.0, 1.5)
    assert args.repetitions == 2
    assert args.renderer == "pretty-midi"
    assert args.humanize is False
    assert args.no_manifest is True


def test_download_soundfont_uses_default_url_and_path() -> None:
    parser = build_parser()

    args = parser.parse_args(["download-soundfont"])

    assert args.output_path.name == "FluidR3_GM.sf2"
    assert args.url == "https://musical-artifacts.com/artifacts/738/FluidR3_GM.sf2"
    assert args.overwrite is False


def test_audio_command_defaults_use_feature_first_config() -> None:
    config = MusicaConfig(
        midi_output_dir=Path("custom/midi"),
        clean_output_dir=Path("custom/clean"),
        soundfont_path=Path("custom/soundfont.sf2"),
        noise_download_dir=Path("custom/noises"),
        noisy_output_dir=Path("custom/noisy"),
        realistic_output_dir=Path("custom/realistic"),
        transposed_output_dir=Path("custom/transposed"),
        audio_manifest_path=Path("custom/manifest.csv"),
    )
    parser = build_parser(config)

    assert parser.parse_args(["generate-midi"]).output_dir == Path("custom/midi")
    wav_args = parser.parse_args(["generate-wav"])
    assert wav_args.output_dir == Path("custom/clean")
    assert wav_args.soundfont == Path("custom/soundfont.sf2")
    assert parser.parse_args(["download-noises"]).output_dir == Path("custom/noises")
    asset_args = parser.parse_args(["download-assets"])
    assert asset_args.soundfont_output_path == Path("custom/soundfont.sf2")
    assert asset_args.noise_output_dir == Path("custom/noises")
    noise_args = parser.parse_args(["augment-noise"])
    assert noise_args.input_dir == Path("custom/clean")
    assert noise_args.noise_dir == Path("custom/noises")
    assert noise_args.output_dir == Path("custom/noisy")
    realistic_args = parser.parse_args(["augment-realistic"])
    assert realistic_args.input_dir == Path("custom/clean")
    assert realistic_args.output_dir == Path("custom/realistic")
    transpose_args = parser.parse_args(["augment-transpose"])
    assert transpose_args.input_dir == Path("custom/clean")
    assert transpose_args.output_dir == Path("custom/transposed")
    assert parser.parse_args(["build-manifest"]).output_path == Path("custom/manifest.csv")


def test_download_soundfont_handler_creates_parent_and_downloads(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def fake_download_url(url, output_path):
        calls.append((url, output_path))
        output_path.write_bytes(b"sf2")

    monkeypatch.setattr("musica.cli.download_url", fake_download_url)
    output_path = tmp_path / "assets" / "soundfonts" / "FluidR3_GM.sf2"
    args = argparse.Namespace(
        output_path=output_path,
        url="https://example.test/FluidR3_GM.sf2",
        overwrite=False,
    )

    run_download_soundfont(args, MusicaConfig())

    assert calls == [("https://example.test/FluidR3_GM.sf2", output_path)]
    assert output_path.read_bytes() == b"sf2"


def test_download_assets_accepts_unified_download_args() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "download-assets",
            "--soundfont-output-path",
            "custom/FluidR3_GM.sf2",
            "--soundfont-url",
            "https://example.test/font.sf2",
            "--noise-output-dir",
            "custom/noises",
            "--noise-url",
            "https://example.test/noise.wav",
            "--overwrite",
        ]
    )

    assert args.command == "download-assets"
    assert args.soundfont_output_path.name == "FluidR3_GM.sf2"
    assert args.soundfont_url == "https://example.test/font.sf2"
    assert args.noise_output_dir.as_posix() == "custom/noises"
    assert args.noise_url == ["https://example.test/noise.wav"]
    assert args.overwrite is True
    assert args.skip_soundfont is False
    assert args.skip_noises is False


def test_download_assets_handler_downloads_soundfont_and_noises(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    soundfont_calls = []
    noise_calls = []

    def fake_download_soundfont(output_path, *, url, overwrite=False):
        soundfont_calls.append((output_path, url, overwrite))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"sf2")
        return True

    def fake_download_noise_wavs(output_dir, *, sources, overwrite=False, manifest_path=None):
        noise_calls.append((output_dir, tuple(source.url for source in sources), overwrite, manifest_path))
        output_dir.mkdir(parents=True, exist_ok=True)
        return [object()]

    monkeypatch.setattr("musica.cli.download_soundfont", fake_download_soundfont)
    monkeypatch.setattr("musica.cli.download_noise_wavs", fake_download_noise_wavs)
    soundfont_path = tmp_path / "assets" / "soundfonts" / "FluidR3_GM.sf2"
    noise_dir = tmp_path / "assets" / "noises"
    manifest_path = tmp_path / "assets" / "noises" / "manifest.csv"
    args = argparse.Namespace(
        soundfont_output_path=soundfont_path,
        soundfont_url="https://example.test/font.sf2",
        noise_output_dir=noise_dir,
        noise_url=["https://example.test/noise.wav"],
        noise_manifest_path=manifest_path,
        overwrite=True,
        skip_soundfont=False,
        skip_noises=False,
    )

    run_download_assets(args, MusicaConfig())

    assert soundfont_calls == [
        (soundfont_path, "https://example.test/font.sf2", True)
    ]
    assert noise_calls == [
        (noise_dir, ("https://example.test/noise.wav",), True, manifest_path)
    ]
