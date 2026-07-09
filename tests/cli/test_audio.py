import pytest

from musica.cli import build_parser


def test_audio_commands_are_registered() -> None:
    parser = build_parser()

    assert parser.parse_args(["generate-midi"]).command == "generate-midi"
    assert parser.parse_args(["generate-wav"]).command == "generate-wav"
    assert parser.parse_args(["download-noises"]).command == "download-noises"
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
