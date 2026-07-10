"""Audio dataset bootstrap helpers shared by scripts and notebooks."""

from __future__ import annotations

import filecmp
import shutil
from dataclasses import dataclass
from pathlib import Path

from musica.audio.chords import (
    HumanizationConfig,
    generate_chord_wav_files,
    write_generated_audio_manifest,
)
from musica.audio.io import list_wav_files
from musica.augmentation.noise import augment_wav_dataset_with_noise
from musica.augmentation.realism import augment_wav_dataset_with_realism
from musica.config import MusicaConfig


@dataclass(frozen=True)
class AudioBootstrapResult:
    dataset_dir: Path
    output_dir: Path
    manifest_path: Path | None
    wav_count: int
    generated_count: int
    renderer: str | None
    noisy_output_dir: Path | None = None
    noisy_generated_count: int = 0
    realistic_output_dir: Path | None = None
    realistic_generated_count: int = 0
    recorded_source_dir: Path | None = None
    recorded_dir: Path | None = None
    recorded_copied_count: int = 0

    @property
    def generated(self) -> bool:
        return self.generated_count > 0


def ensure_training_audio_dataset(
        config: MusicaConfig,
        project_root: Path,
        *,
        renderer: str | None = None,
        max_files: int | None = None,
        force: bool = False,
        include_derived: bool = False,
) -> AudioBootstrapResult:
    """Generate clean chord WAV files when the configured dataset is empty."""
    dataset_dir = config.resolve_path(project_root, config.dataset_dir)
    output_dir = config.resolve_path(project_root, config.clean_output_dir)
    noisy_output_dir = config.resolve_path(project_root, config.noisy_output_dir)
    realistic_output_dir = config.resolve_path(project_root, config.realistic_output_dir)
    recorded_source_dir = project_root / "assets" / "recorded"
    recorded_dir = config.resolve_path(project_root, config.recorded_audio_dir)
    recorded_copied_count = sync_recorded_assets(recorded_source_dir, recorded_dir)

    clean_wavs = list_wav_files(output_dir) if output_dir.exists() else []
    if clean_wavs and not force:
        noisy_generated_count = ensure_noisy_wavs(
            config,
            project_root,
            output_dir,
            noisy_output_dir,
            max_files=max_files,
            force=force,
        ) if include_derived else 0
        realistic_generated_count = ensure_realistic_wavs(
            config,
            output_dir,
            realistic_output_dir,
            max_files=max_files,
            force=force,
        ) if include_derived else 0
        wav_count = len(list_wav_files(dataset_dir)) if dataset_dir.exists() else len(clean_wavs)
        return AudioBootstrapResult(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            manifest_path=None,
            wav_count=wav_count,
            generated_count=0,
            renderer=None,
            noisy_output_dir=noisy_output_dir,
            noisy_generated_count=noisy_generated_count,
            realistic_output_dir=realistic_output_dir,
            realistic_generated_count=realistic_generated_count,
            recorded_source_dir=recorded_source_dir,
            recorded_dir=recorded_dir,
            recorded_copied_count=recorded_copied_count,
        )

    soundfont_path = config.resolve_path(project_root, config.soundfont_path)
    generated = generate_chord_wav_files(
        output_dir,
        duration=config.chord_duration,
        repetitions=config.repetitions,
        max_files=max_files,
        sample_rate=config.chord_sample_rate,
        octave_offsets=config.octave_offsets,
        velocities=config.velocities,
        instrument_programs=config.instrument_programs,
        renderer=renderer or config.renderer,
        soundfont_path=soundfont_path,
        humanization=HumanizationConfig(
            enabled=config.humanize,
            onset_spread_seconds=max(0.0, config.humanize_onset_ms) / 1000.0,
            velocity_jitter=max(0, config.velocity_jitter),
        ),
        random_seed=config.seed,
    )
    manifest_path = output_dir / "manifest.csv"
    write_generated_audio_manifest(generated, manifest_path)
    noisy_generated_count = ensure_noisy_wavs(
        config,
        project_root,
        output_dir,
        noisy_output_dir,
        max_files=max_files,
        force=force,
    ) if include_derived else 0
    realistic_generated_count = ensure_realistic_wavs(
        config,
        output_dir,
        realistic_output_dir,
        max_files=max_files,
        force=force,
    ) if include_derived else 0
    wav_count = len(list_wav_files(dataset_dir)) if dataset_dir.exists() else len(generated)
    resolved_renderer = generated[0].renderer if generated else renderer or config.renderer

    return AudioBootstrapResult(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        manifest_path=manifest_path,
        wav_count=wav_count,
        generated_count=len(generated),
        renderer=resolved_renderer,
        noisy_output_dir=noisy_output_dir,
        noisy_generated_count=noisy_generated_count,
        realistic_output_dir=realistic_output_dir,
        realistic_generated_count=realistic_generated_count,
        recorded_source_dir=recorded_source_dir,
        recorded_dir=recorded_dir,
        recorded_copied_count=recorded_copied_count,
    )


def sync_recorded_assets(source_dir: Path, target_dir: Path) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)
    if not source_dir.exists():
        return 0

    copied = 0
    for source_path in sorted(source_dir.rglob("*")):
        if not source_path.is_file():
            continue
        target_path = target_dir / source_path.relative_to(source_dir)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if recorded_file_is_current(source_path, target_path):
            continue
        shutil.copy2(source_path, target_path)
        copied += 1
    return copied


def recorded_file_is_current(source_path: Path, target_path: Path) -> bool:
    if not target_path.exists():
        return False
    return filecmp.cmp(source_path, target_path, shallow=False)


def ensure_noisy_wavs(
        config: MusicaConfig,
        project_root: Path,
        clean_output_dir: Path,
        noisy_output_dir: Path,
        *,
        max_files: int | None,
        force: bool,
) -> int:
    existing_wavs = list_wav_files(noisy_output_dir) if noisy_output_dir.exists() else []
    if existing_wavs and not force:
        return 0

    noise_dir = config.resolve_path(project_root, config.noise_download_dir)
    generated = augment_wav_dataset_with_noise(
        clean_output_dir,
        noise_dir,
        noisy_output_dir,
        snrs_db=config.noise_snrs_db,
        mode=config.noise_mode,
        max_files=max_files,
        random_seed=config.seed,
    )
    return len(generated)


def ensure_realistic_wavs(
        config: MusicaConfig,
        clean_output_dir: Path,
        realistic_output_dir: Path,
        *,
        max_files: int | None,
        force: bool,
) -> int:
    existing_wavs = list_wav_files(realistic_output_dir) if realistic_output_dir.exists() else []
    if existing_wavs and not force:
        return 0

    generated = augment_wav_dataset_with_realism(
        clean_output_dir,
        realistic_output_dir,
        variants=config.realism_variants,
        sample_rates=config.realism_sample_rates,
        max_files=max_files,
        random_seed=config.seed,
        keep_duration=config.realism_keep_duration,
    )
    return len(generated)
