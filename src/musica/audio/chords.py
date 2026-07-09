"""Utilities for generating chord MIDI files."""

from __future__ import annotations

import csv
import hashlib
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pretty_midi

from musica.modeling.config import MusicaConfig

CHORD_PATTERNS: dict[str, list[int]] = {
    "maj": [0, 4, 7],
    "min": [0, 3, 7],
    "dim": [0, 3, 6],
}

ROOT_NOTES: dict[str, int] = {
    "C": 60,
    "C#": 61,
    "D": 62,
    "D#": 63,
    "E": 64,
    "F": 65,
    "F#": 66,
    "G": 67,
    "G#": 68,
    "A": 69,
    "A#": 70,
    "B": 71,
}

MIDI_MIN_NOTE = 21
MIDI_MAX_NOTE = 108
DEFAULT_CONFIG = MusicaConfig()


@dataclass(frozen=True)
class GeneratedMidi:
    path: Path
    root: str
    quality: str
    instrument: str
    octave_offset: int
    velocity: int
    duration: float
    repetition: int


@dataclass(frozen=True)
class GeneratedAudio:
    path: Path
    root: str
    quality: str
    instrument: str
    octave_offset: int
    velocity: int
    duration: float
    repetition: int
    renderer: str


@dataclass(frozen=True)
class ChordSpec:
    root: str
    quality: str
    instrument: str
    octave_offset: int
    velocity: int
    duration: float = 1.5
    repetition: int = 1


@dataclass(frozen=True)
class HumanizationConfig:
    enabled: bool = DEFAULT_CONFIG.humanize
    onset_spread_seconds: float = DEFAULT_CONFIG.humanize_onset_ms / 1000.0
    velocity_jitter: int = DEFAULT_CONFIG.velocity_jitter
    lower_bass_probability: float = 0.6
    upper_octave_probability: float = 0.35


def safe_key_name(root: str) -> str:
    return root.replace("#", "sharp")


def iter_chord_specs(
        *,
        octave_offsets: tuple[int, ...] = DEFAULT_CONFIG.octave_offsets,
        velocities: tuple[int, ...] = DEFAULT_CONFIG.velocities,
        durations: tuple[float, ...] = (DEFAULT_CONFIG.chord_duration,),
        repetitions: int = DEFAULT_CONFIG.repetitions,
        instrument_programs: dict[str, int] | None = None,
) -> list[ChordSpec]:
    if repetitions < 1:
        raise ValueError("repetitions must be at least 1")

    instruments = instrument_programs or DEFAULT_CONFIG.instrument_programs
    specs: list[ChordSpec] = []
    for root in ROOT_NOTES:
        for quality in CHORD_PATTERNS:
            for instrument in instruments:
                for octave_offset in octave_offsets:
                    for velocity in velocities:
                        for duration in durations:
                            if duration <= 0:
                                raise ValueError("durations must be positive")
                            for repetition in range(1, repetitions + 1):
                                specs.append(
                                    ChordSpec(
                                        root=root,
                                        quality=quality,
                                        instrument=instrument,
                                        octave_offset=octave_offset,
                                        velocity=velocity,
                                        duration=duration,
                                        repetition=repetition,
                                    )
                                )
    return specs


def generate_chord_midi(
        root: str,
        quality: str,
        instrument: str,
        *,
        duration: float = DEFAULT_CONFIG.chord_duration,
        velocity: int = 100,
        octave_offset: int = 0,
        humanization: HumanizationConfig | None = None,
        random_seed: int | None = 0,
        instrument_programs: dict[str, int] | None = None,
) -> pretty_midi.PrettyMIDI:
    midi = pretty_midi.PrettyMIDI()
    instruments = instrument_programs or DEFAULT_CONFIG.instrument_programs
    midi_instrument = pretty_midi.Instrument(program=instruments[instrument])
    midi.instruments.append(midi_instrument)

    root_note = ROOT_NOTES[root] + (octave_offset * 12)
    config = humanization if humanization is not None else HumanizationConfig()
    rng = np.random.default_rng(random_seed)
    chord_pitches = build_chord_voicing(root_note, CHORD_PATTERNS[quality], config, rng)

    for note_number in chord_pitches:
        if MIDI_MIN_NOTE <= note_number <= MIDI_MAX_NOTE:
            start = 0.0
            note_velocity = velocity
            if config.enabled:
                onset_spread = max(0.0, min(config.onset_spread_seconds, duration))
                velocity_jitter = max(0, config.velocity_jitter)
                start = float(rng.uniform(0.0, onset_spread))
                note_velocity = clamp_midi_velocity(
                    velocity + int(rng.integers(-velocity_jitter, velocity_jitter + 1))
                )
            midi_instrument.notes.append(
                pretty_midi.Note(
                    velocity=note_velocity,
                    pitch=note_number,
                    start=start,
                    end=duration,
                )
            )

    return midi


def build_chord_voicing(
        root_note: int,
        chord_pattern: list[int],
        config: HumanizationConfig,
        rng: np.random.Generator,
) -> list[int]:
    pitches = [root_note + interval for interval in chord_pattern]
    if not config.enabled:
        return pitches

    inversion = int(rng.integers(0, len(pitches)))
    voicing = pitches[inversion:] + [pitch + 12 for pitch in pitches[:inversion]]
    voicing = keep_voicing_in_midi_range(voicing)

    doubled_notes: list[int] = []
    if rng.random() < config.lower_bass_probability:
        doubled_notes.append(voicing[0] - 12)
    if rng.random() < config.upper_octave_probability:
        doubled_notes.append(int(rng.choice(voicing)) + 12)

    voicing.extend(note for note in doubled_notes if MIDI_MIN_NOTE <= note <= MIDI_MAX_NOTE)
    return sorted(set(voicing))


def keep_voicing_in_midi_range(pitches: list[int]) -> list[int]:
    adjusted = pitches[:]
    while min(adjusted) < MIDI_MIN_NOTE:
        adjusted = [pitch + 12 for pitch in adjusted]
    while max(adjusted) > MIDI_MAX_NOTE:
        adjusted = [pitch - 12 for pitch in adjusted]
    return adjusted


def clamp_midi_velocity(velocity: int) -> int:
    return max(1, min(127, velocity))


def chord_spec_seed(base_seed: int | None, spec: ChordSpec) -> int | None:
    if base_seed is None:
        return None
    seed_material = (
        f"{base_seed}:{spec.root}:{spec.quality}:{spec.instrument}:"
        f"{spec.octave_offset}:{spec.velocity}:{spec.duration}:{spec.repetition}"
    )
    digest = hashlib.sha256(seed_material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def generated_chord_filename(
        spec: ChordSpec,
        suffix: str,
        *,
        include_variation_suffix: bool = False,
) -> str:
    filename = (
        f"{safe_key_name(spec.root)}_{spec.quality}_{spec.instrument}"
        f"_oct{spec.octave_offset}_vel{spec.velocity}"
    )
    if include_variation_suffix:
        duration_ms = int(round(spec.duration * 1000))
        filename = f"{filename}_dur{duration_ms}ms_rep{spec.repetition:03d}"
    return f"{filename}.{suffix}"


def generate_chord_midi_files(
        output_dir: Path,
        *,
        duration: float = DEFAULT_CONFIG.chord_duration,
        durations: tuple[float, ...] | None = None,
        repetitions: int = DEFAULT_CONFIG.repetitions,
        max_files: int | None = None,
        octave_offsets: tuple[int, ...] = DEFAULT_CONFIG.octave_offsets,
        velocities: tuple[int, ...] = DEFAULT_CONFIG.velocities,
        instrument_programs: dict[str, int] | None = None,
        humanization: HumanizationConfig | None = None,
        random_seed: int | None = 0,
) -> list[GeneratedMidi]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[GeneratedMidi] = []
    effective_durations = durations if durations is not None else (duration,)
    include_variation_suffix = durations is not None or repetitions > 1

    specs = iter_chord_specs(
        octave_offsets=octave_offsets,
        velocities=velocities,
        durations=effective_durations,
        repetitions=repetitions,
        instrument_programs=instrument_programs,
    )
    if max_files is not None:
        specs = specs[:max_files]

    for spec in specs:
        filename = generated_chord_filename(
            spec,
            "mid",
            include_variation_suffix=include_variation_suffix,
        )
        path = output_dir / filename
        midi = generate_chord_midi(
            spec.root,
            spec.quality,
            spec.instrument,
            duration=spec.duration,
            velocity=spec.velocity,
            octave_offset=spec.octave_offset,
            humanization=humanization,
            random_seed=chord_spec_seed(random_seed, spec),
            instrument_programs=instrument_programs,
        )
        midi.write(str(path))
        generated.append(
            GeneratedMidi(
                path=path,
                root=spec.root,
                quality=spec.quality,
                instrument=spec.instrument,
                octave_offset=spec.octave_offset,
                velocity=spec.velocity,
                duration=spec.duration,
                repetition=spec.repetition,
            )
        )

    return generated


def generate_chord_wav_files(
        output_dir: Path,
        *,
        duration: float = DEFAULT_CONFIG.chord_duration,
        durations: tuple[float, ...] | None = None,
        repetitions: int = DEFAULT_CONFIG.repetitions,
        max_files: int | None = None,
        sample_rate: int = DEFAULT_CONFIG.chord_sample_rate,
        octave_offsets: tuple[int, ...] = DEFAULT_CONFIG.octave_offsets,
        velocities: tuple[int, ...] = DEFAULT_CONFIG.velocities,
        instrument_programs: dict[str, int] | None = None,
        renderer: str = DEFAULT_CONFIG.renderer,
        soundfont_path: Path | None = None,
        humanization: HumanizationConfig | None = None,
        random_seed: int | None = 0,
) -> list[GeneratedAudio]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[GeneratedAudio] = []
    resolved_renderer = resolve_audio_renderer(renderer, soundfont_path)
    effective_durations = durations if durations is not None else (duration,)
    include_variation_suffix = durations is not None or repetitions > 1

    specs = iter_chord_specs(
        octave_offsets=octave_offsets,
        velocities=velocities,
        durations=effective_durations,
        repetitions=repetitions,
        instrument_programs=instrument_programs,
    )
    if max_files is not None:
        specs = specs[:max_files]

    for spec in specs:
        key_dir = output_dir / safe_key_name(spec.root)
        key_dir.mkdir(parents=True, exist_ok=True)
        filename = generated_chord_filename(
            spec,
            "wav",
            include_variation_suffix=include_variation_suffix,
        )
        path = key_dir / filename
        midi = generate_chord_midi(
            spec.root,
            spec.quality,
            spec.instrument,
            duration=spec.duration,
            velocity=spec.velocity,
            octave_offset=spec.octave_offset,
            humanization=humanization,
            random_seed=chord_spec_seed(random_seed, spec),
            instrument_programs=instrument_programs,
        )

        if resolved_renderer == "fluidsynth":
            write_wav_with_fluidsynth(
                midi,
                path,
                duration=spec.duration,
                sample_rate=sample_rate,
                soundfont_path=required_soundfont_path(soundfont_path),
            )
        else:
            import soundfile as sf

            audio = midi.synthesize(fs=sample_rate)
            audio = fit_audio_duration(audio, spec.duration, sample_rate)
            sf.write(path, audio, sample_rate, subtype="PCM_16")

        generated.append(
            GeneratedAudio(
                path=path,
                root=spec.root,
                quality=spec.quality,
                instrument=spec.instrument,
                octave_offset=spec.octave_offset,
                velocity=spec.velocity,
                duration=spec.duration,
                repetition=spec.repetition,
                renderer=resolved_renderer,
            )
        )

    return generated


def write_generated_audio_manifest(
        generated: list[GeneratedAudio],
        manifest_path: Path,
        *,
        base_dir: Path | None = None,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in generated:
        path = item.path
        if base_dir is not None:
            path = path.relative_to(base_dir)
        rows.append(
            {
                "path": str(path),
                "root_note": item.root,
                "quality": item.quality,
                "instrument": item.instrument,
                "octave_offset": item.octave_offset,
                "velocity": item.velocity,
                "duration": item.duration,
                "repetition": item.repetition,
                "renderer": item.renderer,
            }
        )

    with manifest_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "path",
                "root_note",
                "quality",
                "instrument",
                "octave_offset",
                "velocity",
                "duration",
                "repetition",
                "renderer",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def resolve_audio_renderer(renderer: str, soundfont_path: Path | None) -> str:
    if renderer not in {"auto", "pretty-midi", "fluidsynth"}:
        raise ValueError(f"Unsupported renderer: {renderer}")

    if renderer == "auto":
        if shutil.which("fluidsynth") and required_soundfont_path(soundfont_path).exists():
            return "fluidsynth"
        return "pretty-midi"

    if renderer == "fluidsynth" and not shutil.which("fluidsynth"):
        raise RuntimeError("fluidsynth is not installed or is not on PATH")

    return renderer


def required_soundfont_path(soundfont_path: Path | None) -> Path:
    if soundfont_path is not None:
        return soundfont_path
    return Path.cwd() / DEFAULT_CONFIG.soundfont_path


def fit_audio_duration(audio: np.ndarray, duration: float, sample_rate: int) -> np.ndarray:
    target_samples = max(1, int(round(duration * sample_rate)))
    if len(audio) > target_samples:
        return audio[:target_samples]
    if len(audio) < target_samples:
        pad_width = [(0, target_samples - len(audio))]
        pad_width.extend((0, 0) for _ in audio.shape[1:])
        return np.pad(audio, pad_width)
    return audio


def write_wav_with_fluidsynth(
        midi: pretty_midi.PrettyMIDI,
        output_path: Path,
        *,
        duration: float,
        sample_rate: int,
        soundfont_path: Path,
) -> None:
    import soundfile as sf

    if not soundfont_path.exists():
        raise FileNotFoundError(f"SoundFont not found: {soundfont_path}")

    with tempfile.NamedTemporaryFile(suffix=".mid") as midi_file:
        midi.write(midi_file.name)
        subprocess.run(
            [
                "fluidsynth",
                "-ni",
                "-F",
                str(output_path),
                "-r",
                str(sample_rate),
                str(soundfont_path),
                midi_file.name,
            ],
            check=True,
            capture_output=True,
        )
    audio, actual_sample_rate = sf.read(output_path)
    audio = fit_audio_duration(audio, duration, actual_sample_rate)
    sf.write(output_path, audio, actual_sample_rate, subtype="PCM_16")
