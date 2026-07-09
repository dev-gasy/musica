# Scripts Musica

Ce document decrit la surface active de Musica apres recentrage sur la construction de datasets audio annotes.

## Dossiers

- `audio/chords/clean/`: WAV d'accords generes.
- `audio/chords/midi/`: MIDI d'accords generes.
- `audio/chords/noisy/`: variantes avec bruit.
- `audio/chords/realistic/`: variantes realistes.
- `audio/chords/transposed/`: variantes transposees et relabelisees.
- `audio/chords/recorded/`: WAV locaux enregistres manuellement, nommes comme `C_maj.wav`.
- `audio/manifest.csv`: manifest global compile par `build-manifest`.
- `assets/noises/`: bruit source et manifests de telechargement.
- `archive/modeling/`: ancien code modele/R&D, hors package actif.

Les sorties `audio/` restent ignorees par Git.

## Aide CLI

```bash
uv run musica --help
uv run musica generate-midi --help
uv run musica generate-wav --help
uv run musica download-noises --help
uv run musica augment-noise --help
uv run musica augment-realistic --help
uv run musica augment-transpose --help
uv run musica build-manifest --help
```

## Generation MIDI

```bash
uv run musica generate-midi --output-dir audio/chords/midi
```

Par defaut, les accords sont humanises de maniere reproductible. Pour des blocs exacts:

```bash
uv run musica generate-midi --no-humanize
```

Pour augmenter les variantes:

```bash
uv run musica generate-midi --durations 1.0,1.5,2.0 --repetitions 3 --output-dir audio/chords/midi_extended
```

## Generation WAV

```bash
uv run musica generate-wav --output-dir audio/chords/clean
```

Smoke test rapide:

```bash
uv run musica generate-wav --output-dir audio/chords/clean --duration 0.5 --max-files 6 --renderer pretty-midi
```

Forcer FluidSynth:

```bash
uv run musica generate-wav --renderer fluidsynth --soundfont assets/soundfonts/FluidR3_GM.sf2 --output-dir audio/chords/clean
```

Ecrire le manifest ailleurs, ou ne pas l'ecrire:

```bash
uv run musica generate-wav --manifest-path audio/chords/manifest.csv
uv run musica generate-wav --no-manifest
```

## Bruit

Telecharger les exemples de bruit integres:

```bash
uv run musica download-noises --output-dir assets/noises/internet
```

Telecharger un bruit personnalise:

```bash
uv run musica download-noises --url https://example.com/noise.wav --output-dir assets/noises/custom
```

Mixer les WAV propres avec du bruit:

```bash
uv run musica augment-noise --input-dir audio/chords/clean --noise-dir assets/noises/internet --output-dir audio/chords/noisy --snrs-db 15
```

Generer plusieurs SNRs:

```bash
uv run musica augment-noise --snrs-db 20,10,5 --mode all --output-dir audio/chords/noisy_large
```

## Realisme

```bash
uv run musica augment-realistic --input-dir audio/chords/clean --output-dir audio/chords/realistic --variants 2
```

Conserver la duree d'origine:

```bash
uv run musica augment-realistic --keep-duration --output-dir audio/chords/realistic_fixed
```

## Transposition

```bash
uv run musica augment-transpose --input-dir audio/chords/clean --output-dir audio/chords/transposed --semitones -5,7
```

Limiter a certaines qualites:

```bash
uv run musica augment-transpose --qualities min,dim --semitones -2,2 --max-files 200
```

## Manifest global

Compiler les manifests generes, derives et les WAV locaux enregistres:

```bash
uv run musica build-manifest --output-path audio/manifest.csv
```

Sources lues par defaut:

- `audio/chords/clean/manifest.csv`
- `audio/chords/noisy/manifest.csv`
- `audio/chords/realistic/manifest.csv`
- `audio/chords/transposed/manifest.csv`
- `audio/chords/recorded/*.wav`

Le manifest global contient les chemins audio, labels, splits stables, dataset source, duree et sample rate.
