# Methodologie de Construction du Dataset Audio d'Accords

Musica construit des fichiers WAV annotes pour accords. Le but actif du projet est de produire des donnees audio
propres, variees et traçables, pas d'entrainer un modele dans ce paquet.

## Sources possibles

### Synthese audio

La synthese est la source la plus reproductible. Musica genere d'abord des accords MIDI, puis les rend en WAV avec
PrettyMIDI ou FluidSynth.

Variations recommandees:

- fondamentales: `C`, `C#`, `D`, `D#`, `E`, `F`, `F#`, `G`, `G#`, `A`, `A#`, `B`;
- qualites: `maj`, `min`, `dim`;
- instruments MIDI: piano, guitare, pad synthetique;
- octaves, velocites, durees et repetitions;
- humanisation activee pour varier voicings, inversions, doublages d'octave et attaques.

Commande de base:

```bash
uv run musica generate-wav --output-dir audio/chords/clean
```

### Enregistrements locaux

Les fichiers enregistres manuellement peuvent etre places dans `audio/chords/recorded/`. Pour etre inclus
automatiquement par `build-manifest`, nommer les WAV avec au moins la fondamentale et la qualite, par exemple:

```text
C_maj_take1.wav
Fsharp_min_voicing2.wav
A_dim_room.wav
```

Conseils d'enregistrement:

- garder un environnement silencieux;
- couper les silences excessifs;
- varier voicings, dynamiques, octaves et repetitions;
- conserver un nom de fichier stable et lisible.

## Augmentations

Musica garde trois augmentations audio actives.

### Bruit

```bash
uv run musica download-noises --output-dir assets/noises/internet
uv run musica augment-noise --input-dir audio/chords/clean --noise-dir assets/noises/internet --output-dir audio/chords/noisy
```

### Realisme

```bash
uv run musica augment-realistic --input-dir audio/chords/clean --output-dir audio/chords/realistic --variants 2
```

### Transposition

```bash
uv run musica augment-transpose --input-dir audio/chords/clean --output-dir audio/chords/transposed --semitones -5,7
```

## Manifest global

Le manifest global compile les sources audio locales:

- `audio/chords/clean/manifest.csv`
- `audio/chords/noisy/manifest.csv`
- `audio/chords/realistic/manifest.csv`
- `audio/chords/transposed/manifest.csv`
- `audio/chords/recorded/*.wav`

Commande:

```bash
uv run musica build-manifest --output-path audio/manifest.csv
```

Le CSV final contient le chemin audio, le label, le dataset source, un split stable, la duree et le sample rate.
