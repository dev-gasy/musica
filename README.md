# Musica

Musica reconnaît des accords dans de courts fichiers WAV.

Le projet sait générer des accords propres, produire des versions bruitées ou
plus réalistes, extraire des features Chroma-CQT, entraîner un CNN, puis tester
les prédictions sur des exemples audio.

## À lire

- [Setup](SETUP.md)
- [Troubleshooting](TROUBLESHOOTING.md)
- [Guide pédagogique](GUIDE.md)
- [Notebook](musica.ipynb)

## Setup rapide

Depuis la racine du dépôt :

```bash
uv sync --extra dev
uv run musica setup-env
```

`setup-env` prépare ce qui manque et laisse tranquille ce qui existe déjà. À la
fin, le dataset de travail doit contenir ces dossiers :

```text
audio/chords/clean
audio/chords/noisy
audio/chords/realistic
audio/chords/recorded
```

Les fichiers de `assets/recorded` sont copiés dans `audio/chords/recorded`
pendant le setup.

Pour voir ce qui serait fait sans écrire de fichiers :

```bash
uv run musica setup-env --plan-only
```

FluidSynth est optionnel. Quand il est installé avec une SoundFont, Musica
l’utilise pour le rendu audio. Sinon, le rendu passe par PrettyMIDI.

## Structure

- `src/musica/` : code Python.
- `musica.toml` : chemins, paramètres audio, splits, modèle et augmentations.
- `assets/noises/` : bruits WAV utilisés pour les augmentations.
- `assets/recorded/` : enregistrements réels source.
- `audio/chords/` : dataset local préparé par le setup.
- `musica.ipynb` : démonstration.

## Commandes courantes

```bash
uv run musica setup-env
uv run python main.py
uv run musica build-manifest
uv run pytest
```
