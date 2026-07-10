# Musica

**Musica** est un prototype de reconnaissance automatique d’accords à partir de
courts extraits audio WAV.

L’objectif est de montrer comment un pipeline audio peut transformer un fichier
sonore en prédiction d’accord. Ce type de brique peut servir, à terme, à des
usages comme l’aide à la transcription, l’indexation simple d’extraits musicaux
ou l’analyse automatique de petits catalogues audio.

## À lire

- [Setup](docs/SETUP.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Guide pédagogique](docs/GUIDE.md)
- [Notebook](musica.ipynb)

## Setup rapide

Depuis la racine du dépôt :

```bash
uv sync --extra dev
uv run musica setup-env
```

Musica supporte Python 3.12 et 3.13. TensorFlow 2.21 ne publie pas encore de
wheel pour Python 3.14, donc le dépôt inclut `.python-version` pour guider `uv`
vers Python 3.13.

`setup-env` prépare ce qui manque. À la
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

Si le téléchargement automatique de la SoundFont échoue avec `HTTP Error 403:
Forbidden`, téléchargez manuellement `FluidR3_GM.sf2` depuis
[Musical Artifacts](https://musical-artifacts.com/artifacts/738/FluidR3_GM.sf2)
ou une autre source fiable, puis placez le fichier ici :

```text
assets/soundfonts/FluidR3_GM.sf2
```

Sans ce fichier, `setup-env` continue quand même avec le renderer `auto`, qui
utilise PrettyMIDI.

## Structure

- `src/musica/` : code Python.
- `musica.toml` : chemins, paramètres audio, splits, modèle et augmentations.
- `assets/noises/` : bruits WAV utilisés pour les augmentations.
- `assets/recorded/` : enregistrements réels source.
- `audio/chords/` : dataset local préparé par le setup.
- `docs/` : guide pédagogique, setup détaillé, troubleshooting, images et exports.
- `musica.ipynb` : démonstration.

## Commandes courantes

```bash
uv run musica setup-env
uv run python main.py
uv run musica build-manifest
uv run pytest
```
