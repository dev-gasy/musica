# Setup

Objectif : partir d’une machine neuve et pouvoir lancer Musica sans préparer les
dossiers à la main. Le README garde la version courte.

## Commande par défaut

Depuis la racine du dépôt :

```bash
uv sync --extra dev
uv run musica setup-env
```

La commande peut être relancée autant de fois que nécessaire. Si une étape est
déjà prête, elle est marquée `SKIP`.

Après un setup normal, ces dossiers doivent exister et contenir des WAV :

```text
audio/chords/clean
audio/chords/noisy
audio/chords/realistic
audio/chords/recorded
```

Les fichiers de `assets/recorded` restent versionnés à cet endroit. Pendant le
setup, ils sont copiés dans `audio/chords/recorded`, qui est le dossier utilisé
par le pipeline.

## Prérequis

- Python `>=3.12`, comme indiqué dans `pyproject.toml`.
- `uv` pour installer les dépendances et lancer les commandes.
- Une connexion réseau au premier setup, si les assets externes ne sont pas déjà
  présents.
- FluidSynth si vous voulez le rendu SF2. Le projet fonctionne aussi sans lui.

Installer `uv` :

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## FluidSynth

FluidSynth n’est pas obligatoire. S’il manque, Musica peut générer les WAV avec
PrettyMIDI.

Installation selon le système :

```bash
# macOS
brew install fluid-synth

# Debian / Ubuntu
sudo apt install fluidsynth

# Fedora
sudo dnf install fluidsynth

# Arch
sudo pacman -S fluidsynth

# Windows avec Chocolatey
choco install fluidsynth
```

Vérifier :

```bash
fluidsynth --version
```

## Ce que fait `setup-env`

```bash
uv run musica setup-env
```

Le setup :

1. affiche les commandes utiles pour reproduire l’environnement ;
2. cherche FluidSynth dans le `PATH` ;
3. vérifie la SoundFont, ou la télécharge si elle manque ;
4. vérifie les bruits WAV, ou les télécharge si besoin ;
5. génère `audio/chords/clean` si le dossier est vide ;
6. génère `audio/chords/noisy` si le dossier est vide ;
7. génère `audio/chords/realistic` si le dossier est vide ;
8. copie `assets/recorded/*.wav` vers `audio/chords/recorded` ;
9. saute les étapes qui sont déjà prêtes.

Pour contrôler le plan sans modifier les fichiers :

```bash
uv run musica setup-env --plan-only
```

## Options

```bash
uv run musica setup-env --skip-assets
uv run musica setup-env --skip-audio
uv run musica setup-env --overwrite-assets
uv run musica setup-env --force-audio
uv run musica setup-env --renderer pretty-midi
uv run musica setup-env --max-audio-files 12
uv run musica setup-env --platform linux
```

- `--skip-assets` : ne prépare pas la SoundFont ni les bruits.
- `--skip-audio` : ne génère pas les WAV.
- `--overwrite-assets` : retélécharge les assets.
- `--force-audio` : régénère les WAV même si les dossiers existent déjà.
- `--renderer` : force `auto`, `pretty-midi` ou `fluidsynth`.
- `--max-audio-files` : limite le nombre de WAV traités pour un test rapide.
- `--platform` : affiche les instructions FluidSynth pour `macos`, `linux` ou
  `windows`.

## Assets

SoundFont attendue :

```text
assets/soundfonts/FluidR3_GM.sf2
```

Commande dédiée :

```bash
uv run musica download-soundfont
```

Bruits attendus :

```text
assets/noises/internet
```

Commande dédiée :

```bash
uv run musica download-noises
```

Enregistrements réels :

```text
assets/recorded          # source versionnée
audio/chords/recorded    # copie utilisée par le dataset
```

`setup-env` recopie seulement les fichiers manquants ou modifiés.

## Dataset local

Le dataset de travail est sous `audio/chords` :

```text
audio/chords/clean      # accords générés propres
audio/chords/noisy      # accords mixés avec des bruits
audio/chords/realistic  # variantes avec effets réalistes
audio/chords/recorded   # enregistrements copiés depuis assets/recorded
```

Les chemins viennent de `musica.toml` :

- `[dataset].dataset_dir`
- `[dataset].recorded_audio_dir`
- `[audio].clean_output_dir`
- `[noise].noisy_output_dir`
- `[realism].realistic_output_dir`

## Rejouer une étape

Ces commandes aident à isoler un problème :

```bash
uv run musica download-assets
uv run musica generate-wav
uv run musica augment-noise
uv run musica augment-realistic
uv run musica build-manifest
```

Forcer PrettyMIDI :

```bash
uv run musica generate-wav --renderer pretty-midi
```

## Après le setup

```bash
uv run musica setup-env --plan-only
uv run pytest
uv run python main.py
```

Le notebook part du principe que le setup a déjà été lancé. S’il manque des WAV
ou un des dossiers `clean`, `noisy`, `realistic`, `recorded`, il affiche les
commandes à lancer au lieu de fabriquer le dataset lui-même.

En cas de blocage, voir [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
