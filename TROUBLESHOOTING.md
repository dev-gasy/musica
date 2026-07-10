# Troubleshooting

Commencer par ces trois commandes depuis la racine du dépôt :

```bash
uv run musica setup-env --plan-only
uv run musica setup-env
uv run pytest
```

`--plan-only` ne modifie rien. Il montre ce que le setup ferait.

## `uv` introuvable

Erreur :

```text
uv: command not found
```

Installer `uv`, puis relancer :

```bash
uv sync --extra dev
uv run musica setup-env
```

Installation officielle : https://docs.astral.sh/uv/getting-started/installation/

## Imports Python cassés

Erreurs possibles :

```text
ModuleNotFoundError
ImportError
No module named musica
```

Recréer l’environnement :

```bash
uv sync --extra dev
uv run pytest
```

Toujours lancer les commandes depuis la racine du dépôt avec `uv run ...`.

## FluidSynth absent

Message dans `setup-env --plan-only` :

```text
TODO FluidSynth: not found on PATH
```

Ce n’est pas bloquant. Sans FluidSynth, Musica peut utiliser PrettyMIDI.

Pour installer FluidSynth :

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

Puis vérifier :

```bash
fluidsynth --version
uv run musica setup-env --plan-only
```

## SoundFont absente

Erreurs possibles :

```text
SoundFont not found
HTTPError
URLError
```

Relancer le setup complet :

```bash
uv run musica setup-env
```

Ou seulement les assets :

```bash
uv run musica download-assets
```

Fichier attendu :

```text
assets/soundfonts/FluidR3_GM.sf2
```

## Aucun WAV sous `audio/chords`

Erreur :

```text
FileNotFoundError: No WAV files found under audio/chords
```

Préparer le dataset :

```bash
uv run musica setup-env
```

Dossiers attendus :

```text
audio/chords/clean
audio/chords/noisy
audio/chords/realistic
audio/chords/recorded
```

Pour repartir sur une génération forcée :

```bash
uv run musica setup-env --force-audio
```

## `recorded` manque dans `audio/chords`

Le dossier source est :

```text
assets/recorded
```

Le dossier utilisé par le dataset est :

```text
audio/chords/recorded
```

Copier via le setup :

```bash
uv run musica setup-env
```

Vérifier :

```bash
find assets/recorded -name "*.wav"
find audio/chords/recorded -name "*.wav"
```

## Bruits WAV absents

Erreur possible :

```text
No noise WAV files found
```

Télécharger les assets, puis relancer le setup :

```bash
uv run musica download-assets
uv run musica setup-env
```

Dossier attendu :

```text
assets/noises/internet
```

## Notebook bloqué au check d’environnement

Le notebook ne prépare pas le dataset. Il vérifie seulement que le setup a déjà
été fait.

À lancer avant d’ouvrir le notebook :

```bash
uv sync --extra dev
uv run musica setup-env
```

Ensuite, redémarrer le kernel et relancer le notebook depuis le début.

## Warning `librosa n_fft`

Message possible pendant les tests :

```text
UserWarning: n_fft=2048 is too large for input signal
```

Ce warning arrive avec certains extraits très courts. Il n’est pas bloquant si
les tests finissent avec `passed`.

## Commandes utiles

```bash
uv run musica setup-env --plan-only
uv run musica setup-env --skip-audio
uv run musica setup-env --skip-assets
uv run python main.py --audio-only
uv run pytest
```

Le setup complet est dans [docs/setup-env.md](SETUP.md).
