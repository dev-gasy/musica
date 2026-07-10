# Plan du Projet Musica

Derniere mise a jour : 2026-07-08

## Objectif actuel

Musica est maintenant centre sur la construction de datasets audio annotes pour accords. Le package actif garde la
generation MIDI/WAV, les manifests audio, les augmentations et l'assemblage des sources locales.

Le code de modelisation et les rapports d'experimentation sont archives dans `archive/modeling/`. Ils restent
consultables, mais ne font plus partie du package actif.

## Surface active

| Zone                  | Etat    | Notes                                                                    |
|:----------------------|:--------|:-------------------------------------------------------------------------|
| Generation MIDI/WAV   | Fait    | Accords humanises, durees multiples, repetitions, PrettyMIDI/FluidSynth. |
| Manifest WAV genere   | Fait    | `generate-wav` ecrit un manifest d'annotations optionnel.                |
| Augmentation bruit    | Fait    | Telechargement de bruits et mixage SNR controle.                         |
| Augmentation realiste | Fait    | Gain, filtres, compression, saturation, reverb, stereo, sample rate.     |
| Transposition         | Fait    | Pitch-shift avec relabelisation de la fondamentale.                      |
| Manifest audio global | Fait    | Sources `clean`, `noisy`, `realistic`, `transposed`, `recorded`.         |
| Modele/R&D            | Archive | Ancien pipeline conserve dans `archive/modeling/`.                       |

## Commandes actives

```bash
uv run musica generate-midi
uv run musica download-assets
uv run musica generate-wav
uv run musica augment-noise
uv run musica augment-realistic
uv run musica augment-transpose
uv run musica build-manifest
```

## Verification de base

```bash
uv run pytest
uv run musica --help
uv run musica download-assets
uv run musica generate-wav --duration 0.5 --max-files 6 --renderer pretty-midi
uv run musica build-manifest
```

## TODO courts

- [ ] Verifier `uv run pytest` apres chaque modification du pipeline audio.
- [ ] Garder `docs/scripts.md` synchronise avec les commandes CLI actives.
- [ ] Ajouter des fixtures audio plus representatives si les augmentations evoluent.
- [ ] Ne pas reintroduire de dependance modele dans `src/musica` sans decision explicite.
