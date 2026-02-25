# P.I.R.A.T.E.

**Perceptual Invariant Rhythmic Audio Taxonomy Engine**

A local-first music fingerprinting and similarity engine. Shazam meets iTunes Genius, running entirely on your own library. No cloud, no subscriptions, no streaming service deciding what you like.

## What it does

- **Identify** — "what song is this?" against your own library, invariant to BPM/pitch/length changes
- **Similar** — "play me something like this" via energy/vibe clustering
- **Playlist** — automatic smart playlists by perceptual features, not metadata tags
- **UI** — browser-based library explorer with built-in player *(coming in v0.2)*

## How it works

Standard spectrograms break when songs have different lengths or BPM. PIRATE uses a **modulation spectrum**: FFT along the time axis of each mel frequency band. This converts "energy at frequency F at time T" into "energy at frequency F that oscillates at rate R". The result is a fixed-size fingerprint that's length-invariant and BPM-normalizable.

```
Audio → Mel Spectrogram → FFT per band along time axis → Modulation Spectrum
                                                          (128 mel bands × 64 log-spaced modulation bins)
```

---

## Installation

### Option A — Desktop app (recommended for most users)

> **Coming soon.** A native installer (Windows `.exe`, macOS `.dmg`, Linux `.AppImage`) is planned for v0.2. It will bundle everything including ffmpeg — download, double-click, done. Track progress in [issue #3](https://github.com/dudebot/pirate/issues/3).

### Option B — pipx (recommended for CLI/power users)

[pipx](https://pipx.pypa.io) installs Python CLI tools in isolated environments. If you don't have it:

```bash
# macOS
brew install pipx && pipx ensurepath

# Windows (in PowerShell)
scoop install pipx

# Linux
pip install --user pipx && pipx ensurepath
```

Then install PIRATE:

```bash
pipx install "pirate-audio[all]"
```

> **Note:** `pirate-audio` on PyPI is coming in v0.2 ([issue #4](https://github.com/dudebot/pirate/issues/4)). Until then, install from source (Option C).

### Option C — From source (developers)

Requires Python 3.10+ and [ffmpeg](https://ffmpeg.org/download.html) in your PATH.

```bash
git clone https://github.com/dudebot/pirate
cd pirate
pipx install --editable ".[all]"
```

Or in a plain virtualenv:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[all]"
```

### ffmpeg

PIRATE uses ffmpeg to decode audio. It must be installed separately and available in your PATH.

| Platform | Install |
|----------|---------|
| macOS | `brew install ffmpeg` |
| Windows | `scoop install ffmpeg` or [download from ffmpeg.org](https://ffmpeg.org/download.html) |
| Ubuntu/Debian | `sudo apt install ffmpeg` |

---

## Getting started

### 1. Scan your library

```bash
pirate scan ~/Music
```

First run takes a while — roughly 2–5 seconds per track. Progress is shown. Re-runs with `--update` only process new or changed files:

```bash
pirate scan ~/Music --update
```

Use `-j` to control parallelism (defaults to CPU count):

```bash
pirate scan ~/Music --update -j 8
```

### 2. Find similar songs

```bash
# By artist/title search
pirate similar "Daft Punk"

# By file path
pirate similar "/path/to/song.mp3"

# More results
pirate similar "Boards of Canada" -k 20

# Include DJ mixes and long tracks (excluded by default)
pirate similar "Aphex Twin" --mixes
```

### 3. Identify a recording

Works like Shazam against your own library. Feed it a clip, a recording, or a full file:

```bash
pirate identify mystery_clip.mp3
pirate identify recording.wav -k 10
```

### 4. Generate a playlist

```bash
# 60-minute playlist seeded from one track
pirate playlist --like "Knife Party" --length 60m --output mix.m3u

# Multiple seeds, steer away from something, custom length
pirate playlist --like "Deadmau5" --like "Daft Punk" --unlike "vocals" --length 1h30m -o mix.m3u
```

The output is a standard M3U file compatible with VLC, foobar2000, mpv, and most players.

### 5. Cluster your library

```bash
# K-means into N clusters (fast, deterministic)
pirate cluster --count 20

# Auto-detect cluster count with HDBSCAN (requires: pip install hdbscan)
pirate cluster --auto

# Export cluster assignments as JSON
pirate cluster --count 20 --export clusters.json
```

### 6. Other commands

```bash
# Show BPM estimate for a track
pirate bpm "Knife Party"

# Library stats
pirate stats
```

---

## MCP server (Claude integration)

> **Coming soon** — [issue #5](https://github.com/dudebot/pirate/issues/5). Will let you query your library from Claude:
> *"make me a 2-hour late night playlist similar to Boards of Canada, nothing harsh"*

---

## Storage

Fingerprints live in `~/.pirate/library.db` (SQLite). A 10,000 song library takes roughly 10 MB.

Set `PIRATE_DATA_DIR` to use a different location:

```bash
export PIRATE_DATA_DIR=/external/drive/.pirate
pirate scan /external/drive/Music
```

---

## Requirements

- Python 3.10+
- ffmpeg in PATH
- numpy, scipy (installed automatically)
- scikit-learn (for `pirate cluster`)
- mutagen (for reading tags — strongly recommended)
