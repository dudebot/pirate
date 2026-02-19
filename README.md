# P.I.R.A.T.E.

**Perceptual Invariant Rhythmic Audio Taxonomy Engine**

A local-first music fingerprinting and similarity engine. Shazam meets iTunes Genius, running entirely on your own library. No cloud, no subscriptions, no streaming service deciding what you like.

## What it does

- **Identify** — "what song is this?" against your own library, invariant to BPM/pitch/length changes
- **Similar** — "play me something like this" via energy/vibe clustering
- **Playlist** — automatic smart playlists by perceptual features, not metadata tags

## How it works

Standard spectrograms break when songs have different lengths or BPM. PIRATE uses a **modulation spectrum**: FFT along the time axis of each mel frequency band. This converts "energy at frequency F at time T" into "energy at frequency F that oscillates at rate R". The result is a fixed-size fingerprint that's length-invariant and BPM-normalizable.

```
Audio → Mel Spectrogram → FFT per band along time axis → Modulation Spectrum
                                                          (128 mel bands × 64 log-spaced modulation bins)
```

## Quick start

```bash
pip install -e .

# Scan your library (first run — takes a while)
pirate scan ~/Music

# Find similar songs
pirate similar "Artist - Track Name"

# Identify a clip
pirate identify recording.wav

# Auto-cluster your library
pirate cluster --auto

# Generate a playlist
pirate playlist --like "Track A" --like "Track B" --length 60m
```

## Requirements

- Python 3.10+
- ffmpeg (system dependency, must be in PATH)

## Storage

Fingerprints live in `~/.pirate/library.db` (SQLite). A 10,000 song library takes ~10 MB with PCA compression, ~320 MB full.
