"""PIRATE configuration and paths."""

import os
from pathlib import Path

# Data directory: ~/.pirate/
DATA_DIR = Path(os.environ.get("PIRATE_DATA_DIR", Path.home() / ".pirate"))
DB_PATH = DATA_DIR / "library.db"

# Feature extraction parameters
SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
F_MIN = 20.0
F_MAX = 11000.0

# Modulation spectrum parameters
MOD_FFT_FRAMES = 2048        # frames used for modulation FFT (zero-pad shorter songs)
N_MOD_BINS = 64              # log-spaced modulation frequency bins
MOD_FREQ_MIN = 0.1           # Hz
MOD_FREQ_MAX = 20.0          # Hz

# Fingerprint
FINGERPRINT_DIM = N_MELS * N_MOD_BINS   # 8192 full, or PCA_DIM post-PCA
PCA_DIM = 256

# Pipeline version — increment when the extraction logic changes
PIPELINE_VERSION = 2

# Audio formats supported via ffmpeg
AUDIO_EXTENSIONS = {".mp3", ".flac", ".wav", ".m4a", ".ogg", ".opus", ".aac", ".wma"}
