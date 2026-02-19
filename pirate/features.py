"""Feature extraction pipeline: audio → modulation spectrum fingerprint."""

import hashlib
import io
import subprocess
from pathlib import Path

import numpy as np
import scipy.signal
import scipy.fft

from .config import (
    SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
    N_MELS,
    F_MIN,
    F_MAX,
    MOD_FFT_FRAMES,
    N_MOD_BINS,
    MOD_FREQ_MIN,
    MOD_FREQ_MAX,
)


# ---------------------------------------------------------------------------
# Step 1: Audio decode
# ---------------------------------------------------------------------------

def decode_audio(path: str | Path) -> np.ndarray:
    """
    Decode any audio file to mono float32 PCM at SAMPLE_RATE using ffmpeg.
    Returns a 1D float32 array.
    """
    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-i", str(path),
        "-ac", "1",                        # mono
        "-ar", str(SAMPLE_RATE),           # resample
        "-f", "f32le",                     # raw float32 little-endian PCM
        "-",                               # stdout
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path}: {result.stderr.decode()}")
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    if len(audio) == 0:
        raise RuntimeError(f"ffmpeg produced no audio for {path}")
    # Peak normalize
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak
    return audio


# ---------------------------------------------------------------------------
# Step 2: Mel spectrogram (pure numpy/scipy — no librosa required at runtime)
# ---------------------------------------------------------------------------

def _hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """Build a [n_mels × (n_fft//2+1)] mel filterbank matrix."""
    n_freqs = n_fft // 2 + 1
    freqs = np.linspace(0, sr / 2, n_freqs)

    mel_min = _hz_to_mel(fmin)
    mel_max = _hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = np.array([_mel_to_hz(m) for m in mel_points])

    filters = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        low, center, high = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        up = (freqs - low) / (center - low + 1e-10)
        down = (high - freqs) / (high - center + 1e-10)
        filters[i] = np.maximum(0, np.minimum(up, down))
    return filters


def mel_spectrogram(audio: np.ndarray) -> np.ndarray:
    """
    Compute log-power mel spectrogram.
    Returns [N_MELS × T] float32 array (T = number of frames).
    """
    # Build mel filterbank once (it's cheap)
    filterbank = _mel_filterbank(SAMPLE_RATE, N_FFT, N_MELS, F_MIN, F_MAX)

    n_freqs = N_FFT // 2 + 1
    window = scipy.signal.get_window("hann", N_FFT)

    # Frame the signal
    n_frames = (len(audio) - N_FFT) // HOP_LENGTH + 1
    if n_frames <= 0:
        # Audio too short — zero-pad to at least one frame
        audio = np.pad(audio, (0, N_FFT))
        n_frames = 1

    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(n_frames, N_FFT),
        strides=(audio.strides[0] * HOP_LENGTH, audio.strides[0]),
    ).copy()
    frames *= window

    # FFT → power spectrum
    spectra = np.fft.rfft(frames, n=N_FFT, axis=1)          # [T × n_freqs]
    power = (spectra.real ** 2 + spectra.imag ** 2).astype(np.float32)  # [T × n_freqs]

    # Apply mel filterbank: [n_mels × n_freqs] @ [n_freqs × T] → [n_mels × T]
    mel = filterbank @ power.T                                # [N_MELS × T]

    # Log scale
    log_mel = 10.0 * np.log10(mel + 1e-10)
    return log_mel.astype(np.float32)


# ---------------------------------------------------------------------------
# Step 3 & 4: Modulation spectrum with log-frequency axis
# ---------------------------------------------------------------------------

def modulation_spectrum(log_mel: np.ndarray) -> np.ndarray:
    """
    Compute the modulation spectrum from a log-mel spectrogram.

    For songs longer than MOD_FFT_FRAMES, computes multiple overlapping windows
    and averages their magnitude spectra. This gives a stable, position-invariant
    fingerprint that represents the full song rather than just the first 47 seconds.

    For clips shorter than MOD_FFT_FRAMES, zero-pads to maintain fixed output size.

    Returns [N_MELS × N_MOD_BINS] float32 array.
    """
    n_mels, n_frames = log_mel.shape

    frame_rate = SAMPLE_RATE / HOP_LENGTH                      # ~43.07 Hz
    n_mod_linear = MOD_FFT_FRAMES // 2 + 1
    mod_freqs_linear = np.linspace(0, frame_rate / 2, n_mod_linear)
    cutoff_idx = np.searchsorted(mod_freqs_linear, MOD_FREQ_MAX)
    mod_freqs_linear = mod_freqs_linear[:cutoff_idx]
    log_bins = np.geomspace(MOD_FREQ_MIN, MOD_FREQ_MAX, N_MOD_BINS)
    hann = np.hanning(MOD_FFT_FRAMES).astype(np.float32)

    def _window_mod(frames_2d: np.ndarray) -> np.ndarray:
        """Compute modulation magnitude for one MOD_FFT_FRAMES-wide window."""
        windowed = frames_2d * hann[np.newaxis, :]
        fft = np.fft.rfft(windowed, n=MOD_FFT_FRAMES, axis=1)
        mag = np.abs(fft).astype(np.float32)[:, :cutoff_idx]
        out = np.zeros((n_mels, N_MOD_BINS), dtype=np.float32)
        for i in range(n_mels):
            out[i] = np.interp(log_bins, mod_freqs_linear, mag[i])
        return out

    if n_frames < MOD_FFT_FRAMES:
        # Short clip: zero-pad to MOD_FFT_FRAMES
        padded = np.zeros((n_mels, MOD_FFT_FRAMES), dtype=np.float32)
        padded[:, :n_frames] = log_mel
        return _window_mod(padded)

    # Long song: average over 50%-overlapping windows to cover the full track
    hop = MOD_FFT_FRAMES // 2
    windows = []
    for start in range(0, n_frames - MOD_FFT_FRAMES + 1, hop):
        windows.append(_window_mod(log_mel[:, start:start + MOD_FFT_FRAMES]))
    # Always include a window anchored at the end of the track
    windows.append(_window_mod(log_mel[:, n_frames - MOD_FFT_FRAMES:]))

    return np.mean(windows, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Step 5: Fingerprint normalization
# ---------------------------------------------------------------------------

def fingerprint(mod_spec: np.ndarray) -> np.ndarray:
    """
    Normalize the modulation spectrum and flatten to a 1D fingerprint vector.
    L2-normalizes each mel band independently, then flattens.
    Returns float32 array of shape (N_MELS * N_MOD_BINS,).
    """
    # L2 normalize each mel band
    norms = np.linalg.norm(mod_spec, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    normed = mod_spec / norms
    return normed.flatten().astype(np.float32)


# ---------------------------------------------------------------------------
# BPM estimation
# ---------------------------------------------------------------------------

def estimate_bpm(mod_spec: np.ndarray) -> float | None:
    """
    Estimate BPM from the modulation spectrum.
    Uses the dominant modulation frequency in low mel bands (roughly < 500 Hz).
    Checks for half-time ambiguity: if 2× the detected peak also has strong energy,
    prefer the doubled value (handles dubstep/half-time genres).
    Returns BPM or None if no clear peak.
    """
    n_low = N_MELS // 4
    low_energy = mod_spec[:n_low].mean(axis=0)   # [N_MOD_BINS]

    log_bins = np.geomspace(MOD_FREQ_MIN, MOD_FREQ_MAX, N_MOD_BINS)

    # Only look in the BPM-plausible range: 0.5–4 Hz = 30–240 BPM
    valid = (log_bins >= 0.5) & (log_bins <= 4.0)
    if not valid.any():
        return None

    candidate_freqs = log_bins[valid]
    candidate_energy = low_energy[valid]

    peak_idx = candidate_energy.argmax()
    peak_energy = candidate_energy[peak_idx]
    peak_freq = candidate_freqs[peak_idx]

    # Require peak to be meaningfully above noise
    if peak_energy < candidate_energy.mean() * 1.5:
        return None

    # Half-time check: if 2× the peak frequency is within the valid range and
    # has at least 60% of the peak's energy, the detected peak is likely the
    # half-time sub-bass and the real tempo is double.
    double_freq = peak_freq * 2.0
    if double_freq <= 4.0:
        double_energy = float(np.interp(double_freq, log_bins, low_energy))
        if double_energy >= peak_energy * 0.6:
            peak_freq = double_freq

    return round(peak_freq * 60.0, 1)


# ---------------------------------------------------------------------------
# Convenience: full pipeline for one file
# ---------------------------------------------------------------------------

def extract(path: str | Path) -> tuple[np.ndarray, float | None]:
    """
    Full pipeline: file → (fingerprint_vector, bpm_estimate).
    bpm_estimate may be None if the track doesn't have a clear beat.
    """
    audio = decode_audio(path)
    mel = mel_spectrogram(audio)
    mod = modulation_spectrum(mel)
    fp = fingerprint(mod)
    bpm = estimate_bpm(mod)
    return fp, bpm


def file_hash(path: str | Path) -> str:
    """SHA256 of the first 1 MB of a file — fast enough for change detection."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(1024 * 1024))
    return h.hexdigest()
