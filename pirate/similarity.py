"""Similarity engine: cosine distance, KNN, BPM-invariant matching."""

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_similarity_matrix(query: np.ndarray, library: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a query vector and every row in a matrix.
    query:   [D]
    library: [N × D]
    Returns: [N] similarity scores
    """
    norms = np.linalg.norm(library, axis=1)
    norms = np.where(norms < 1e-10, 1.0, norms)
    query_norm = np.linalg.norm(query)
    if query_norm < 1e-10:
        return np.zeros(len(library))
    return (library @ query) / (norms * query_norm)


def top_k(scores: np.ndarray, song_ids: list[int], k: int, exclude_ids: set[int] | None = None) -> list[tuple[int, float]]:
    """
    Return top-k (song_id, score) pairs, highest score first.
    Optionally exclude specific song IDs (e.g. the query itself).
    """
    exclude = exclude_ids or set()
    indexed = [(song_ids[i], float(scores[i])) for i in range(len(scores)) if song_ids[i] not in exclude]
    indexed.sort(key=lambda x: x[1], reverse=True)
    return indexed[:k]


def find_similar(
    query_vec: np.ndarray,
    library_vecs: np.ndarray,
    library_ids: list[int],
    k: int = 10,
    exclude_ids: set[int] | None = None,
) -> list[tuple[int, float]]:
    """
    Find the k most similar songs to a query fingerprint.
    Returns list of (song_id, similarity_score) sorted descending.
    """
    scores = cosine_similarity_matrix(query_vec, library_vecs)
    return top_k(scores, library_ids, k, exclude_ids)


def blend_seeds(vecs: list[np.ndarray]) -> np.ndarray:
    """
    Blend multiple seed fingerprints into a single query vector by averaging.
    Each seed is L2-normalized before averaging so no seed dominates.
    """
    normalized = []
    for v in vecs:
        n = np.linalg.norm(v)
        normalized.append(v / n if n > 1e-10 else v)
    blended = np.mean(normalized, axis=0)
    return blended.astype(np.float32)


def bpm_invariant_similarity(
    query_mod: np.ndarray,
    candidate_mod: np.ndarray,
) -> tuple[float, float]:
    """
    BPM-invariant similarity via cross-correlation along the log-modulation axis.

    Both inputs are [N_MELS × N_MOD_BINS] modulation spectra (before flattening).
    Returns (similarity_score, bpm_ratio_estimate).

    bpm_ratio > 1.0 means the query is faster than the candidate.
    """
    from .config import MOD_FREQ_MIN, MOD_FREQ_MAX, N_MOD_BINS

    # Average across mel bands to get 1D modulation profiles
    q_profile = query_mod.mean(axis=0)
    c_profile = candidate_mod.mean(axis=0)

    # Normalize
    q_profile = q_profile / (np.linalg.norm(q_profile) + 1e-10)
    c_profile = c_profile / (np.linalg.norm(c_profile) + 1e-10)

    # Cross-correlate
    corr = np.correlate(q_profile, c_profile, mode="full")
    peak_offset = int(np.argmax(corr)) - (N_MOD_BINS - 1)

    # Convert log-bin offset to BPM ratio
    # Each bin step in log space = log(MOD_FREQ_MAX/MOD_FREQ_MIN) / N_MOD_BINS
    log_step = np.log(MOD_FREQ_MAX / MOD_FREQ_MIN) / N_MOD_BINS
    bpm_ratio = np.exp(peak_offset * log_step)

    similarity = float(corr.max())
    return similarity, float(bpm_ratio)


def playlist_walk(
    seed_ids: set[int],
    library_vecs: np.ndarray,
    library_ids: list[int],
    target_duration_s: float,
    song_durations: dict[int, float],
    dislike_ids: set[int] | None = None,
    step_k: int = 5,
) -> list[int]:
    """
    Greedy nearest-neighbor playlist walk from a seed set.
    At each step, pick the most similar unplayed song to the current average.
    Stops when target duration is reached.
    Returns ordered list of song_ids.
    """
    exclude = set(seed_ids) | (dislike_ids or set())
    playlist: list[int] = []
    total_s = 0.0

    # Build seed vector
    seed_indices = [library_ids.index(sid) for sid in seed_ids if sid in library_ids]
    if not seed_indices:
        return []
    current_vec = blend_seeds([library_vecs[i] for i in seed_indices])

    while total_s < target_duration_s:
        candidates = find_similar(current_vec, library_vecs, library_ids, k=step_k, exclude_ids=exclude)
        if not candidates:
            break
        next_id, _ = candidates[0]
        playlist.append(next_id)
        exclude.add(next_id)
        total_s += song_durations.get(next_id, 0.0)
        # Slide the current vector toward the newly added song
        next_idx = library_ids.index(next_id)
        current_vec = blend_seeds([current_vec, library_vecs[next_idx]])

    return playlist
