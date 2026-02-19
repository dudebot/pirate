"""PIRATE CLI."""

import multiprocessing
import os
import sys
from pathlib import Path

import click
import numpy as np
from tqdm import tqdm

from .config import AUDIO_EXTENSIONS, DB_PATH
from .features import extract, file_hash
from .similarity import find_similar, blend_seeds, playlist_walk
from .store import Store


# ---------------------------------------------------------------------------
# Parallel scan worker (runs in subprocess — no DB access here)
# ---------------------------------------------------------------------------

def _scan_worker(args: tuple) -> dict:
    """
    Extract fingerprint + metadata for one file.
    Runs in a worker process. Returns a plain dict so it's picklable.
    """
    path_str, has_mutagen = args
    path = Path(path_str)
    result = {
        "path": path_str,
        "fhash": None,
        "title": None, "artist": None, "album": None, "duration_s": None,
        "fp_vec": None, "bpm": None,
        "error": None,
    }
    try:
        result["fhash"] = file_hash(path)

        if has_mutagen:
            try:
                import mutagen
                meta = mutagen.File(path, easy=True)
                if meta:
                    result["title"]      = (meta.get("title")  or [None])[0]
                    result["artist"]     = (meta.get("artist") or [None])[0]
                    result["album"]      = (meta.get("album")  or [None])[0]
                    result["duration_s"] = getattr(meta.info, "length", None)
            except Exception:
                pass

        fp_vec, bpm = extract(path)
        result["fp_vec"] = fp_vec
        result["bpm"]    = bpm
    except Exception as e:
        result["error"] = str(e)
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_library(store: Store) -> tuple[np.ndarray, list[int]]:
    """Load all fingerprints into a matrix. Returns (matrix, song_ids)."""
    pairs = list(store.all_fingerprints())
    if not pairs:
        return np.empty((0,), dtype=np.float32), []
    ids, vecs = zip(*pairs)
    return np.stack(vecs), list(ids)


def _resolve_song(store: Store, query: str):
    """
    Resolve a query string to a song row.
    Accepts a file path or a search string against title/artist/album.
    Exits with an error if nothing found or ambiguous.
    """
    path = Path(query)
    if path.exists():
        row = store.get_song_by_path(str(path.resolve()))
        if row is None:
            click.echo(f"File found but not in library: {query}. Run `pirate scan` first.", err=True)
            sys.exit(1)
        return row

    results = store.search_songs(query)
    if not results:
        click.echo(f"No song found matching: {query}", err=True)
        sys.exit(1)
    if len(results) > 1:
        click.echo(f"Ambiguous query '{query}'. Matches:", err=True)
        for r in results[:10]:
            click.echo(f"  {r['artist']} — {r['title']}  ({r['path']})", err=True)
        sys.exit(1)
    return results[0]


def _format_song(row) -> str:
    parts = []
    if row["artist"]:
        parts.append(row["artist"])
    if row["title"]:
        parts.append(row["title"])
    if not parts:
        parts.append(Path(row["path"]).name)
    label = " — ".join(parts)
    if row["duration_s"]:
        m, s = divmod(int(row["duration_s"]), 60)
        label += f"  [{m}:{s:02d}]"
    return label


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.group()
def cli():
    """PIRATE — Perceptual Invariant Rhythmic Audio Taxonomy Engine.

    Local-first music fingerprinting and similarity. No cloud required.
    """


@cli.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--update", is_flag=True, help="Only process new or changed files.")
@click.option("--workers", "-j", default=None, type=int,
              help="Parallel workers (default: CPU count).")
@click.option("--db", type=click.Path(path_type=Path), default=None,
              help="Database path (default: ~/.pirate/library.db).")
def scan(directory: Path, update: bool, workers: int | None, db: Path | None):
    """Scan DIRECTORY and build fingerprints for all audio files."""
    db_path = db or DB_PATH
    n_workers = workers or os.cpu_count() or 4

    audio_files = [
        p for p in directory.rglob("*")
        if p.suffix.lower() in AUDIO_EXTENSIONS
    ]
    click.echo(f"Found {len(audio_files)} audio files in {directory}")

    try:
        import mutagen
        has_mutagen = True
    except ImportError:
        has_mutagen = False

    # If --update, pre-filter files that are already current so workers
    # don't waste time on them. Hash check is fast enough to do on the main thread.
    if update:
        with Store(db_path) as store:
            filtered = []
            skipped = 0
            for path in audio_files:
                path_str = str(path.resolve())
                existing = store.get_song_by_path(path_str)
                if existing:
                    fhash = file_hash(path)
                    if existing["file_hash"] == fhash and not store.needs_fingerprint(existing["id"], fhash):
                        skipped += 1
                        continue
                filtered.append(path)
            click.echo(f"Skipping {skipped} unchanged files, processing {len(filtered)}.")
            audio_files = filtered

    if not audio_files:
        click.echo("Nothing to do.")
        return

    click.echo(f"Processing with {n_workers} workers...")

    skipped = 0
    errors = 0
    work_args = [(str(p.resolve()), has_mutagen) for p in audio_files]

    # Workers handle all CPU-bound work; main thread owns the DB connection.
    with Store(db_path) as store:
        with multiprocessing.Pool(processes=n_workers) as pool:
            with tqdm(total=len(work_args), unit="track") as pbar:
                for result in pool.imap_unordered(_scan_worker, work_args):
                    pbar.update(1)
                    if result["error"]:
                        tqdm.write(f"ERROR {Path(result['path']).name}: {result['error']}")
                        errors += 1
                        continue
                    song_id = store.upsert_song(
                        path=result["path"],
                        file_hash=result["fhash"],
                        title=result["title"],
                        artist=result["artist"],
                        album=result["album"],
                        duration_s=result["duration_s"],
                        bpm_est=result["bpm"],
                    )
                    store.upsert_fingerprint(song_id, result["fp_vec"])

    click.echo(f"Done. Errors: {errors}.")


@cli.command()
@click.argument("target")
@click.option("-k", "--count", default=10, show_default=True, help="Number of results.")
@click.option("--db", type=click.Path(path_type=Path), default=None)
def similar(target: str, count: int, db: Path | None):
    """Find songs similar to TARGET (file path or search string)."""
    db_path = db or DB_PATH
    with Store(db_path) as store:
        row = _resolve_song(store, target)
        fp = store.get_fingerprint(row["id"])
        if fp is None:
            click.echo("No fingerprint for this song. Run `pirate scan` first.", err=True)
            sys.exit(1)

        library, ids = _load_library(store)
        if len(ids) == 0:
            click.echo("Library is empty. Run `pirate scan` first.", err=True)
            sys.exit(1)

        results = find_similar(fp, library, ids, k=count, exclude_ids={row["id"]})

        click.echo(f"\nSimilar to: {_format_song(row)}\n")
        for rank, (song_id, score) in enumerate(results, 1):
            r = store.get_song_by_id(song_id)
            click.echo(f"  {rank:2d}. [{score:.3f}]  {_format_song(r)}")


@cli.command()
@click.argument("clip", type=click.Path(exists=True, path_type=Path))
@click.option("-k", "--count", default=5, show_default=True)
@click.option("--db", type=click.Path(path_type=Path), default=None)
def identify(clip: Path, count: int, db: Path | None):
    """Identify CLIP against the library."""
    db_path = db or DB_PATH
    click.echo(f"Extracting fingerprint from {clip.name}...")
    try:
        fp, bpm = extract(clip)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if bpm:
        click.echo(f"Estimated BPM: {bpm}")

    with Store(db_path) as store:
        library, ids = _load_library(store)
        if len(ids) == 0:
            click.echo("Library is empty. Run `pirate scan` first.", err=True)
            sys.exit(1)

        results = find_similar(fp, library, ids, k=count)
        click.echo(f"\nTop {count} matches:\n")
        for rank, (song_id, score) in enumerate(results, 1):
            r = store.get_song_by_id(song_id)
            confidence = "high" if score > 0.85 else "medium" if score > 0.65 else "low"
            click.echo(f"  {rank}. [{score:.3f} {confidence}]  {_format_song(r)}")


@cli.command()
@click.option("--like", "likes", multiple=True, required=True, help="Seed song(s) to build toward.")
@click.option("--unlike", "dislikes", multiple=True, help="Song(s) to steer away from.")
@click.option("--length", default="60m", show_default=True, help="Target duration (e.g. 60m, 1h30m, 90m).")
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None, help="Output M3U file path.")
@click.option("--db", type=click.Path(path_type=Path), default=None)
def playlist(likes: tuple, dislikes: tuple, length: str, output: Path | None, db: Path | None):
    """Generate a playlist from seed songs."""
    db_path = db or DB_PATH

    # Parse duration string
    import re
    total_s = 0.0
    for m in re.finditer(r"(\d+)(h|m|s)", length):
        n, unit = int(m.group(1)), m.group(2)
        total_s += n * {"h": 3600, "m": 60, "s": 1}[unit]
    if total_s == 0:
        click.echo(f"Could not parse duration: {length}", err=True)
        sys.exit(1)

    with Store(db_path) as store:
        seed_rows = [_resolve_song(store, q) for q in likes]
        dislike_rows = [_resolve_song(store, q) for q in dislikes]

        seed_ids = {r["id"] for r in seed_rows}
        dislike_ids = {r["id"] for r in dislike_rows}

        library, ids = _load_library(store)
        if len(ids) == 0:
            click.echo("Library is empty.", err=True)
            sys.exit(1)

        song_durations = {}
        for song_id in ids:
            r = store.get_song_by_id(song_id)
            if r and r["duration_s"]:
                song_durations[song_id] = r["duration_s"]

        track_ids = playlist_walk(
            seed_ids=seed_ids,
            library_vecs=library,
            library_ids=ids,
            target_duration_s=total_s,
            song_durations=song_durations,
            dislike_ids=dislike_ids,
        )

        # Output
        lines = ["#EXTM3U"]
        total_actual = 0.0
        for song_id in track_ids:
            r = store.get_song_by_id(song_id)
            dur = int(r["duration_s"] or 0)
            total_actual += dur
            artist = r["artist"] or ""
            title = r["title"] or Path(r["path"]).stem
            lines.append(f"#EXTINF:{dur},{artist} - {title}")
            lines.append(r["path"])

        m3u = "\n".join(lines) + "\n"
        m_actual, s_actual = divmod(int(total_actual), 60)

        if output:
            output.write_text(m3u)
            click.echo(f"Wrote {len(track_ids)} tracks ({m_actual}:{s_actual:02d}) to {output}")
        else:
            click.echo(m3u)


@cli.command()
@click.option("-n", "--count", default=None, type=int, help="Number of clusters (K-means).")
@click.option("--auto", is_flag=True, help="Auto-detect cluster count (HDBSCAN).")
@click.option("--export", "export_path", type=click.Path(path_type=Path), default=None, help="Export cluster assignments as JSON.")
@click.option("--db", type=click.Path(path_type=Path), default=None)
def cluster(count: int | None, auto: bool, export_path: Path | None, db: Path | None):
    """Cluster the library by perceptual similarity."""
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize

    db_path = db or DB_PATH

    if not count and not auto:
        click.echo("Specify --n COUNT or --auto", err=True)
        sys.exit(1)

    with Store(db_path) as store:
        library, ids = _load_library(store)
        if len(ids) < 2:
            click.echo("Need at least 2 songs to cluster.", err=True)
            sys.exit(1)

        click.echo(f"Loaded {len(ids)} fingerprints. Reducing dimensions...")
        pca = PCA(n_components=min(256, len(ids), library.shape[1]))
        reduced = pca.fit_transform(library)
        reduced = normalize(reduced)

        if auto:
            try:
                import hdbscan
            except ImportError:
                click.echo("hdbscan not installed. Run: pip install hdbscan", err=True)
                sys.exit(1)
            click.echo("Running HDBSCAN...")
            clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
            labels = clusterer.fit_predict(reduced)
            method = "hdbscan"
        else:
            click.echo(f"Running K-means with k={count}...")
            km = KMeans(n_clusters=count, n_init=10, random_state=42)
            labels = km.fit_predict(reduced)
            method = f"kmeans-{count}"

        # Store cluster assignments
        from sklearn.metrics.pairwise import euclidean_distances
        centroids = {}
        for cluster_id in set(labels):
            if cluster_id == -1:
                continue
            mask = labels == cluster_id
            centroids[cluster_id] = reduced[mask].mean(axis=0)

        for i, (song_id, label) in enumerate(zip(ids, labels)):
            if label == -1:
                dist = 999.0
            else:
                dist = float(np.linalg.norm(reduced[i] - centroids[label]))
            store.upsert_cluster(song_id, int(label), dist, method)

        n_clusters = len(set(labels) - {-1})
        n_noise = (labels == -1).sum() if auto else 0
        click.echo(f"Found {n_clusters} clusters. Noise points: {n_noise}")

        if export_path:
            import json
            output = {}
            for song_id, label in zip(ids, labels):
                r = store.get_song_by_id(song_id)
                output[str(song_id)] = {
                    "cluster": int(label),
                    "path": r["path"],
                    "artist": r["artist"],
                    "title": r["title"],
                }
            export_path.write_text(json.dumps(output, indent=2))
            click.echo(f"Exported cluster assignments to {export_path}")


@cli.command()
@click.argument("target")
@click.option("--db", type=click.Path(path_type=Path), default=None)
def bpm(target: str, db: Path | None):
    """Show the estimated BPM for TARGET."""
    db_path = db or DB_PATH
    with Store(db_path) as store:
        row = _resolve_song(store, target)
        if row["bpm_est"]:
            click.echo(f"{_format_song(row)}: {row['bpm_est']} BPM")
        else:
            click.echo(f"{_format_song(row)}: no clear BPM detected")


@cli.command()
@click.option("--db", type=click.Path(path_type=Path), default=None)
def stats(db: Path | None):
    """Show library statistics."""
    db_path = db or DB_PATH
    with Store(db_path) as store:
        s = store.stats()
        click.echo(f"Songs:         {s['total_songs']}")
        click.echo(f"Fingerprinted: {s['fingerprinted']}")
        click.echo(f"Clustered:     {s['clustered']}")
        click.echo(f"Database:      {db_path}")


if __name__ == "__main__":
    cli()
