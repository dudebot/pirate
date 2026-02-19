"""SQLite fingerprint store."""

import gzip
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np

from .config import DATA_DIR, DB_PATH, PIPELINE_VERSION

SCHEMA = """
CREATE TABLE IF NOT EXISTS songs (
    id          INTEGER PRIMARY KEY,
    path        TEXT NOT NULL UNIQUE,
    title       TEXT,
    artist      TEXT,
    album       TEXT,
    duration_s  REAL,
    bpm_est     REAL,
    scanned_at  TEXT NOT NULL,
    file_hash   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fingerprints (
    song_id     INTEGER PRIMARY KEY REFERENCES songs(id),
    vector      BLOB NOT NULL,
    version     INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS clusters (
    song_id     INTEGER REFERENCES songs(id),
    cluster_id  INTEGER NOT NULL,
    distance    REAL NOT NULL,
    method      TEXT NOT NULL,
    PRIMARY KEY (song_id, method)
);
"""


def _pack(vector: np.ndarray) -> bytes:
    """Gzip-compress a float32 numpy array to bytes."""
    return gzip.compress(vector.astype(np.float32).tobytes())


def _unpack(blob: bytes) -> np.ndarray:
    """Decompress bytes back to float32 numpy array."""
    return np.frombuffer(gzip.decompress(blob), dtype=np.float32)


class Store:
    def __init__(self, path: Path = DB_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------
    # Songs
    # ------------------------------------------------------------------

    def upsert_song(
        self,
        path: str,
        file_hash: str,
        title: str | None = None,
        artist: str | None = None,
        album: str | None = None,
        duration_s: float | None = None,
        bpm_est: float | None = None,
    ) -> int:
        """Insert or update a song record. Returns the song id."""
        now = datetime.now(timezone.utc).isoformat()
        cur = self._conn.execute(
            """
            INSERT INTO songs (path, title, artist, album, duration_s, bpm_est, scanned_at, file_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                title      = excluded.title,
                artist     = excluded.artist,
                album      = excluded.album,
                duration_s = excluded.duration_s,
                bpm_est    = excluded.bpm_est,
                scanned_at = excluded.scanned_at,
                file_hash  = excluded.file_hash
            """,
            (path, title, artist, album, duration_s, bpm_est, now, file_hash),
        )
        self._conn.commit()
        if cur.lastrowid:
            return cur.lastrowid
        row = self._conn.execute("SELECT id FROM songs WHERE path = ?", (path,)).fetchone()
        return row["id"]

    def get_song_by_path(self, path: str) -> sqlite3.Row | None:
        return self._conn.execute("SELECT * FROM songs WHERE path = ?", (path,)).fetchone()

    def get_song_by_id(self, song_id: int) -> sqlite3.Row | None:
        return self._conn.execute("SELECT * FROM songs WHERE id = ?", (song_id,)).fetchone()

    def all_songs(self) -> list[sqlite3.Row]:
        return self._conn.execute("SELECT * FROM songs ORDER BY artist, album, title").fetchall()

    def search_songs(self, query: str) -> list[sqlite3.Row]:
        q = f"%{query}%"
        return self._conn.execute(
            "SELECT * FROM songs WHERE title LIKE ? OR artist LIKE ? OR album LIKE ? OR path LIKE ?",
            (q, q, q, q),
        ).fetchall()

    # ------------------------------------------------------------------
    # Fingerprints
    # ------------------------------------------------------------------

    def upsert_fingerprint(self, song_id: int, vector: np.ndarray):
        self._conn.execute(
            """
            INSERT INTO fingerprints (song_id, vector, version)
            VALUES (?, ?, ?)
            ON CONFLICT(song_id) DO UPDATE SET vector = excluded.vector, version = excluded.version
            """,
            (song_id, _pack(vector), PIPELINE_VERSION),
        )
        self._conn.commit()

    def get_fingerprint(self, song_id: int) -> np.ndarray | None:
        row = self._conn.execute(
            "SELECT vector FROM fingerprints WHERE song_id = ?", (song_id,)
        ).fetchone()
        return _unpack(row["vector"]) if row else None

    def all_fingerprints(self) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (song_id, vector) for all songs with current-version fingerprints."""
        rows = self._conn.execute(
            "SELECT song_id, vector FROM fingerprints WHERE version = ?", (PIPELINE_VERSION,)
        ).fetchall()
        for row in rows:
            yield row["song_id"], _unpack(row["vector"])

    def needs_fingerprint(self, song_id: int, file_hash: str) -> bool:
        """True if no current-version fingerprint exists for this song."""
        row = self._conn.execute(
            "SELECT version FROM fingerprints WHERE song_id = ?", (song_id,)
        ).fetchone()
        return row is None or row["version"] != PIPELINE_VERSION

    # ------------------------------------------------------------------
    # Clusters
    # ------------------------------------------------------------------

    def upsert_cluster(self, song_id: int, cluster_id: int, distance: float, method: str):
        self._conn.execute(
            """
            INSERT INTO clusters (song_id, cluster_id, distance, method)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(song_id, method) DO UPDATE SET
                cluster_id = excluded.cluster_id,
                distance   = excluded.distance
            """,
            (song_id, cluster_id, distance, method),
        )
        self._conn.commit()

    def get_cluster_members(self, cluster_id: int, method: str) -> list[sqlite3.Row]:
        return self._conn.execute(
            """
            SELECT s.*, c.distance FROM songs s
            JOIN clusters c ON c.song_id = s.id
            WHERE c.cluster_id = ? AND c.method = ?
            ORDER BY c.distance
            """,
            (cluster_id, method),
        ).fetchall()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        total = self._conn.execute("SELECT COUNT(*) FROM songs").fetchone()[0]
        fingerprinted = self._conn.execute(
            "SELECT COUNT(*) FROM fingerprints WHERE version = ?", (PIPELINE_VERSION,)
        ).fetchone()[0]
        clustered = self._conn.execute(
            "SELECT COUNT(DISTINCT song_id) FROM clusters"
        ).fetchone()[0]
        return {
            "total_songs": total,
            "fingerprinted": fingerprinted,
            "clustered": clustered,
        }
