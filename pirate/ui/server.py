"""FastAPI web UI server for PIRATE."""

import os
import re
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from ..store import Store
from ..similarity import find_similar, rerank_with_chunks
from ..config import DB_PATH

STATIC_DIR = Path(__file__).parent / "static"

AUDIO_CONTENT_TYPES = {
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
    ".aac": "audio/aac",
    ".wma": "audio/x-ms-wma",
}


# ---------------------------------------------------------------------------
# App state — loaded once at startup, shared across requests
# ---------------------------------------------------------------------------

class _State:
    store: Store | None = None
    library_vecs: np.ndarray | None = None
    library_ids: list[int] = []

    def load(self):
        db = Path(os.environ.get("PIRATE_DB_PATH", str(DB_PATH)))
        self.store = Store(db)
        pairs = list(self.store.all_fingerprints())
        if pairs:
            ids, vecs = zip(*pairs)
            self.library_ids = list(ids)
            self.library_vecs = np.stack(vecs)
        else:
            self.library_ids = []
            self.library_vecs = np.empty((0,), dtype=np.float32)


state = _State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.load()
    yield
    if state.store:
        state.store.close()


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# Audio serving with Range request support (required for seeking)
# ---------------------------------------------------------------------------

@app.get("/audio/{song_id}")
async def serve_audio(song_id: int, request: Request):
    row = state.store.get_song_by_id(song_id)
    if not row:
        raise HTTPException(404, "Song not found")

    path = row["path"]
    if not os.path.exists(path):
        raise HTTPException(404, f"File not found on disk: {path}")

    file_size = os.path.getsize(path)
    content_type = AUDIO_CONTENT_TYPES.get(Path(path).suffix.lower(), "audio/mpeg")

    range_header = request.headers.get("Range")
    if range_header:
        m = re.match(r"bytes=(\d+)-(\d*)", range_header)
        if not m:
            raise HTTPException(416, "Invalid Range header")
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else file_size - 1
        end = min(end, file_size - 1)
        length = end - start + 1

        def _iter():
            with open(path, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(65536, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        return StreamingResponse(
            _iter(), status_code=206,
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(length),
                "Content-Type": content_type,
            },
        )

    def _iter_full():
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                yield chunk

    return StreamingResponse(
        _iter_full(),
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
            "Content-Type": content_type,
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _song_dict(row, cluster_id: int | None = None) -> dict:
    dur = int(row["duration_s"] or 0)
    return {
        "id": row["id"],
        "artist": row["artist"] or "",
        "title": row["title"] or Path(row["path"]).stem,
        "album": row["album"] or "",
        "duration_s": dur,
        "duration": f"{dur // 60}:{dur % 60:02d}" if dur else "",
        "bpm": round(row["bpm_est"], 1) if row["bpm_est"] else None,
        "cluster": cluster_id,
    }


def _attach_clusters(rows) -> list[dict]:
    ids = [r["id"] for r in rows]
    cluster_map = {}
    if ids:
        ph = ",".join("?" * len(ids))
        cluster_rows = state.store._conn.execute(
            f"SELECT song_id, cluster_id FROM clusters WHERE method='kmeans-20' AND song_id IN ({ph})",
            ids,
        ).fetchall()
        cluster_map = {r["song_id"]: r["cluster_id"] for r in cluster_rows}
    return [_song_dict(r, cluster_map.get(r["id"])) for r in rows]


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.get("/api/songs")
async def list_songs(q: str = Query(default="")):
    rows = state.store.search_songs(q) if q else state.store.all_songs()
    return _attach_clusters(rows)


@app.get("/api/songs/{song_id}")
async def get_song(song_id: int):
    row = state.store.get_song_by_id(song_id)
    if not row:
        raise HTTPException(404)
    cluster_row = state.store._conn.execute(
        "SELECT cluster_id FROM clusters WHERE song_id=? AND method='kmeans-20'", (song_id,)
    ).fetchone()
    return _song_dict(row, cluster_row["cluster_id"] if cluster_row else None)


@app.get("/api/similar/{song_id}")
async def similar_songs(song_id: int, k: int = Query(default=12)):
    fp = state.store.get_fingerprint(song_id)
    if fp is None:
        raise HTTPException(404, "No fingerprint for this song")
    if not state.library_ids:
        return {"results": [], "near_dupes": []}

    row = state.store.get_song_by_id(song_id)
    exclude_ids = set()
    if row:
        same = state.store._conn.execute(
            "SELECT id FROM songs WHERE file_hash=?", (row["file_hash"],)
        ).fetchall()
        exclude_ids = {r["id"] for r in same}

    candidates = find_similar(
        fp, state.library_vecs, state.library_ids,
        k=max(k * 10, 100), exclude_ids=exclude_ids,
    )
    query_chunks = state.store.get_chunks(song_id)
    if query_chunks:
        results = rerank_with_chunks(candidates, query_chunks, state.store, k=k)
    else:
        results = candidates[:k]

    def _enrich(pairs):
        out = []
        for sid, score in pairs:
            r = state.store.get_song_by_id(sid)
            if r:
                d = _song_dict(r)
                d["score"] = round(score, 4)
                out.append(d)
        return out

    return {"results": _enrich(results)}


@app.get("/api/stats")
async def stats():
    s = state.store.stats()
    s["library_loaded"] = len(state.library_ids)
    return s


# ---------------------------------------------------------------------------
# Static files + SPA root
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return HTMLResponse((STATIC_DIR / "index.html").read_text())
