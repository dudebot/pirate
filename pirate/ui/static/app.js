// PIRATE web UI — vanilla JS, no framework

const $ = id => document.getElementById(id);
const audio = $('audio');

// ── State ──────────────────────────────────────────────────────────────────────

const S = {
  songs: [],       // full library from /api/songs
  filtered: [],    // after client-side search filter
  sortKey: 'artist',
  sortDir: 1,      // 1 = asc, -1 = desc
  selectedId: null,
  queue: [],
  queuePos: -1,
};

// ── Utilities ──────────────────────────────────────────────────────────────────

const esc = s => String(s)
  .replace(/&/g, '&amp;').replace(/</g, '&lt;')
  .replace(/>/g, '&gt;').replace(/"/g, '&quot;');

const fmtDur = s => s
  ? `${Math.floor(s / 60)}:${String(Math.floor(s) % 60).padStart(2, '0')}`
  : '';

// ── Library loading ────────────────────────────────────────────────────────────

async function loadLibrary(q = '') {
  const url = q ? `/api/songs?q=${encodeURIComponent(q)}` : '/api/songs';
  try {
    const songs = await fetch(url).then(r => r.json());
    S.songs = songs;
    S.filtered = songs;
    renderLibrary();
  } catch (e) {
    $('library-body').innerHTML = `<tr><td colspan="6" class="loading">Error loading library: ${esc(String(e))}</td></tr>`;
  }
}

async function loadStats() {
  try {
    const s = await fetch('/api/stats').then(r => r.json());
    $('stats').textContent = `${s.total_songs.toLocaleString()} songs · ${s.fingerprinted.toLocaleString()} fingerprinted`;
  } catch (_) {}
}

// ── Rendering ──────────────────────────────────────────────────────────────────

function renderLibrary() {
  const { filtered, sortKey, sortDir, selectedId } = S;
  const playingId = S.queue[S.queuePos] ?? null;

  const sorted = [...filtered].sort((a, b) => {
    const av = a[sortKey] ?? '', bv = b[sortKey] ?? '';
    if (typeof av === 'number' && typeof bv === 'number') return (av - bv) * sortDir;
    return String(av).localeCompare(String(bv)) * sortDir;
  });

  // Update sort indicators on headers
  document.querySelectorAll('#library-table th[data-sort]').forEach(th => {
    th.classList.remove('sort-asc', 'sort-desc');
    if (th.dataset.sort === sortKey) {
      th.classList.add(sortDir === 1 ? 'sort-asc' : 'sort-desc');
    }
  });

  if (!sorted.length) {
    $('library-body').innerHTML = '<tr><td colspan="6" class="loading">No songs found.</td></tr>';
    return;
  }

  $('library-body').innerHTML = sorted.map(s => {
    const selected = s.id === selectedId ? 'selected' : '';
    const playing  = s.id === playingId  ? 'playing'  : '';
    const cluster  = s.cluster !== null && s.cluster !== undefined
      ? `<span class="cluster-badge">${s.cluster}</span>` : '';
    return `<tr data-id="${s.id}" class="${selected} ${playing}">
      <td class="col-artist" title="${esc(s.artist)}">${esc(s.artist)}</td>
      <td class="col-title"  title="${esc(s.title)}">${esc(s.title)}</td>
      <td class="col-album"  title="${esc(s.album)}">${esc(s.album)}</td>
      <td class="col-dur">${esc(s.duration)}</td>
      <td class="col-bpm">${s.bpm ?? ''}</td>
      <td class="col-clus">${cluster}</td>
    </tr>`;
  }).join('');
}

// ── Sorting ────────────────────────────────────────────────────────────────────

$('library-table').querySelector('thead').addEventListener('click', e => {
  const th = e.target.closest('th[data-sort]');
  if (!th) return;
  const key = th.dataset.sort;
  if (S.sortKey === key) {
    S.sortDir *= -1;
  } else {
    S.sortKey = key;
    S.sortDir = 1;
  }
  renderLibrary();
});

// ── Song selection + detail panel ─────────────────────────────────────────────

async function selectSong(id) {
  S.selectedId = id;
  renderLibrary();

  const song = S.songs.find(s => s.id === id);
  if (!song) return;

  $('detail-pane').classList.remove('hidden');
  $('detail-artist').textContent = song.artist;
  $('detail-title').textContent  = song.title;

  const meta = [
    song.album    ? `<b>Album:</b> ${esc(song.album)}`           : '',
    song.duration ? `<b>Duration:</b> ${esc(song.duration)}`     : '',
    song.bpm      ? `<b>BPM:</b> ${song.bpm}`                   : '',
    song.cluster !== null && song.cluster !== undefined
                  ? `<b>Cluster:</b> ${song.cluster}`            : '',
  ].filter(Boolean).join('<br>');
  $('detail-meta').innerHTML = meta;

  $('play-now').onclick  = () => playSong(id);
  $('add-queue').onclick = () => addToQueue(id);

  $('similar-list').innerHTML = '<div class="loading">Loading…</div>';

  try {
    const data = await fetch(`/api/similar/${id}?k=12`).then(r => r.json());
    renderSimilar(data);
  } catch (e) {
    $('similar-list').innerHTML = `<div class="empty">Error: ${esc(String(e))}</div>`;
  }
}

function renderSimilar({ results = [] }) {
  if (!results.length) {
    $('similar-list').innerHTML = '<div class="empty">No similar songs found.</div>';
    return;
  }

  // Score bar: map 0.8–1.0 → 0–100%
  const pct = score => Math.max(0, Math.min(100, (score - 0.8) / 0.2 * 100)).toFixed(1);

  $('similar-list').innerHTML = results.map(s => `
    <div class="similar-item" data-id="${s.id}">
      <div class="similar-artist">${esc(s.artist)}</div>
      <div class="similar-title">${esc(s.title)}</div>
      <div class="similar-meta">
        <span>${fmtDur(s.duration_s)}</span>
        <div class="score-bar-wrap"><div class="score-bar" style="width:${pct(s.score)}%"></div></div>
        <span>${s.score.toFixed(3)}</span>
      </div>
    </div>`).join('');
}

// ── Playback ───────────────────────────────────────────────────────────────────

function playSong(id) {
  const song = S.songs.find(s => s.id === id);
  if (!song) return;

  // Replace queue with just this song (double-click = play now)
  S.queue = [id];
  S.queuePos = 0;

  audio.src = `/audio/${id}`;
  audio.play();

  $('player-track-artist').textContent = song.artist || '';
  $('player-track-title').textContent  = song.title || '';

  renderLibrary(); // highlight playing row
}

function addToQueue(id) {
  if (S.queuePos === -1) {
    // Nothing playing — start now
    S.queue = [id];
    S.queuePos = 0;
    playSong(id);
  } else {
    // Insert after current position
    S.queue.splice(S.queuePos + 1, 0, id);
  }
}

audio.addEventListener('timeupdate', () => {
  const { currentTime, duration } = audio;
  if (!isNaN(duration) && duration > 0) {
    $('player-time').textContent     = fmtDur(currentTime);
    $('player-duration').textContent = fmtDur(duration);
    $('player-progress').value       = (currentTime / duration) * 100;
  }
});

audio.addEventListener('ended', () => {
  const next = S.queuePos + 1;
  if (next < S.queue.length) {
    S.queuePos = next;
    playSong(S.queue[S.queuePos]);
  } else {
    $('btn-play').textContent = '▶';
  }
});

audio.addEventListener('play',  () => { $('btn-play').innerHTML = '&#9646;&#9646;'; });
audio.addEventListener('pause', () => { $('btn-play').innerHTML = '&#9654;'; });
audio.addEventListener('error', () => {
  $('player-track-title').textContent = 'Error loading file';
});

$('btn-play').addEventListener('click', () => audio.paused ? audio.play() : audio.pause());

$('btn-next').addEventListener('click', () => {
  const next = S.queuePos + 1;
  if (next < S.queue.length) { S.queuePos = next; playSong(S.queue[S.queuePos]); }
});

$('btn-prev').addEventListener('click', () => {
  if (audio.currentTime > 3) { audio.currentTime = 0; return; }
  const prev = S.queuePos - 1;
  if (prev >= 0) { S.queuePos = prev; playSong(S.queue[S.queuePos]); }
});

$('player-progress').addEventListener('input', e => {
  if (!isNaN(audio.duration)) {
    audio.currentTime = (e.target.value / 100) * audio.duration;
  }
});

$('player-volume').addEventListener('input', e => { audio.volume = parseFloat(e.target.value); });
audio.volume = 0.8;

// ── Search ─────────────────────────────────────────────────────────────────────

let searchTimer;
$('search').addEventListener('input', e => {
  clearTimeout(searchTimer);
  const q = e.target.value.trim();
  // Client-side filter first (instant), then server search for metadata-rich queries
  if (q) {
    const ql = q.toLowerCase();
    S.filtered = S.songs.filter(s =>
      s.artist.toLowerCase().includes(ql) ||
      s.title.toLowerCase().includes(ql)  ||
      s.album.toLowerCase().includes(ql)
    );
    renderLibrary();
    // Also hit the server for path-based matches etc.
    searchTimer = setTimeout(() => loadLibrary(q), 400);
  } else {
    S.filtered = S.songs;
    renderLibrary();
  }
});

// ── Event delegation ───────────────────────────────────────────────────────────

$('library-body').addEventListener('click', e => {
  const tr = e.target.closest('tr[data-id]');
  if (tr) selectSong(parseInt(tr.dataset.id));
});

$('library-body').addEventListener('dblclick', e => {
  const tr = e.target.closest('tr[data-id]');
  if (tr) playSong(parseInt(tr.dataset.id));
});

$('similar-list').addEventListener('click', e => {
  const item = e.target.closest('.similar-item[data-id]');
  if (item) selectSong(parseInt(item.dataset.id));
});

$('similar-list').addEventListener('dblclick', e => {
  const item = e.target.closest('.similar-item[data-id]');
  if (item) playSong(parseInt(item.dataset.id));
});

// ── Init ───────────────────────────────────────────────────────────────────────

loadStats();
loadLibrary();
