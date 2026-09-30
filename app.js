"use strict";
// Curvature Explorer: UI, worker pool (worker.js = Pyodide + py/explorer.py), drawing and statistics.

const CONTROLS = [
  { k: "sigma", fs: "cannyFs", label: "Smoothing σ (px)", hint: "blur before the gradient", min: 0.5, max: 5, step: 0.25, def: 2 },
  { k: "qlo", fs: "cannyFs", label: "Low threshold (quantile)", hint: "weaker edges kept if connected", min: 0.5, max: 0.98, step: 0.01, def: 0.85 },
  { k: "qhi", fs: "cannyFs", label: "High threshold (quantile)", hint: "edges strong enough to start a chain", min: 0.6, max: 0.995, step: 0.005, def: 0.95 },
  { k: "minlen", fs: "cannyFs", label: "Shortest chain (px)", hint: "shorter chains are dropped", min: 20, max: 300, step: 10, def: 60 },
  { k: "W", fs: "aacFs", label: "Window W (px)", hint: "length of contour judged at once", min: 24, max: 256, step: 8, def: 96 },
  { k: "sfrac", fs: "aacFs", label: "Contour smoothing (× W)", hint: "Gaussian σ along the contour", min: 0.02, max: 0.5, step: 0.005, def: 0.125 },
  { k: "mfrac", fs: "aacFs", label: "Median tangent filter (× W)", hint: "removes wobble; 0 = off", min: 0, max: 1, step: 0.05, def: 0.5 },
  { k: "straight", fs: "aacFs", label: "Straight if turning below (°)", hint: "", min: 1, max: 30, step: 1, def: 8 },
  { k: "conc", fs: "aacFs", label: "Corner if a quarter window holds ≥", hint: "share of the net turning", min: 0.2, max: 1, step: 0.05, def: 0.6 },
  { k: "netpath", fs: "aacFs", label: "Arc if net / path turning ≥", hint: "turns one way, steadily", min: 0.3, max: 1, step: 0.05, def: 0.8 },
];
const DEFAULTS = Object.fromEntries(CONTROLS.map((c) => [c.k, c.def]));
const CODE_VAR = ["--none", "--straight", "--arc", "--corner", "--jagged"];
const $ = (id) => document.getElementById(id);
const decimals = (step) => (String(step).split(".")[1] || "").length;
const snap = (c, v) => +(Math.round(Math.min(c.max, Math.max(c.min, v)) / c.step) * c.step).toFixed(decimals(c.step));
const pkey = (p) => JSON.stringify(CONTROLS.map((c) => p[c.k]));
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

const S = {
  rooms: [], byId: {}, params: { ...DEFAULTS }, sel: null, results: {}, preview: null,
  photos: {}, normals: {}, workers: [], run: null, refStats: null,
};

// ------------------------------------------------------------------ statistics
function mean(a) { return a.reduce((s, x) => s + x, 0) / a.length; }
function pearson(x, y) {
  const mx = mean(x), my = mean(y); let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < x.length; i++) { const dx = x[i] - mx, dy = y[i] - my; sxy += dx * dy; sxx += dx * dx; syy += dy * dy; }
  return sxy / Math.sqrt(sxx * syy);
}
function ranks(a) {
  const idx = a.map((v, i) => [v, i]).sort((p, q) => p[0] - q[0]); const r = new Array(a.length);
  for (let i = 0; i < idx.length;) { let j = i; while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++; for (let k = i; k <= j; k++) r[idx[k][1]] = (i + j) / 2 + 1; i = j + 1; }
  return r;
}
function cohenD(x, g) {
  const a = x.filter((_, i) => g[i]), b = x.filter((_, i) => !g[i]);
  const va = a.reduce((s, v) => s + (v - mean(a)) ** 2, 0), vb = b.reduce((s, v) => s + (v - mean(b)) ** 2, 0);
  return (mean(a) - mean(b)) / Math.sqrt((va + vb) / (a.length + b.length - 2));
}
/** Stats over all rooms for one result set; undefined scores take the mean of the defined ones (as in the research code). */
function statsFor(res) {
  const defined = S.rooms.map((r) => res[r.id]).filter((v) => v != null && isFinite(v));
  if (defined.length < 3) return null;
  const m = mean(defined);
  const x = S.rooms.map((r) => (res[r.id] != null && isFinite(res[r.id]) ? res[r.id] : m));
  const y = S.rooms.map((r) => r.rating), g = S.rooms.map((r) => r.label === "round");
  return { r: pearson(x, y), rho: pearson(ranks(x), ranks(y)), d: cohenD(x, g), undefinedN: S.rooms.length - defined.length, x, y };
}
const fmt = (v, n = 3) => (v == null || !isFinite(v) ? "–" : (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(n));

// ------------------------------------------------------------------ workers
const NW = Math.max(2, Math.min((navigator.hardwareConcurrency || 4) - 1, 5));
const q0 = [];   // worker 0 queue (preview + normals); a newer preview replaces a queued one
function startWorkers() {
  for (let i = 0; i < NW; i++) {
    const w = new Worker("worker.js", { type: "module" }); w.ready = false; w.busy = false; w.i = i;
    w.onmessage = (e) => onMessage(w, e.data);
    w.onerror = (e) => setStatus(`Worker error: ${e.message}`, "error");
    S.workers.push(w);
  }
  setStatus(`Python is still loading (${NW} browser workers; the first visit downloads about 40 MB and takes 20-30 s). Settings and Run work once it is ready.`, "loading");
}
function readyCount() { return S.workers.filter((w) => w.ready).length; }
function onMessage(w, m) {
  if (m.type === "ready") {
    w.ready = true;
    const k = readyCount();
    if (k === NW) setStatus("Ready. Change a setting to recompute the selected room; press Run to score every room.", "ready");
    else setStatus(`Python is still loading: ${k} of ${NW} workers ready. Settings and Run work once it is ready.`, "loading");
    renderResults();
    if (w.i === 0) pump0(); else dispatch();
    return;
  }
  if (m.type === "error" && !m.req) { setStatus(`Python failed to load: ${m.error}`, "error"); return; }
  w.busy = false;
  if (m.error) { setStatus(`Error on ${m.id}: ${m.error}`, "error"); console.error(m.error); }
  else if (m.type === "preview") onPreview(m);
  else if (m.type === "normals") { S.normals[m.id] = toCanvas(m.result.rgba, S.byId[m.id]); if (m.id === S.sel) draw(); }
  else if (m.type === "score") onScore(m);
  if (w.i === 0) pump0(); else dispatch();
}
function send0(msg) {
  if (msg.type === "preview") { const i = q0.findIndex((x) => x.type === "preview"); if (i >= 0) q0.splice(i, 1); }
  q0.push(msg); pump0();
}
function pump0() {
  const w = S.workers[0];
  if (!w || !w.ready || w.busy || !q0.length) return;
  w.busy = true; w.postMessage(q0.shift());
}

// ------------------------------------------------------------------ preview of the selected room
let pvTimer = null;
function requestPreview() {
  clearTimeout(pvTimer);
  pvTimer = setTimeout(() => { $("viewBusy").hidden = false; send0({ type: "preview", id: S.sel, params: { ...S.params }, req: pkey(S.params) }); }, 150);
}
function onPreview(m) {
  store(m.req, m.id, m.result.score);
  if (m.id !== S.sel) return;
  S.preview = { ...m.result, id: m.id, key: m.req };
  $("viewBusy").hidden = m.req === pkey(S.params);
  draw(); roomInfo(); renderResults(); updateList();
}
function store(key, id, score) { (S.results[key] ||= {})[id] = score == null ? NaN : score; }

// ------------------------------------------------------------------ run over every room
function run() {
  const key = pkey(S.params), res = (S.results[key] ||= {});
  const todo = S.rooms.map((r) => r.id).filter((id) => !(id in res));
  if (!todo.length) { renderResults(); return; }
  S.run = { key, params: { ...S.params }, todo, total: S.rooms.length };
  $("progress").hidden = false; progress(); dispatch();
}
function dispatch() {
  const R = S.run; if (!R) return;
  for (const w of S.workers.slice(1)) {
    if (!w.ready || w.busy || !R.todo.length) continue;
    w.busy = true; w.postMessage({ type: "score", id: R.todo.shift(), params: R.params, req: R.key });
  }
}
let renderTimer = null;
function onScore(m) {
  store(m.req, m.id, m.result.score);
  if (S.run && S.run.key === m.req) {
    progress();
    const done = Object.keys(S.results[m.req]).length >= S.rooms.length;
    if (done) { S.run = null; $("progress").hidden = true; }
    clearTimeout(renderTimer); renderTimer = setTimeout(() => { renderResults(); updateList(); roomInfo(); }, done ? 0 : 300);
  }
}
function progress() {
  const R = S.run; if (!R) return;
  const n = Object.keys(S.results[R.key] || {}).length;
  $("progress").firstElementChild.style.width = `${(100 * n) / R.total}%`;
  $("run").textContent = `Scoring… ${n} of ${R.total}`;
}

// ------------------------------------------------------------------ controls
function buildControls() {
  for (const c of CONTROLS) {
    const d = document.createElement("div"); d.className = "ctl"; d.id = `ctl-${c.k}`;
    d.innerHTML = `<label for="r-${c.k}"><span>${c.label}</span><span class="hint">${c.hint}</span></label>
      <input type="range" id="r-${c.k}" min="${c.min}" max="${c.max}" step="${c.step}" value="${c.def}">
      <input type="number" id="n-${c.k}" min="${c.min}" max="${c.max}" step="${c.step}" value="${c.def}" aria-label="${c.label}">`;
    $(c.fs).appendChild(d);
    const set = (v) => setParam(c, v);
    d.querySelector("input[type=range]").addEventListener("input", (e) => set(+e.target.value));
    d.querySelector("input[type=number]").addEventListener("change", (e) => set(+e.target.value));
  }
  $("reset").addEventListener("click", () => { S.params = { ...DEFAULTS }; syncControls(); changed(); });
  $("run").addEventListener("click", run);
}
function setParam(c, v) {
  if (!isFinite(v)) return;
  S.params[c.k] = snap(c, v);
  const lo = CONTROLS.find((x) => x.k === "qlo"), hi = CONTROLS.find((x) => x.k === "qhi");
  if (S.params.qlo >= S.params.qhi) {
    if (c.k === "qlo") S.params.qhi = snap(hi, S.params.qlo + 0.01); else S.params.qlo = snap(lo, S.params.qhi - 0.01);
    if (S.params.qlo >= S.params.qhi) S.params.qlo = snap(lo, S.params.qhi - 0.01);
  }
  syncControls(); changed();
}
function syncControls() {
  for (const c of CONTROLS) {
    $(`r-${c.k}`).value = S.params[c.k]; $(`n-${c.k}`).value = S.params[c.k];
    $(`ctl-${c.k}`).classList.toggle("changed", S.params[c.k] !== c.def);
  }
}
function changed() {
  requestPreview(); renderResults(); updateList();
}

// ------------------------------------------------------------------ drawing
function toCanvas(rgba, room) {
  const c = document.createElement("canvas"); c.width = room.width; c.height = room.height;
  c.getContext("2d").putImageData(new ImageData(new Uint8ClampedArray(rgba.buffer, rgba.byteOffset, rgba.byteLength), room.width, room.height), 0, 0);
  return c;
}
function edgeCanvas(p) {
  const n = p.width * p.height, rgba = new Uint8ClampedArray(n * 4);
  const col = css("--text").match(/\w\w/g)?.map((h) => parseInt(h, 16)) || [255, 255, 255];
  for (let i = 0; i < n; i++) if (p.edges[i]) { rgba[4 * i] = col[0]; rgba[4 * i + 1] = col[1]; rgba[4 * i + 2] = col[2]; rgba[4 * i + 3] = 255; }
  return toCanvas(rgba, p);
}
function draw() {
  const room = S.byId[S.sel], cv = $("view"), ctx = cv.getContext("2d");
  cv.width = room.width; cv.height = room.height;
  const base = document.querySelector("input[name=base]:checked").value;
  if (base === "normals") {
    if (S.normals[room.id]) ctx.drawImage(S.normals[room.id], 0, 0);
    else { ctx.fillStyle = css("--line"); ctx.fillRect(0, 0, cv.width, cv.height); send0({ type: "normals", id: room.id }); }
  } else {
    const img = S.photos[room.id];
    if (img && img.complete) ctx.drawImage(img, 0, 0);
  }
  if ($("dim").checked) { ctx.globalAlpha = 0.5; ctx.fillStyle = css("--panel"); ctx.fillRect(0, 0, cv.width, cv.height); ctx.globalAlpha = 1; }
  const p = S.preview;
  if (!p || p.id !== room.id) return;
  if ($("showEdges").checked) {
    if (!p.edgeCanvas) p.edgeCanvas = edgeCanvas(p);
    ctx.drawImage(p.edgeCanvas, 0, 0);
  }
  if ($("showContours").checked) {
    const pts = new Float32Array(p.pts.buffer, p.pts.byteOffset, p.pts.byteLength / 4);
    const off = new Int32Array(p.offsets.buffer, p.offsets.byteOffset, p.offsets.byteLength / 4);
    const paths = CODE_VAR.map(() => new Path2D());
    for (let c = 0; c + 1 < off.length; c++) {
      for (let i = off[c]; i < off[c + 1] - 1; i++) {
        const path = paths[p.codes[i]];
        path.moveTo(pts[2 * i + 1], pts[2 * i]); path.lineTo(pts[2 * i + 3], pts[2 * i + 2]);
      }
    }
    ctx.lineCap = "round"; ctx.lineWidth = Math.max(2, room.width / 320);
    for (const code of [0, 1, 4, 2, 3]) { ctx.strokeStyle = css(CODE_VAR[code]); ctx.stroke(paths[code]); }
  }
}

// ------------------------------------------------------------------ room info, list, results
function currentScores() { return S.results[pkey(S.params)] || {}; }
function pct(v, arr) { const a = arr.filter((x) => isFinite(x)); return Math.round((100 * a.filter((x) => x < v).length) / Math.max(1, a.length - 1)); }
function roomInfo() {
  const r = S.byId[S.sel]; if (!r) return;
  const res = currentScores(), s = res[r.id], p = S.preview && S.preview.id === r.id && S.preview.key === pkey(S.params) ? S.preview : null;
  const all = Object.values(res), complete = all.length >= S.rooms.length;
  $("roomInfo").innerHTML = `
    <div>Room <b>${r.key.replace(/\.\w+$/, "")}</b></div>
    <div>${r.source} · <span class="tag ${r.label}">${r.label}</span></div>
    <div>Rating <b>${fmt(r.rating, 2)}</b> (${pct(r.rating, S.rooms.map((x) => x.rating))}th pct)</div>
    <div>Score <b>${s == null ? "…" : isFinite(s) ? s.toFixed(3) : "undefined"}</b>${complete && isFinite(s) ? ` (${pct(s, all)}th pct)` : ""}</div>
    ${p ? `<div>Arc <b>${p.arc}</b> · corner <b>${p.corner}</b></div><div>Straight <b>${p.straight}</b> · jagged <b>${p.jagged}</b> · chains <b>${p.chains}</b></div>` : ""}`;
}
function fitResid(res) {
  const st = statsFor(res); if (!st) return null;
  const mx = mean(st.x), my = mean(st.y); let b = 0, sxx = 0;
  st.x.forEach((x, i) => { b += (x - mx) * (st.y[i] - my); sxx += (x - mx) ** 2; }); b /= sxx;
  return Object.fromEntries(S.rooms.map((r, i) => [r.id, Math.abs(st.y[i] - (my + b * (st.x[i] - mx)))]));
}
function buildList() {
  const ol = $("rooms");
  for (const r of S.rooms) {
    const li = document.createElement("li"); li.dataset.id = r.id;
    li.innerHTML = `<button type="button" title="${r.key}"><img loading="lazy" src="data/rooms/thumbs/${r.id}.jpg" alt="">
      <span class="cap"><span class="tag ${r.label}">${r.label}</span><span>${fmt(r.rating, 2)}</span><span class="sc"></span></span></button>`;
    li.querySelector("button").addEventListener("click", () => select(r.id));
    ol.appendChild(li);
  }
  $("sort").addEventListener("change", updateList);
}
function updateList() {
  const res = currentScores(), how = $("sort").value, resid = how === "resid" ? fitResid(res) || {} : {};
  const val = (r) => ({ rating: -r.rating, score: -(isFinite(res[r.id]) ? res[r.id] : -1), resid: -(resid[r.id] ?? -1), name: 0 }[how]);
  const order = [...S.rooms].sort((a, b) => val(a) - val(b) || a.key.localeCompare(b.key));
  const ol = $("rooms");
  for (const r of order) {
    const li = ol.querySelector(`li[data-id="${r.id}"]`); ol.appendChild(li);
    li.querySelector(".sc").textContent = isFinite(res[r.id]) ? res[r.id].toFixed(2) : "";
    li.querySelector("button").setAttribute("aria-current", r.id === S.sel ? "true" : "false");
  }
}
function renderResults() {
  const key = pkey(S.params), res = S.results[key] || {}, n = Object.keys(res).length, complete = n >= S.rooms.length;
  const st = complete ? statsFor(res) : null, ref = S.refStats, isDef = key === pkey(DEFAULTS);
  const tile = (k, v, rv, digits) => `<div class="stat"><div class="k">${k}</div><div class="v">${st ? fmt(v, digits) : "–"}</div>
    <div class="ref">${isDef ? "defaults" : `defaults ${fmt(rv, digits)}`}</div></div>`;
  $("stats").innerHTML = tile("Pearson r", st?.r, ref.r, 3) + tile("Spearman ρ", st?.rho, ref.rho, 3) + tile("Round vs square d", st?.d, ref.d, 2);
  const runBtn = $("run");
  if (!S.run || S.run.key !== key) { const pyReady = S.workers.slice(1).some((w) => w.ready);
    runBtn.textContent = complete ? "All rooms scored" : pyReady ? `Run on all ${S.rooms.length} rooms` : "Run (waiting for Python)";
    runBtn.disabled = complete || !pyReady; }
  $("scatterNote").textContent = complete
    ? `Each dot is a room (${S.rooms.length}); click one to view it.${st.undefinedN ? ` ${st.undefinedN} rooms had no arc or corner points and take the mean score.` : ""}`
    : `These settings have scores for ${n} of ${S.rooms.length} rooms. Press Run to score the rest (the first run downloads the normal maps, about 410 MB). Grey dots: the default settings.`;
  scatter(res, complete);
}
function scatter(res, complete) {
  const svg = $("scatter"), W = 420, H = 300, m = { l: 40, r: 10, t: 10, b: 34 };
  const ys = S.rooms.map((r) => r.rating), y0 = Math.floor(Math.min(...ys) * 2) / 2, y1 = Math.ceil(Math.max(...ys) * 2) / 2;
  const X = (v) => m.l + v * (W - m.l - m.r), Y = (v) => H - m.b - ((v - y0) / (y1 - y0)) * (H - m.t - m.b);
  let g = `<line class="axis" x1="${m.l}" y1="${H - m.b}" x2="${W - m.r}" y2="${H - m.b}"/><line class="axis" x1="${m.l}" y1="${m.t}" x2="${m.l}" y2="${H - m.b}"/>`;
  for (const t of [0, 0.25, 0.5, 0.75, 1]) g += `<text x="${X(t)}" y="${H - m.b + 14}" text-anchor="middle">${t}</text>`;
  for (let t = y0; t <= y1 + 1e-9; t += 0.5) g += `<text x="${m.l - 6}" y="${Y(t) + 4}" text-anchor="end">${t.toFixed(1)}</text>`;
  g += `<text x="${(m.l + W - m.r) / 2}" y="${H - 4}" text-anchor="middle">score = arc / (arc + corner)</text>`;
  g += `<text transform="translate(11 ${(m.t + H - m.b) / 2}) rotate(-90)" text-anchor="middle">curvedness rating</text>`;
  let sel = "";
  if (!complete) {   // faint reference: the default settings' scores
    const def = S.results[pkey(DEFAULTS)];
    for (const r of S.rooms) if (isFinite(def[r.id])) g += `<circle class="ghost" cx="${X(def[r.id]).toFixed(1)}" cy="${Y(r.rating).toFixed(1)}" r="3"/>`;
  }
  for (const r of S.rooms) {
    const v = res[r.id]; if (v == null || !isFinite(v)) continue;
    const c = `<circle data-id="${r.id}" class="${r.label}${r.id === S.sel ? " selected" : ""}" cx="${X(v).toFixed(1)}" cy="${Y(r.rating).toFixed(1)}" r="${r.id === S.sel ? 5.5 : 3.5}" fill-opacity="${complete ? 0.8 : 0.5}"><title>${r.key} · ${r.label} · rating ${r.rating.toFixed(2)} · score ${v.toFixed(3)}</title></circle>`;
    if (r.id === S.sel) sel = c; else g += c;
  }
  g += sel;
  g += `<g transform="translate(${W - m.r - 110} ${m.t + 4})"><circle cx="5" cy="5" r="4" class="round"/><text x="14" y="9">round</text><circle cx="65" cy="5" r="4" class="square"/><text x="74" y="9">square</text></g>`;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.innerHTML = g;
}

// ------------------------------------------------------------------ selection
function select(id) {
  S.sel = id; S.preview = S.preview && S.preview.id === id ? S.preview : null;
  if (!S.photos[id]) { const img = new Image(); img.onload = () => { if (S.sel === id) draw(); }; img.src = `data/rooms/photos/${id}.jpg`; S.photos[id] = img; }
  draw(); roomInfo(); updateList(); renderResults(); requestPreview();
}
// kind: "loading" or "error" show a large red banner; "ready" shows a quiet line
function setStatus(text, kind = "ready") {
  const s = $("status"); s.textContent = text;
  s.classList.toggle("alert", kind !== "ready"); s.classList.toggle("err", kind === "error");
}

// ------------------------------------------------------------------ start
async function main() {
  const man = await (await fetch("data/rooms/manifest.json", { cache: "no-cache" })).json();
  S.rooms = man.rooms; S.byId = Object.fromEntries(S.rooms.map((r) => [r.id, r]));
  S.results[pkey(DEFAULTS)] = Object.fromEntries(S.rooms.map((r) => [r.id, r.n_default]));
  S.refStats = statsFor(S.results[pkey(DEFAULTS)]);
  buildControls(); buildList();
  document.querySelectorAll("input[name=base], #showEdges, #showContours, #dim").forEach((e) => e.addEventListener("change", draw));
  $("scatter").addEventListener("click", (e) => { const id = e.target.closest("circle[data-id]")?.dataset.id; if (id) select(id); });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { if (S.preview) S.preview.edgeCanvas = null; draw(); });
  startWorkers();
  select([...S.rooms].sort((a, b) => b.rating - a.rating)[0].id);
}
main().catch((e) => setStatus(`Failed to start: ${e}`, "error"));
