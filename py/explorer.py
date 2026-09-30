"""Curvature Explorer: the N measure (Marigold normals -> Canny -> A/A+C-r) with its settings exposed. Runs in the browser under Pyodide
and under CPython (scripts/web_export/parity.py). With DEFAULTS it equals acp/union.py: n_measure (M19; rooms r +0.667, d +1.20).

Pipeline: Canny (sigma, hysteresis at the qlo / qhi quantiles of the gradient magnitude) on each component of the unit normal map, OR-ed ->
chains traced between junctions (>= minlen px) -> each chain resampled every 2 px and smoothed with sigma sfrac*W -> tangent angle,
median-filtered over mfrac*W (0 = off) -> for every window of W px: straight if the path turning < `straight` deg; else corner if the
largest turn over a quarter window is >= `conc` of the net turning; else arc if net >= `netpath` x path; else jagged.
Score = arc points / (arc + corner points), pooled over all chains of the image."""
from __future__ import annotations

import io
from collections import OrderedDict

import numpy as np
from skimage import feature

import global_curv as gc

DEFAULTS = dict(sigma=2.0, qlo=0.85, qhi=0.95, minlen=60.0, W=96.0, sfrac=0.125, mfrac=0.5, straight=8.0, conc=0.6, netpath=0.8)
CODES = {"none": 0, "straight": 1, "arc": 2, "corner": 3, "jagged": 4}
_normals: "OrderedDict[str, np.ndarray]" = OrderedDict()   # id -> float16 (H, W, 3); a few kept
_chains: dict = {}                                          # id -> (canny key, edges, chains); last Canny setting per image
KEEP_NORMALS = 6


def add_normals(rid: str, npz_bytes) -> None:
    if hasattr(npz_bytes, "to_bytes"): npz_bytes = npz_bytes.to_bytes()      # a JS Uint8Array under Pyodide
    _normals[rid] = np.load(io.BytesIO(bytes(npz_bytes)))["n"]; _normals.move_to_end(rid)
    while len(_normals) > KEEP_NORMALS: _normals.popitem(last=False)


def has_normals(rid: str) -> bool:
    return rid in _normals


def has_chains(rid: str, p: dict) -> bool:
    return rid in _chains and _chains[rid][0] == _ckey(p)


def _ckey(p):
    return (float(p["sigma"]), float(p["qlo"]), float(p["qhi"]), float(p["minlen"]))


def edges(n: np.ndarray, sigma: float, qlo: float, qhi: float) -> np.ndarray:
    return np.logical_or.reduce([feature.canny(n[..., c], sigma=sigma, low_threshold=qlo, high_threshold=qhi, use_quantiles=True)
                                 for c in range(3)])


def trace_chains(edges: np.ndarray, min_len: float):
    """acp/global_curv.py: trace_chains with one change: each chain starts at its smallest (row, col) end point (or pixel, if closed)
    instead of the first one met in a Python set. Set order follows the hash width (64-bit CPython vs 32-bit Pyodide), so without this the
    browser walks some chains from the other end and 31 of the 191 rooms' scores differ from Python's in the 3rd decimal."""
    from scipy import ndimage
    from skimage import morphology
    sk = morphology.skeletonize(edges)
    nb = ndimage.convolve(sk.astype(int), np.ones((3, 3), int), mode="constant") - 1
    seg = sk & ~ndimage.binary_dilation(sk & (nb >= 3), structure=np.ones((3, 3), bool))
    lab, _ = ndimage.label(seg, structure=np.ones((3, 3)))
    nb_seg = ndimage.convolve(seg.astype(int), np.ones((3, 3), int), mode="constant") - 1
    out = []
    for _, (ys, xs) in ndimage.value_indices(lab, ignore_value=0).items():
        if len(ys) < min_len / 1.5: continue
        pset = {(int(y), int(x)) for y, x in zip(ys, xs)}
        ends = sorted((y, x) for (y, x) in pset if nb_seg[y, x] <= 1)
        cur = ends[0] if ends else min(pset); path = [cur]; seen = {cur}
        while True:
            y, x = cur
            cand = [(y + dy, x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy or dx) and (y + dy, x + dx) in pset and (y + dy, x + dx) not in seen]
            if not cand: break
            cand.sort(key=lambda c: abs(c[0] - y) + abs(c[1] - x)); cur = cand[0]; path.append(cur); seen.add(cur)
        if len(seen) < 0.9 * len(pset): continue
        pts = np.array(path, float)
        if np.linalg.norm(np.diff(pts, axis=0), axis=1).sum() >= min_len:
            out.append((pts, not ends and np.linalg.norm(pts[0] - pts[-1]) < 2.0))
    return out


def chains(rid: str, p: dict):
    k = _ckey(p)
    if rid not in _chains or _chains[rid][0] != k:
        E = edges(_normals[rid].astype(float), k[0], k[1], k[2])
        _chains[rid] = (k, E, trace_chains(E, min_len=k[3]))
    return _chains[rid][1], _chains[rid][2]


def labels(pts, closed, W, sfrac, mfrac, straight, conc, netpath):
    """Resampled points and one label code per point (as acp/union.py: chain_labels, thresholds exposed)."""
    p = gc.resample(pts, closed, sigma=W * sfrac)
    lab = np.zeros(len(p), np.uint8)
    if len(p) >= 8:
        phi = gc.tangent_angle(p, closed)
        if mfrac > 0: phi = gc.median_tangent(phi, closed, W * mfrac)
        net, path, cn = gc.window_stats(phi, closed, W)
        if len(net):
            nd, pdg = np.rad2deg(net), np.rad2deg(path)
            s = pdg < straight; turning = ~s & (nd >= straight)
            c = turning & (cn >= conc); a = turning & (cn < conc) & (net >= netpath * path); j = ~(s | c | a)
            h = int(round(W / (2 * gc.STEP))); cen = np.arange(len(p)) if closed else np.arange(h, len(p) - h)
            lab[cen] = np.select([c, a, s, j], [3, 2, 1, 4], 0)
    return p, lab


def _labelled(rid, p):
    E, ch = chains(rid, p)
    return E, [labels(pts, closed, float(p["W"]), float(p["sfrac"]), float(p["mfrac"]), float(p["straight"]), float(p["conc"]),
                      float(p["netpath"])) for pts, closed in ch]


def _summary(L):
    cnt = np.bincount(np.concatenate([lab for _, lab in L]) if L else np.zeros(0, np.uint8), minlength=5)
    a, c = int(cnt[2]), int(cnt[3])
    return dict(score=a / (a + c) if a + c else None, arc=a, corner=c, straight=int(cnt[1]), jagged=int(cnt[4]), chains=len(L))


def score(rid: str, p: dict) -> dict:
    p = {**DEFAULTS, **dict(p)}; _, L = _labelled(rid, p); return _summary(L)


def preview(rid: str, p: dict) -> dict:
    """Score plus what the page draws: edge pixels (uint8 per pixel), chain points (float32 row, col pairs), chain offsets, label codes."""
    p = {**DEFAULTS, **dict(p)}; E, L = _labelled(rid, p); out = _summary(L)
    pts = np.concatenate([q for q, _ in L]).astype(np.float32) if L else np.zeros((0, 2), np.float32)
    out.update(edges=E.astype(np.uint8).tobytes(), pts=pts.tobytes(), offsets=np.cumsum([0] + [len(q) for q, _ in L]).astype(np.int32).tobytes(),
               codes=(np.concatenate([lab for _, lab in L]) if L else np.zeros(0, np.uint8)).tobytes(),
               height=int(E.shape[0]), width=int(E.shape[1]))
    return out


def normals_rgba(rid: str) -> bytes:
    """Normal map as an RGBA image (x right -> red, y up -> green, z toward the viewer -> blue; (n + 1) / 2)."""
    n = _normals[rid].astype(np.float32); rgb = np.clip((n + 1) * 127.5, 0, 255).astype(np.uint8)
    return np.concatenate([rgb, np.full(rgb.shape[:2] + (1,), 255, np.uint8)], -1).tobytes()
