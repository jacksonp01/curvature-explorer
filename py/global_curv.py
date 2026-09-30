"""Candidate global curvature measures (see docs/research_log/PRESPEC_aac_photos.md).

Contour family: classify each arc-length window of a traced contour as straight / corner / arc / jagged
and summarise (fractions, large-scale mean curvature). Image family: orientation entropy and
orientation-field curvature, no contour tracing.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage import morphology

STEP = 2.0  # px, arc-length resampling
SIGMA = 6.0  # px, Gaussian smoothing along arc length (2.0 in PRESPEC; see addendum)
SCALES = (24, 48, 96)
MIN_CHAIN = 60.0  # px
STRAIGHT_DEG = 8.0
CONC_CORNER = 0.6
NETPATH_ARC = 0.8


def resample(points: np.ndarray, closed: bool, step: float = STEP, sigma: float = SIGMA) -> np.ndarray:
    pts = np.asarray(points, float)
    if closed and not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    n = max(int(s[-1] / step), 8)
    t = np.linspace(0, s[-1], n, endpoint=not closed)
    out = np.stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])], axis=1)
    mode = "wrap" if closed else "nearest"
    return np.stack([ndimage.gaussian_filter1d(out[:, i], sigma / step, mode=mode) for i in range(2)], axis=1)


def tangent_angle(p: np.ndarray, closed: bool) -> np.ndarray:
    """Unwrapped tangent angle at each sample (central differences)."""
    if closed:
        d = np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)
    else:
        d = np.gradient(p, axis=0)
    phi = np.arctan2(d[:, 0], d[:, 1])
    return np.unwrap(phi)


def window_stats(phi: np.ndarray, closed: bool, W: float, step: float = STEP):
    """Per window centre: net turning, path turning, concentration. Returns arrays (only valid centres)."""
    h = int(round(W / (2 * step)))
    q = max(int(round(W / (4 * step))), 1)
    n = len(phi)
    if closed and n <= 2 * h:
        return np.array([]), np.array([]), np.array([])  # window would wrap around the loop
    if closed:
        # unwrap across the seam by extending the array periodically in angle
        total = phi[-1] - phi[0]
        # tile 3 copies offset by the net winding so the seam is continuous
        wind = np.round(total / (2 * np.pi)) * 2 * np.pi
        big = np.concatenate([phi - wind, phi, phi + wind])
        lo = n
        centres = np.arange(n)
        arr = big
        base = lo
    else:
        arr = phi
        centres = np.arange(h, n - h)
        base = 0
    if len(centres) == 0:
        return np.array([]), np.array([]), np.array([])
    dphi = np.abs(np.diff(arr))
    cum = np.concatenate([[0], np.cumsum(dphi)])
    idx = centres + base
    net = np.abs(arr[idx + h] - arr[idx - h])
    path = cum[idx + h] - cum[idx - h]
    sub = np.abs(arr[q:] - arr[:-q])  # sub[j] = turn over [j, j+q]
    # max over j in [idx-h, idx+h-q]
    span = 2 * h - q + 1
    mx = ndimage.maximum_filter1d(sub, size=span, mode="nearest")
    # maximum_filter1d centres the window; index of centre of [idx-h, idx+h-q] is idx-h+(span-1)/2
    c = (idx - h + (span - 1) // 2).astype(int)
    conc = mx[np.clip(c, 0, len(mx) - 1)] / np.maximum(net, 1e-9)
    return net, path, conc


def classify(net, path, conc):
    net_deg = np.rad2deg(net)
    path_deg = np.rad2deg(path)
    straight = path_deg < STRAIGHT_DEG
    turning = ~straight & (net_deg >= STRAIGHT_DEG)
    corner = turning & (conc >= CONC_CORNER)
    arc = turning & (conc < CONC_CORNER) & (net >= NETPATH_ARC * path)
    jagged = ~(straight | corner | arc)
    return straight, corner, arc, jagged


def contour_measures(chains: list[tuple[np.ndarray, bool]], scales=SCALES, q: float = 1.0) -> dict:
    """chains: list of (points (N,2) in px, closed?). Returns C1..C5 at each scale. q: window weight = (contour length)^(q-1)
    (q = 1: every window counts equally, i.e. length-weighted contours; larger q emphasises long contours)."""
    out = {}
    for W in scales:
        counts = np.zeros(4)  # straight, corner, arc, jagged
        net_sum, n_win = 0.0, 0
        t_arc = t_cor = 0.0  # net turning summed over arc / corner windows (turning-weighted shares)
        for pts, closed in chains:
            p = resample(pts, closed)
            if len(p) < 8:
                continue
            phi = tangent_angle(p, closed)
            net, path, conc = window_stats(phi, closed, W)
            if len(net) == 0:
                continue
            s, c, a, j = classify(net, path, conc)
            wt = (len(p) * STEP) ** (q - 1.0)
            counts += wt * np.array([s.sum(), c.sum(), a.sum(), j.sum()])
            t_arc += wt * net[a].sum(); t_cor += wt * net[c].sum()
            net_sum += wt * net.sum() / W
            n_win += wt * len(net)
        if n_win == 0:
            vals = dict(C1=np.nan, C2=np.nan, C3=np.nan, AAC=np.nan, C5=np.nan)
        else:
            f = counts / n_win
            vals = dict(C1=f[2], C2=f[0], C3=f[1], AAC=(f[2] / (f[2] + f[1]) if f[1] + f[2] > 0 else np.nan), C5=net_sum / n_win)
        for k, v in vals.items():
            out[f"{k}_W{W}"] = float(v)
        out[f"T4_W{W}"] = float(t_arc / (t_arc + t_cor)) if (t_arc + t_cor) > 0 else np.nan
        out[f"jag_W{W}"] = float(counts[3] / n_win) if n_win else np.nan
        out[f"nwin_W{W}"] = n_win
    return out


def median_tangent(phi: np.ndarray, closed: bool, length: float, step: float = STEP) -> np.ndarray:
    """Running median of the unwrapped tangent angle over `length` px of arc (odd number of samples). Monotone sequences (straight runs,
    corners, circles, fillets) pass unchanged; oscillation shorter than the kernel is removed (PRESPEC_scaled_smoothing.md Addendum A)."""
    k = max(int(round(length / step)) | 1, 3)
    if closed:
        wind = np.round((phi[-1] - phi[0]) / (2 * np.pi)) * 2 * np.pi
        n = len(phi); big = np.concatenate([phi - wind, phi, phi + wind])
        return ndimage.median_filter(big, size=k, mode="nearest")[n:2 * n]
    return ndimage.median_filter(phi, size=k, mode="nearest")


def contour_measures_scaled(chains: list[tuple[np.ndarray, bool]], scales=SCALES, frac: float | None = 1 / 8, median: float | None = None) -> dict:
    """A/A+C-s / A/A+C-r (docs/research_log/PRESPEC_scaled_smoothing.md): as contour_measures (q = 1), but each contour is smoothed with
    sigma = frac * W for window W instead of the fixed SIGMA (frac None = SIGMA), so wobble finer than the window's scale is removed while
    an ideal sharp corner still meets the concentration test (sigma <= ~1.19 W/8); frac = 1/8 gives SIGMA at W = 48. median: if given,
    the tangent angle is also median-filtered over median * W px of arc (Addendum A uses 1/2)."""
    out = {}
    for W in scales:
        counts = np.zeros(4); n_win = 0
        for pts, closed in chains:
            p = resample(pts, closed, sigma=SIGMA if frac is None else frac * W)
            if len(p) < 8:
                continue
            phi = tangent_angle(p, closed)
            if median:
                phi = median_tangent(phi, closed, median * W)
            net, path, conc = window_stats(phi, closed, W)
            if len(net) == 0:
                continue
            s, c, a, j = classify(net, path, conc)
            counts += np.array([s.sum(), c.sum(), a.sum(), j.sum()]); n_win += len(net)
        f = counts / n_win if n_win else np.full(4, np.nan)
        out[f"AACs_W{W}"] = float(f[2] / (f[2] + f[1])) if n_win and (f[1] + f[2]) > 0 else np.nan
        out[f"jags_W{W}"] = float(f[3]) if n_win else np.nan
        out[f"nwins_W{W}"] = n_win
    return out


def trace_chains(edges: np.ndarray, min_len: float = MIN_CHAIN) -> list[tuple[np.ndarray, bool]]:
    """Ordered pixel chains of a binary edge map, split at junctions."""
    sk = morphology.skeletonize(edges)
    nb = ndimage.convolve(sk.astype(int), np.ones((3, 3), int), mode="constant") - 1
    junction = sk & (nb >= 3)
    seg = sk & ~ndimage.binary_dilation(junction, structure=np.ones((3, 3), bool))
    lab, n = ndimage.label(seg, structure=np.ones((3, 3)))
    nb_seg = ndimage.convolve(seg.astype(int), np.ones((3, 3), int), mode="constant") - 1
    chains = []
    coords = ndimage.value_indices(lab, ignore_value=0)
    for k, pix in coords.items():
        ys, xs = pix
        if len(ys) < min_len / 1.5:
            continue
        pset = {(int(y), int(x)) for y, x in zip(ys, xs)}
        ends = [(y, x) for (y, x) in pset if nb_seg[y, x] <= 1]
        closed = len(ends) == 0
        start = ends[0] if ends else next(iter(pset))
        path = [start]; seen = {start}
        cur = start
        while True:
            y, x = cur
            cand = [(y + dy, x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy or dx) and (y + dy, x + dx) in pset and (y + dy, x + dx) not in seen]
            if not cand:
                break
            cand.sort(key=lambda c: abs(c[0] - y) + abs(c[1] - x))  # prefer 4-neighbours
            cur = cand[0]; path.append(cur); seen.add(cur)
        if len(seen) < 0.9 * len(pset):
            continue  # branched/ambiguous component
        pts = np.array(path, float)
        L = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
        if L >= min_len:
            if closed and np.linalg.norm(pts[0] - pts[-1]) < 2.0:
                chains.append((pts, True))
            else:
                chains.append((pts, False))
    return chains


def image_measures(gray: np.ndarray) -> dict:
    gy = ndimage.gaussian_filter(gray, 2.0, order=(1, 0))
    gx = ndimage.gaussian_filter(gray, 2.0, order=(0, 1))
    mag = np.hypot(gx, gy)
    theta = np.mod(np.arctan2(gy, gx), np.pi)
    hist, _ = np.histogram(theta, bins=36, range=(0, np.pi), weights=mag)
    p = hist / hist.sum()
    p = p[p > 0]
    out = {"O1": float(-(p * np.log2(p)).sum())}
    for s in (8, 16):
        Jxx = ndimage.gaussian_filter(gx * gx, s)
        Jyy = ndimage.gaussian_filter(gy * gy, s)
        Jxy = ndimage.gaussian_filter(gx * gy, s)
        t = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy)
        vy, vx = np.cos(2 * t), np.sin(2 * t)
        dvy = np.gradient(vy); dvx = np.gradient(vx)
        jac = 0.5 * np.sqrt(dvy[0] ** 2 + dvy[1] ** 2 + dvx[0] ** 2 + dvx[1] ** 2)
        w = Jxx + Jyy
        out[f"O2_s{s}"] = float((w * jac).sum() / w.sum())
    return out


# ------------------------------------------------------------------ A/A+C-sag (PRESPEC_aac_sag.md)
SAG_Z = 3.0
SAG_MIN = 0.5   # px; set by the synthetic gate (scripts/aac_sag/gate.py), see PRESPEC_aac_sag.md


def _window_centres(n: int, closed: bool, W: float, step: float = STEP) -> np.ndarray:
    h = int(round(W / (2 * step)))
    return np.arange(n) if closed else np.arange(h, n - h)


def sag_test(xy: np.ndarray, z: float = SAG_Z, s_min: float = SAG_MIN):
    """Cubic fit in the chord frame (PRESPEC_aac_sag.md Addendum A). The stretch is curved iff the quadratic or the cubic
    coefficient is significant (|b|/SE >= z, SE inflated for serial correlation) and the fitted curve's largest
    deviation from its own chord (sag) is >= s_min px. Returns (is_curved, sag_px, max z)."""
    d = xy[-1] - xy[0]; l = np.linalg.norm(d)
    if l < 1e-6 or len(xy) < 8:
        return False, 0.0, 0.0
    u = d / l; v = np.array([-u[1], u[0]])
    x = ((xy - xy[0]) @ u - l / 2) / (l / 2); y = (xy - xy[0]) @ v          # x scaled to [-1, 1]
    X = np.column_stack([np.ones_like(x), x, x * x, x ** 3])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b; n = len(x)
    rho = np.corrcoef(res[:-1], res[1:])[0, 1] if n > 4 and res.std() > 1e-12 else 0.0
    rho = 0.0 if not np.isfinite(rho) else max(rho, 0.0)
    n_eff = max(n * (1 - rho) / (1 + rho), 5.0)
    s2 = (res @ res) / max(n - 4, 1) * n / n_eff
    try:
        cov = s2 * np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return False, 0.0, 0.0
    zz = max(abs(b[2]) / np.sqrt(cov[2, 2]) if cov[2, 2] > 0 else np.inf, abs(b[3]) / np.sqrt(cov[3, 3]) if cov[3, 3] > 0 else np.inf)
    xs = np.linspace(-1, 1, 101); f = b[0] + b[1] * xs + b[2] * xs ** 2 + b[3] * xs ** 3
    chord = f[0] + (f[-1] - f[0]) * (xs + 1) / 2
    sag = float(np.abs(f - chord).max())
    return bool(zz >= z and sag >= s_min), sag, float(zz)


def sag_labels(pts: np.ndarray, closed: bool, W: float, z: float = SAG_Z, s_min: float = SAG_MIN, log: list | None = None):
    """A/A+C window labels at scale W after sag-based relabelling of straight/arc runs.
    Returns (straight, corner, arc, jagged, net, n_promoted, n_demoted) or None if the chain has no windows."""
    p = resample(pts, closed); praw = resample(pts, closed, sigma=1e-3)
    if len(p) < 8:
        return None
    phi = tangent_angle(p, closed)
    net, path, conc = window_stats(phi, closed, W)
    if len(net) == 0:
        return None
    s, c, a, j = classify(net, path, conc)
    s, a = s.copy(), a.copy()
    cen = _window_centres(len(p), closed, W)
    # signed turning rate smoothed at sd W/2 -> inflection points
    if closed:
        wind = np.round((phi[-1] - phi[0]) / (2 * np.pi)) * 2 * np.pi
        dphi = np.diff(np.concatenate([phi, [phi[0] + wind]]))
    else:
        dphi = np.gradient(phi)
    k = ndimage.gaussian_filter1d(dphi, (W / 2) / STEP, mode="wrap" if closed else "nearest")
    sign = np.sign(k[cen])
    lab = np.where(s, 0, np.where(a, 1, -1))            # 0 straight, 1 arc, -1 corner/jagged (never changed)
    promoted = demoted = 0
    minlen = int(round(W / STEP)); m = len(cen); near = np.deg2rad(2 * STRAIGHT_DEG)

    def test(i, e, kind):
        nonlocal promoted, demoted
        lo, hi = cen[i], cen[e] + 1
        turn = float(np.ptp(phi[lo:min(hi, len(phi))]))
        res = sag_test(praw[lo:hi], z, s_min)
        if log is not None:
            log.append((kind, res[0], res[1], res[2], float(np.rad2deg(turn))))
        rs = slice(i, e + 1)
        if kind in (0, 2) and res[0]:                    # straight run or near-straight stretch that is curved -> arc
            promoted += int(s[rs].sum()); a[rs] |= s[rs]; s[rs] = False
        elif kind in (1, 2) and turn < near and not res[0]:   # near-straight arc windows that fail -> straight
            demoted += int(a[rs].sum()); s[rs] |= a[rs]; a[rs] = False

    def runs(mask_fn, i0, i1):
        i = i0
        while i <= i1:
            if not mask_fn(i):
                i += 1; continue
            e = i
            while e + 1 <= i1 and mask_fn(e + 1) and lab[e + 1] == lab[i] and cen[e + 1] == cen[e] + 1:
                e += 1
            yield i, e
            i = e + 1

    i = 0
    while i < m:                                          # maximal corner-free stretches
        if lab[i] < 0:
            i += 1; continue
        e = i
        while e + 1 < m and lab[e + 1] >= 0 and cen[e + 1] == cen[e] + 1:
            e += 1
        lo, hi = cen[i], cen[e] + 1
        if e - i + 1 >= minlen and np.ptp(phi[lo:min(hi, len(phi))]) < near:
            test(i, e, 2)                                 # whole stretch turns < 16 deg: one test (Addendum A)
        else:
            for ri, re_ in runs(lambda t: lab[t] >= 0, i, e):
                if re_ - ri + 1 >= minlen:
                    test(ri, re_, int(lab[ri]))
        i = e + 1
    return s, c, a, j, net, promoted, demoted


def contour_measures_sag(chains: list[tuple[np.ndarray, bool]], scales=SCALES, z: float = SAG_Z, s_min: float = SAG_MIN) -> dict:
    """A/A+C-sag per W (PRESPEC_aac_sag.md), plus the share of windows promoted to arc / demoted to straight."""
    out = {}
    for W in scales:
        counts = np.zeros(4); n_win = pro = dem = 0
        for pts, closed in chains:
            r = sag_labels(pts, closed, W, z, s_min)
            if r is None:
                continue
            s, c, a, j, net, pr, de = r
            counts += np.array([s.sum(), c.sum(), a.sum(), j.sum()]); n_win += len(net); pro += pr; dem += de
        f = counts / n_win if n_win else np.full(4, np.nan)
        out[f"AACS_W{W}"] = float(f[2] / (f[2] + f[1])) if n_win and (f[1] + f[2]) > 0 else np.nan
        out[f"C1S_W{W}"] = float(f[2]) if n_win else np.nan
        out[f"promoted_W{W}"] = pro / n_win if n_win else np.nan
        out[f"demoted_W{W}"] = dem / n_win if n_win else np.nan
    return out
