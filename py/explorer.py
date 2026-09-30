"""The N curvature measure (Marigold normals -> Canny edges -> A/A+C windows), run in the browser by Pyodide and checked in CPython."""
import io
from collections import OrderedDict

import numpy as np
from scipy import ndimage
from skimage import feature, morphology

import global_curv as gc

# Default settings (the project's N measure: r +0.667 with curvedness ratings over the 191 rooms)
DEFAULTS = dict(
    sigma=2.0,      # Canny: Gaussian blur before the gradient (px)
    qlo=0.85,       # Canny: low hysteresis threshold, as a quantile of the gradient magnitude
    qhi=0.95,       # Canny: high hysteresis threshold, as a quantile of the gradient magnitude
    minlen=60.0,    # shortest traced chain kept (px)
    W=96.0,         # A/A+C: window length along the contour (px)
    sfrac=0.125,    # A/A+C: contour smoothing sigma, as a fraction of W
    mfrac=0.5,      # A/A+C: median filter on the tangent angle, as a fraction of W (0 = off)
    straight=8.0,   # A/A+C: a window turning less than this (degrees) is straight
    conc=0.6,       # A/A+C: a window is a corner if one quarter of it holds this share of its net turning
    netpath=0.8,    # A/A+C: otherwise it is an arc if its net turning is at least this share of its path turning
)

# Label codes sent to the page for drawing
NONE, STRAIGHT, ARC, CORNER, JAGGED = 0, 1, 2, 3, 4

MAX_NORMAL_MAPS = 6        # normal maps kept in memory per worker (each is a few MB)
_normal_maps = OrderedDict()  # room id -> float16 normal map (height, width, 3), least recently used first
_chain_cache = {}             # room id -> (Canny settings, edge map, chains) for the last Canny settings used


# ------------------------------------------------------------------ normal maps
def add_normals(room_id, npz_bytes):
    """Store a room's normal map from the bytes of its .npz file."""
    if hasattr(npz_bytes, "to_bytes"):
        npz_bytes = npz_bytes.to_bytes()  # a JavaScript Uint8Array when running under Pyodide
    _normal_maps[room_id] = np.load(io.BytesIO(bytes(npz_bytes)))["n"]
    _normal_maps.move_to_end(room_id)
    while len(_normal_maps) > MAX_NORMAL_MAPS:
        _normal_maps.popitem(last=False)  # forget the least recently used map


def has_normals(room_id):
    return room_id in _normal_maps


def normals_rgba(room_id):
    """The normal map as RGBA bytes for display: x (right) -> red, y (up) -> green, z (toward the viewer) -> blue."""
    normals = _normal_maps[room_id].astype(np.float32)
    rgb = np.clip((normals + 1) * 127.5, 0, 255).astype(np.uint8)
    alpha = np.full(rgb.shape[:2] + (1,), 255, np.uint8)
    return np.concatenate([rgb, alpha], axis=-1).tobytes()


# ------------------------------------------------------------------ step 1: Canny edges on the normal map
def edges(normals, sigma, qlo, qhi):
    """Canny on each of the three normal components; a pixel is an edge if any component has one there."""
    edge_map = np.zeros(normals.shape[:2], bool)
    for component in range(3):
        edge_map |= feature.canny(normals[..., component], sigma=sigma, low_threshold=qlo, high_threshold=qhi, use_quantiles=True)
    return edge_map


# ------------------------------------------------------------------ step 2: trace edges into chains
def trace_chains(edge_map, min_len):
    """Thin the edges to 1 px, cut them at junctions and trace each piece into an ordered chain of pixels."""
    skeleton = morphology.skeletonize(edge_map)
    box = np.ones((3, 3), int)

    # Remove junction pixels (3+ neighbours) and their neighbours so every piece is a simple curve
    neighbours = ndimage.convolve(skeleton.astype(int), box, mode="constant") - 1
    junctions = skeleton & (neighbours >= 3)
    pieces = skeleton & ~ndimage.binary_dilation(junctions, structure=np.ones((3, 3), bool))
    labels, _ = ndimage.label(pieces, structure=box)
    piece_neighbours = ndimage.convolve(pieces.astype(int), box, mode="constant") - 1

    chains = []
    for rows, cols in ndimage.value_indices(labels, ignore_value=0).values():
        if len(rows) < min_len / 1.5:
            continue  # too few pixels to reach the minimum length
        pixels = {(int(r), int(c)) for r, c in zip(rows, cols)}

        # Start at an end (a pixel with one neighbour); take the smallest so every platform picks the same one
        ends = sorted(p for p in pixels if piece_neighbours[p] <= 1)
        current = ends[0] if ends else min(pixels)
        path, seen = [current], {current}

        # Walk to unvisited neighbours, preferring side neighbours over diagonal ones
        while True:
            r, c = current
            steps = [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                     if (dr or dc) and (r + dr, c + dc) in pixels and (r + dr, c + dc) not in seen]
            if not steps:
                break
            steps.sort(key=lambda p: abs(p[0] - r) + abs(p[1] - c))
            current = steps[0]
            path.append(current)
            seen.add(current)

        if len(seen) < 0.9 * len(pixels):
            continue  # the walk missed part of the piece: it was branched, so skip it
        points = np.array(path, float)
        if np.linalg.norm(np.diff(points, axis=0), axis=1).sum() >= min_len:
            is_loop = not ends and np.linalg.norm(points[0] - points[-1]) < 2.0
            chains.append((points, is_loop))
    return chains


def _canny_key(settings):
    return (float(settings["sigma"]), float(settings["qlo"]), float(settings["qhi"]), float(settings["minlen"]))


def has_chains(room_id, settings):
    """True if this room's chains for these Canny settings are already computed (then no normal map is needed)."""
    return room_id in _chain_cache and _chain_cache[room_id][0] == _canny_key(settings)


def chains(room_id, settings):
    """Edge map and chains of a room, recomputed only when a Canny setting changes."""
    key = _canny_key(settings)
    if not has_chains(room_id, settings):
        sigma, qlo, qhi, min_len = key
        edge_map = edges(_normal_maps[room_id].astype(float), sigma, qlo, qhi)
        _chain_cache[room_id] = (key, edge_map, trace_chains(edge_map, min_len))
    _, edge_map, room_chains = _chain_cache[room_id]
    return edge_map, room_chains


# ------------------------------------------------------------------ step 3: label every window of every chain
def window_labels(points, closed, W, sfrac, mfrac, straight, conc, netpath):
    """Resampled points of one chain and a label per point (the label of the W px window centred on it)."""
    resampled = gc.resample(points, closed, sigma=W * sfrac)
    labels = np.full(len(resampled), NONE, np.uint8)
    if len(resampled) < 8:
        return resampled, labels

    angle = gc.tangent_angle(resampled, closed)
    if mfrac > 0:
        angle = gc.median_tangent(angle, closed, W * mfrac)
    net, path, concentration = gc.window_stats(angle, closed, W)
    if len(net) == 0:
        return resampled, labels

    is_straight = np.rad2deg(path) < straight
    is_turning = ~is_straight & (np.rad2deg(net) >= straight)
    is_corner = is_turning & (concentration >= conc)
    is_arc = is_turning & (concentration < conc) & (net >= netpath * path)
    is_jagged = ~(is_straight | is_corner | is_arc)

    # Open chains have no window centred within W/2 of their ends
    half = int(round(W / (2 * gc.STEP)))
    centres = np.arange(len(resampled)) if closed else np.arange(half, len(resampled) - half)
    labels[centres] = np.select([is_corner, is_arc, is_straight, is_jagged], [CORNER, ARC, STRAIGHT, JAGGED], NONE)
    return resampled, labels


def _labelled_chains(room_id, settings):
    edge_map, room_chains = chains(room_id, settings)
    window = [float(settings[k]) for k in ("W", "sfrac", "mfrac", "straight", "conc", "netpath")]
    labelled = [window_labels(points, closed, *window) for points, closed in room_chains]
    return edge_map, labelled


# ------------------------------------------------------------------ step 4: score = arc / (arc + corner)
def _summary(labelled):
    all_labels = np.concatenate([labels for _, labels in labelled]) if labelled else np.zeros(0, np.uint8)
    counts = np.bincount(all_labels, minlength=5)
    arc, corner = int(counts[ARC]), int(counts[CORNER])
    return dict(
        score=arc / (arc + corner) if arc + corner else None,  # None: no arc or corner points
        arc=arc, corner=corner, straight=int(counts[STRAIGHT]), jagged=int(counts[JAGGED]), chains=len(labelled),
    )


def score(room_id, settings):
    """Score and label counts of one room; missing settings take their defaults."""
    settings = {**DEFAULTS, **dict(settings)}
    _, labelled = _labelled_chains(room_id, settings)
    return _summary(labelled)


def preview(room_id, settings):
    """Score plus what the page draws: edge pixels, chain points (row, col as float32), chain start offsets and label codes."""
    settings = {**DEFAULTS, **dict(settings)}
    edge_map, labelled = _labelled_chains(room_id, settings)
    result = _summary(labelled)

    points = np.concatenate([p for p, _ in labelled]) if labelled else np.zeros((0, 2))
    labels = np.concatenate([lab for _, lab in labelled]) if labelled else np.zeros(0, np.uint8)
    offsets = np.cumsum([0] + [len(p) for p, _ in labelled])  # chain i is points[offsets[i]:offsets[i + 1]]
    result.update(
        edges=edge_map.astype(np.uint8).tobytes(),
        pts=points.astype(np.float32).tobytes(),
        offsets=offsets.astype(np.int32).tobytes(),
        codes=labels.astype(np.uint8).tobytes(),
        height=int(edge_map.shape[0]),
        width=int(edge_map.shape[1]),
    )
    return result
