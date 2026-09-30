"""Contour geometry for the A/A+C window classifier: a readable copy of what the page needs from the research acp/global_curv.py."""
import numpy as np
from scipy import ndimage

STEP = 2.0  # spacing of resampled contour points, in pixels


def resample(points, closed, sigma, step=STEP):
    """Resample a pixel chain every `step` px along its length, then smooth it with a Gaussian of `sigma` px."""
    pts = np.asarray(points, float)

    # Repeat the first point at the end of a loop so its last segment is included
    if closed and not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[:1]])

    # Distance along the chain at each input point
    segment_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc_length = np.concatenate([[0], np.cumsum(segment_lengths)])

    # Evenly spaced positions along the chain (at least 8 of them)
    n = max(int(arc_length[-1] / step), 8)
    positions = np.linspace(0, arc_length[-1], n, endpoint=not closed)
    rows = np.interp(positions, arc_length, pts[:, 0])
    cols = np.interp(positions, arc_length, pts[:, 1])

    # Smooth along the contour: loops wrap around, open chains repeat their end points
    mode = "wrap" if closed else "nearest"
    rows = ndimage.gaussian_filter1d(rows, sigma / step, mode=mode)
    cols = ndimage.gaussian_filter1d(cols, sigma / step, mode=mode)
    return np.stack([rows, cols], axis=1)


def tangent_angle(points, closed):
    """Direction of the contour at each point, in radians, unwrapped so it changes continuously."""
    if closed:
        # Central difference that wraps around the loop
        delta = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    else:
        delta = np.gradient(points, axis=0)
    angle = np.arctan2(delta[:, 0], delta[:, 1])
    return np.unwrap(angle)


def _extend_loop(angle):
    """Three copies of a loop's angles, shifted by its total winding so the seam is continuous."""
    winding = np.round((angle[-1] - angle[0]) / (2 * np.pi)) * 2 * np.pi
    return np.concatenate([angle - winding, angle, angle + winding])


def median_tangent(angle, closed, length, step=STEP):
    """Running median of the tangent angle over `length` px: removes wobble, keeps corners and steady bends."""
    size = max(int(round(length / step)) | 1, 3)  # odd number of samples, at least 3
    if not closed:
        return ndimage.median_filter(angle, size=size, mode="nearest")

    # Filter the middle copy of the extended loop so the seam is treated like any other point
    n = len(angle)
    filtered = ndimage.median_filter(_extend_loop(angle), size=size, mode="nearest")
    return filtered[n:2 * n]


def window_stats(angle, closed, W, step=STEP):
    """Net turning, path turning and turning concentration of every W px window (open chains: whole windows only)."""
    empty = (np.array([]), np.array([]), np.array([]))
    half = int(round(W / (2 * step)))              # samples from the window centre to either end
    quarter = max(int(round(W / (4 * step))), 1)  # samples in a quarter of the window
    n = len(angle)

    if closed:
        if n <= 2 * half:
            return empty  # the window would wrap all the way around the loop
        angle = _extend_loop(angle)
        centres = np.arange(n) + n  # every point of the middle copy
    else:
        centres = np.arange(half, n - half)
        if len(centres) == 0:
            return empty

    # Net turning: change in direction between the two ends of the window
    net = np.abs(angle[centres + half] - angle[centres - half])

    # Path turning: all turning inside the window, whichever way it goes
    cumulative_turn = np.concatenate([[0], np.cumsum(np.abs(np.diff(angle)))])
    path = cumulative_turn[centres + half] - cumulative_turn[centres - half]

    # Concentration: the largest turn over any quarter window inside the window, divided by the net turning
    quarter_turn = np.abs(angle[quarter:] - angle[:-quarter])
    span = 2 * half - quarter + 1  # number of quarter windows that fit in one window
    largest = ndimage.maximum_filter1d(quarter_turn, size=span, mode="nearest")
    middle = (centres - half + (span - 1) // 2).astype(int)  # index of the filter's centre for each window
    concentration = largest[np.clip(middle, 0, len(largest) - 1)] / np.maximum(net, 1e-9)

    return net, path, concentration
