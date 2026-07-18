"""
Tier 0 synthetic GT shapes for the Mask2PolyMin benchmark.

Step 0 (this file, in progress): hand-authored unit-scale family definitions,
reviewed one at a time via `render_preview`. Placement (scale/rotate/center),
rasterization, and the committed gt_shapes/ output land in step 1.
"""
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

PREVIEW_DIR = Path(__file__).parent / "shape_review"


def _finalize(raw) -> np.ndarray:
    """Center a hand-authored polygon by its bounding box and scale so the
    circumscribed radius (max vertex distance from center) is exactly 1.
    """
    verts = np.asarray(raw, dtype=np.float64)
    center = (verts.min(axis=0) + verts.max(axis=0)) / 2.0
    verts = verts - center
    radius = np.linalg.norm(verts, axis=1).max()
    return verts / radius


def _reflex_mask(verts: np.ndarray) -> np.ndarray:
    """True at each vertex whose interior turn is reflex (interior angle > 180 deg)."""
    nxt = np.roll(verts, -1, axis=0)
    prev = np.roll(verts, 1, axis=0)
    signed_area = 0.5 * np.sum(verts[:, 0] * nxt[:, 1] - nxt[:, 0] * verts[:, 1])
    orientation = np.sign(signed_area)
    edge_in = verts - prev
    edge_out = nxt - verts
    cross = edge_in[:, 0] * edge_out[:, 1] - edge_in[:, 1] * edge_out[:, 0]
    return np.sign(cross) != orientation


def _rect() -> np.ndarray:
    hw, hh = 1.8, 1.0  # elongation 1.8:1, not a square
    return _finalize([(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)])


def _lshape() -> np.ndarray:
    w, h = 0.66, 1.0    # narrower than tall: letter-like proportions, not a square
    t = 0.32            # uniform stroke thickness (was 0.4 on the old unit square; 4/5 as thick)
    raw = [
        (0, 0),
        (t, 0),
        (t, h - t),
        (w, h - t),
        (w, h),
        (0, h),
    ]
    return _finalize(raw)


def _hexagon() -> np.ndarray:
    angles = np.deg2rad([-90, -30, 30, 90, 150, 210])  # pointy top/bottom, flat left/right sides
    raw = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    return _finalize(raw)


def _star() -> np.ndarray:
    outer_r, inner_r = 1.0, 0.45  # inner/outer ratio 0.45 per the catalog
    k = np.arange(5)
    outer_ang = np.deg2rad(-90 + 72 * k)   # 5 outer tips, first pointing straight up
    inner_ang = outer_ang + np.deg2rad(36)  # 5 inner (reflex) notches, interleaved
    raw = np.empty((10, 2))
    raw[0::2, 0] = outer_r * np.cos(outer_ang)
    raw[0::2, 1] = outer_r * np.sin(outer_ang)
    raw[1::2, 0] = inner_r * np.cos(inner_ang)
    raw[1::2, 1] = inner_r * np.sin(inner_ang)
    return _finalize(raw)


def _tab() -> np.ndarray:
    # union of two rectangles, same proportions as examples/simple_bitmap_example.py:
    # body 40x60, a 10x20 bump on the right edge, offset 10 from the top.
    raw = [
        (0, 0),
        (40, 0),
        (40, 10),
        (50, 10),
        (50, 30),
        (40, 30),
        (40, 60),
        (0, 60),
    ]
    return _finalize(raw)


def _plane() -> np.ndarray:
    # top-view airliner silhouette, nose up. Traced from an icon (shape_review/plane.jpg):
    # thresholded, contour via cv2.findContours, fitted with Mask2PolyMin at tolerance 1.0,
    # rotated nose-up, mirror-symmetrized, unit-normalized. Min vertex spacing 0.077 unit
    # (engine bumps / nose arc), so this family's smallest legal size is d128, not d48.
    raw = [
        (-0.1344, -0.2943),
        (-0.2605, -0.2258),
        (-0.3117, -0.3030),
        (-0.4099, -0.2911),
        (-0.4246, -0.1323),
        (-0.9529, 0.1678),
        (-0.9550, 0.2965),
        (-0.1302, 0.0754),
        (-0.0909, 0.6571),
        (-0.3647, 0.8140),
        (-0.3669, 0.9135),
        (0.0000, 0.8162),
        (0.3669, 0.9135),
        (0.3647, 0.8140),
        (0.0909, 0.6571),
        (0.1302, 0.0754),
        (0.9550, 0.2965),
        (0.9529, 0.1678),
        (0.4246, -0.1323),
        (0.4099, -0.2911),
        (0.3117, -0.3030),
        (0.2605, -0.2258),
        (0.1344, -0.2943),
        (0.1359, -0.7526),
        (0.0387, -0.9337),
        (-0.0387, -0.9337),
        (-0.1359, -0.7526),
    ]
    return _finalize(raw)


def _house() -> np.ndarray:
    # pentagon body (gable roof, no overhang) with a centered door notch in the bottom edge
    half_w = 0.5          # wall half-width
    apex_y, eave_y, bottom_y = -0.55, 0.0, 0.7
    door_hw, door_top = 0.12, 0.4  # door half-width, door lintel height
    raw = [
        (0.0, apex_y),
        (half_w, eave_y),
        (half_w, bottom_y),
        (door_hw, bottom_y),
        (door_hw, door_top),   # reflex: door lintel, right
        (-door_hw, door_top),  # reflex: door lintel, left
        (-door_hw, bottom_y),
        (-half_w, bottom_y),
        (-half_w, eave_y),
    ]
    return _finalize(raw)


def _ship() -> np.ndarray:
    # side-view ship, bow to the right. Nothing axis-aligned except the cabin roofs:
    # raked stern, deck sheer rising toward the bow, trapezoid (leaning) cabin walls.
    raw = [
        (-0.90, -0.03),  # stern deck corner (raked stern)
        (-0.62, -0.05),  # lower cabin base, left (reflex)
        (-0.55, -0.32),  # lower cabin roof, left (wall leans inward)
        (-0.37, -0.32),  # step to upper cabin (reflex)
        (-0.32, -0.56),  # upper cabin roof, left
        (-0.10, -0.56),  # upper cabin roof, right
        (-0.05, -0.32),  # step down (reflex)
        (0.13, -0.32),   # lower cabin roof, right
        (0.20, -0.07),   # lower cabin base, right (reflex)
        (0.90, -0.14),   # bow tip
        (0.58, 0.35),    # hull bottom, bow side (raked)
        (-0.74, 0.35),   # hull bottom, stern side
    ]
    return _finalize(raw)


def _arrow() -> np.ndarray:
    # map direction-arrow pointing up: triangular head (~57 deg tip) on a straight
    # shaft, two reflex barbs where the head base meets the shaft.
    raw = [
        (0.00, -1.00),   # tip
        (0.38, -0.30),   # head base, right
        (0.16, -0.30),   # barb, right (reflex)
        (0.16, 0.95),    # shaft end, right
        (-0.16, 0.95),   # shaft end, left
        (-0.16, -0.30),  # barb, left (reflex)
        (-0.38, -0.30),  # head base, left
    ]
    return _finalize(raw)


def _car() -> np.ndarray:
    # rear-view car silhouette: flat roof, obtuse roof shoulders, straight sides,
    # two wheel bumps protruding below the floor. The wheel-to-body-corner gap is
    # deliberately small (min spacing ~0.18 unit): this family expects the d64
    # fallback per the catalog - the bumps must stay small enough to tempt the
    # fitter to drop them.
    raw = [
        (-0.46, -0.50),  # roof, left
        (0.46, -0.50),   # roof, right
        (0.62, -0.26),   # roof shoulder, right (obtuse)
        (0.62, -0.14),   # right mirror root, top (reflex)
        (0.75, -0.15),   # right mirror tip, top
        (0.76, -0.03),   # right mirror tip, bottom
        (0.62, -0.01),   # right mirror root, bottom (reflex)
        (0.62, 0.55),    # body bottom, right
        (0.48, 0.55),    # right wheel, outer top (reflex)
        (0.48, 0.75),    # right wheel, outer bottom
        (0.22, 0.75),    # right wheel, inner bottom
        (0.22, 0.55),    # right wheel, inner top (reflex)
        (-0.22, 0.55),   # left wheel, inner top (reflex)
        (-0.22, 0.75),   # left wheel, inner bottom
        (-0.48, 0.75),   # left wheel, outer bottom
        (-0.48, 0.55),   # left wheel, outer top (reflex)
        (-0.62, 0.55),   # body bottom, left
        (-0.62, -0.01),  # left mirror root, bottom (reflex)
        (-0.76, -0.03),  # left mirror tip, bottom
        (-0.75, -0.15),  # left mirror tip, top
        (-0.62, -0.14),  # left mirror root, top (reflex)
        (-0.62, -0.26),  # roof shoulder, left (obtuse)
    ]
    return _finalize(raw)


FAMILIES = {
    "rect": _rect,
    "lshape": _lshape,
    "hexagon": _hexagon,
    "star": _star,
    "tab": _tab,
    "plane": _plane,
    "house": _house,
    "ship": _ship,
    "arrow": _arrow,
    "car": _car,
}


def render_preview(family: str) -> Path:
    """Rasterize a unit-scale family at a fixed review canvas: filled mask,
    GT outline, and vertex dots (reflex corners marked separately).
    """
    verts = FAMILIES[family]()
    reflex = _reflex_mask(verts)

    canvas_r, margin = 250, 50
    size = 2 * (canvas_r + margin)
    c = size / 2.0
    px = verts * canvas_r + c
    closed_px = np.vstack([px, px[:1]])

    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(px).astype(np.int32).reshape(-1, 1, 2)], 1)

    dmat = np.linalg.norm(verts[:, None, :] - verts[None, :, :], axis=2)
    np.fill_diagonal(dmat, np.inf)
    min_spacing = float(dmat.min())
    bbox_unit = verts.max(axis=0) - verts.min(axis=0)
    bbox_px = bbox_unit * canvas_r

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(mask, cmap="gray", alpha=0.35)
    ax.plot(closed_px[:, 0], closed_px[:, 1], "r-", linewidth=1.5)
    ax.scatter(px[~reflex, 0], px[~reflex, 1], c="blue", s=40, zorder=3, label="corner")
    if reflex.any():
        ax.scatter(px[reflex, 0], px[reflex, 1], c="orange", s=70, marker="s", zorder=3, label="reflex")
    ax.set_title(f"{family}  ({len(verts)} corners, {int(reflex.sum())} reflex)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)

    PREVIEW_DIR.mkdir(exist_ok=True)
    out_path = PREVIEW_DIR / f"{family}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"{family}: {len(verts)} corners, {int(reflex.sum())} reflex, "
          f"min unit spacing {min_spacing:.3f} (need >= 0.19 @48px)")
    print(f"  bounding box: {bbox_unit[0]:.3f} x {bbox_unit[1]:.3f} unit "
          f"= {bbox_px[0]:.0f} x {bbox_px[1]:.0f} px on this {size}x{size} review canvas")
    return out_path


if __name__ == "__main__":
    import sys
    names = sys.argv[1:] or list(FAMILIES)
    for name in names:
        render_preview(name)
