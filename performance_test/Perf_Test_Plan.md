# PolyFit2D — Performance & Quality Benchmark Plan

## Goal

Produce evidence — not rhetoric — for the two claims currently in [../README.md](../README.md):

1. PolyFit2D yields fewer segments than common simplifiers at equivalent fidelity.
2. PolyFit2D preserves area and corners (no shrinkage, no rounding) better than common simplifiers at equivalent segment count.

Secondary goal: characterize wall-time vs. contour length against the practical baseline (`cv2.approxPolyDP`).

The output of this work is a benchmark harness, a [performance_test/](.) folder of result CSVs and plots, and a short report section appended to the README.

## Status

- **Environment.** No venv. Using system Python 3.12 directly. Already installed: `opencv-python` 4.14.0-pre, `scipy` 1.11.4, `scikit-image` 0.22.0, `matplotlib` 3.6.3, `pycocotools`, `numpy` 1.26.4.
- **Tolerance semantics resolved.** `FitterConfig` originally had a single `tolerance` field doing triple duty across three different formulas (per-segment MSE, weighted-SSE global aggregate, no-improvement delta). It has now been split into three knobs (`segment_tolerance`, `global_tolerance`, `min_split_improvement`) and the public API converted to **linear pixels with RMS-perpendicular-distance semantics**, squared once internally. See [../src/fit_to_points_sequence.py](../src/fit_to_points_sequence.py).
- **Tolerance ↔ ε mapping derived.** For smooth arc-shaped residuals (cosine model), `ε_rdp ≈ √2 · segment_tolerance`. Used to align native-tolerance sweeps below.

## Metrics

Three primary metrics, each answering a distinct question. Skip Fréchet (redundant with Hausdorff for closed contours, slow to compute).

| Metric | Question it answers | How to compute |
|---|---|---|
| **Segment count** | Representation cost | `len(segments)` |
| **Hausdorff distance** (symmetric, in pixels) | Worst-case fidelity | `scipy.spatial.distance.directed_hausdorff` both directions, take max |
| **IoU** (rasterized polygon vs. original mask) | Area / corner preservation | rasterize fitted polygon → bitmap, compare with input mask |
| Mean distance (point → polyline) | Average fidelity (supplementary) | per-point min distance, mean |
| Wall time per contour | Speed | `time.perf_counter()` around the fit call |

Rasterization for IoU: `skimage.draw.polygon(rr, cc, shape=mask.shape)` or `cv2.fillPoly`.

## Baselines

### Required: OpenCV `cv2.approxPolyDP` (RDP)

Already installed (system Python). Usage on a contour produced by `skimage.measure.find_contours`:

```python
import cv2
import numpy as np

# skimage gives (row, col) float; cv2 wants (x, y) int32 with shape (N, 1, 2)
pts = np.flip(contour, axis=1).astype(np.float32)
pts_cv = pts.reshape(-1, 1, 2).astype(np.int32)
approx = cv2.approxPolyDP(pts_cv, epsilon=tolerance, closed=True)
approx_xy = approx.reshape(-1, 2)  # (M, 2) in (x, y)
```

`epsilon` is the Hausdorff tolerance in pixels. `closed=True` for closed contours.

### Optional: Imai–Iri (exact min-# under Hausdorff)

No trustworthy pip package known. If we want the optimality-gap number, write it ourselves — the DAG-shortest-path version is short:

1. For each pair `(i, j)` with `i < j` (or wrapping for closed contours), check if the chord from point i to point j approximates the subarc `i..j` within ε. This is an O(n³) edge-feasibility scan; acceptable for benchmark contours of a few hundred points.
2. BFS from vertex 0 to vertex n-1 over feasible edges → fewest-segments path.
3. For closed contours: pick a starting vertex, run BFS, repeat for several start vertices and take the min, or solve the cyclic version directly.

Defer this until after the RDP comparison is done. Only invest the day if the RDP results are promising and we want a "near-optimal" claim.

## Tolerance alignment

Two algorithms, two native tolerances. We use the **native-tolerance / post-hoc-metric** convention: each algorithm runs on its own tolerance, then we measure the same downstream metrics (Hausdorff, IoU, mean distance) and plot in shared metric space.

The native tolerances are aligned via the cosine-residual model `ε_rdp ≈ √2 · segment_tolerance` to start the sweeps in roughly comparable regimes:

| RDP `epsilon` (px) | PolyFit2D `tolerance` ≈ ε / √2 (px) |
|---|---|
| 0.5 | 0.35 |
| 1.0 | 0.71 |
| 1.5 | 1.06 |
| 2.0 | 1.41 |
| 3.0 | 2.12 |
| 5.0 | 3.54 |
| 8.0 | 5.66 |

For the first pass, set `segment_tolerance = global_tolerance = min_split_improvement = tolerance` (legacy single-knob behavior). Studying the three knobs as orthogonal axes is a separate experiment.

## Datasets

Use real segmentation masks, not synthetic. Three options, in order of effort:

### Tier 1 — quick start: COCO val2017 instance masks
- Free, no registration. Download: `http://images.cocodataset.org/zips/val2017.zip` (~1 GB) and `http://images.cocodataset.org/annotations/annotations_trainval2017.zip`.
- Pull masks: `coco.annToMask(ann)` returns a binary mask. Run `skimage.measure.find_contours` on it to get the input contour for the fitter.
- Filter: keep masks with ≥ 200 contour points (otherwise the simplification problem is trivial), single connected component, no holes for the first pass.
- Sample size: 1000 masks across diverse categories is plenty for stable curves.

### Tier 2 — published benchmark comparability: Cityscapes
- Requires free registration at cityscapes-dataset.com.
- Instance polygons in `gtFine/*_polygons.json`. Use these directly (already polygons) as the *ground truth*, rasterize to masks, extract contour, refit, compare back to ground-truth polygons.
- Useful only if we want numbers comparable to Polygon-RNN++ / Curve-GCN papers. Skip for the first benchmark pass.

### Tier 3 — stress shapes: ADE20K
- Free download. Broader category set; useful for finding pathological contours (very thin, many concavities). Skip unless Tier 1 results raise questions.

Recommendation: start with **Tier 1 (COCO val2017, 1000 masks)**. Add Tier 2 only if publishing.

## Implementation outline

Code lives in this folder ([performance_test/](.)). Add to [../.gitignore](../.gitignore): `performance_test/data/`, `performance_test/results/`.

```
performance_test/
  Perf_Test_Plan.md        # this file
  fetch_coco.py            # download + cache COCO val2017 if missing
  extract_contours.py      # iterate masks, find_contours, filter, save .npz
  run_benchmark.py         # for each contour, sweep tolerance, run polyfit2d + cv2 + (optional) imai_iri
  metrics.py               # hausdorff, iou_rasterized, mean_distance
  baselines.py             # wrappers: rdp_opencv(contour, eps), polyfit2d(contour, tol), imai_iri(contour, eps)
  plot_results.py          # produce the figures listed below
  data/                    # (gitignored) COCO download + extracted contours
  results/
    raw.csv                # one row per (contour_id, algorithm, tolerance)
    summary.csv            # aggregated
    *.png                  # plots
```

Sweep schedule (linear pixels):
- **RDP `epsilon`**: `{0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0}`
- **PolyFit2D `tolerance`**: `{0.35, 0.71, 1.06, 1.41, 2.12, 3.54, 5.66}`

Each algorithm runs once per (contour, tolerance). Output row schema:

```
contour_id, n_input_points, algorithm, tolerance,
n_segments, hausdorff, iou, mean_dist, wall_time_ms
```

Aggregate (mean, median, p95) across `contour_id` for each `(algorithm, tolerance)` pair → `summary.csv`.

## Build order

1. **`metrics.py`** — `hausdorff`, `iou_rasterized`, `mean_distance`. Verify on synthetic shapes (square, noisy circle).
2. **`baselines.py`** — `rdp_opencv(contour, eps)` and `polyfit2d(contour, tol)` wrappers returning a uniform `(M, 2)` polyline.
3. **Smoke test** — one synthetic noisy-circle contour, both algorithms at one tolerance each, print the metrics. Sanity check: at tolerance → 0, IoU should be ≥ 0.99 for both (this is the rasterization noise floor; differences smaller than this are not meaningful).
4. **`fetch_coco.py` + `extract_contours.py`** — download once, cache contours as `.npz`.
5. **`run_benchmark.py`** — full sweep on cached contours.
6. **`plot_results.py`** — three figures + summary table.

## Plots in the report

Three figures, each with PolyFit2D and OpenCV-RDP curves (and Imai–Iri if implemented):

1. **Segments vs. Hausdorff** — x: Hausdorff (px), y: median segment count. Shaded band: p25–p75. Lower-left is better.
2. **Segments vs. IoU** — x: median segment count, y: median IoU. Upper-left is better. This is the headline figure for the area-preservation claim.
3. **Wall time vs. contour length** — x: input-contour length (binned), y: median wall time per contour. Single tolerance (e.g. ε = 2.0). Two curves.

Plus a small table of aggregate numbers at one canonical tolerance (ε = 2.0 px / `tolerance` ≈ 1.41 px):

| algorithm | median #segs | median IoU | median Hausdorff | median ms/contour |
|---|---|---|---|---|

## Report structure

Append a section to [../README.md](../README.md) titled "Benchmarks":

- One paragraph: dataset (COCO val2017, N contours), tolerance sweep, hardware.
- The three plots inline.
- The table.
- One paragraph of honest interpretation: where PolyFit2D wins, where it doesn't, what contour regimes (length, complexity) favor each method.
- Link to [performance_test/](.) so results are reproducible.

If results are good, this is also the basis for a blog post / arXiv note. If results are mixed, it tells us where to focus algorithmic work before packaging.

## Open questions to resolve before starting

- **Closed-contour handling in `cv2.approxPolyDP`**: confirm `closed=True` does what we expect on the wrapped index range.
- **IoU rasterization stability**: at very low tolerance, rasterizing a polygon and the original mask can disagree by ±1 pixel along the boundary even for the identity transformation. Establish a noise floor by running PolyFit2D and OpenCV at tolerance → 0 and reporting the IoU we get (should be > 0.99); use it to discount differences smaller than that.
- **Tolerance-mapping check**: the `ε ≈ √2 · tolerance` mapping comes from a smooth-arc model. If empirically the curves don't overlay (one algorithm runs at a much tighter effective ε than the other across the sweep), shift PolyFit2D's tolerance schedule by a small factor and rerun.

## Effort estimate

- Tier 1 dataset + harness + RDP comparison: 1–2 days.
- Plots and report: half a day.
- Imai–Iri implementation + integration: 1 day, only if motivated by Tier 1 results.
