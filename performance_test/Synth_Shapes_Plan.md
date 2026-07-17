# Tier 0 Shape Generator — Plan (`synth_shapes.py`)

Detail plan for step 3 of [Perf_Test_Plan.md](Perf_Test_Plan.md): the synthetic GT shapes,
their storage, and the distortion pipeline. Review gates are marked ⏸.

## Shape catalog — 10 families

| Family | Corners | Reflex | Why it is in the set |
|---|---|---|---|
| `rect` | 4 | – | 90° corners, elongation 1.8:1 (not just a square) |
| `lshape` | 6 | 1 | 90° corners plus one 270° concavity |
| `hexagon` | 6 | – | obtuse 120° corners — hardest to localize precisely |
| `star` (5-point) | 10 | 5 | acute ~45° tips — blur erodes them hardest; inner/outer radius ratio 0.45 |
| `tab` | 8 | 2 | the union-of-two-rectangles shape from [simple_bitmap_example.py](../examples/simple_bitmap_example.py), proportions 50×60 |
| `plane` | ~14 | yes | stylized aircraft pictogram (top view): pointed nose, swept wings, tailplane |
| `house` | 9 | 2 | stylized house pictogram: gable roof + door notch in the bottom edge |
| `ship` | ~11 | yes | stylized ship pictogram: hull with pointed bow + stepped superstructure |
| `arrow` | 7 | 2 | map direction-arrow (shaft + triangular head): acute tip, two reflex barbs where head meets shaft; strongly orientation-dependent |
| `car` | ~14 | 4 | stylized car rear silhouette: obtuse roof shoulders + two wheel bumps below the body — small sub-features that tempt the fitter to drop them; expect the 64 px fallback |

Pictogram vertex coordinates are hand-authored constants in the module, defined at unit
scale (circumscribed radius 1), mirror-symmetric where natural. Exact proportions are
judged at the shape-review gate, not in this document.

**Corner-spacing constraint:** `corner_metrics` matches nearest-neighbour, so GT corners
must stay ≥ 2·τ = 4 px apart (enforced at ≥ 4.5 px). At the smallest size (⌀ 48 px,
radius 24) this means ≥ 0.19 unit spacing between any two vertices — pictograms must
stay coarse (no chimneys, masts, or thin funnels). The generator asserts this for every
(shape, size). Fallback if a pictogram cannot meet it at ⌀ 48 without looking wrong:
raise that family's small size to 64 px rather than distort the design.

## Dataset axes

| Axis | Values | Count |
|---|---|---|
| family | see catalog | 10 |
| size (circumscribed ⌀) | 48, 128, 320 px | 3 |
| rotation | 0°, 10°, 22.5°, 37°, 45° | 5 |
| **base GT shapes** | | **150** |
| noise level | 0 (clean), 1, 2 | 3 |
| seeds per noisy level | 3 (level 0 deterministic → 1) | |
| **distorted contours** | 150 × (1 + 2×3) | **1050** |

0° and 45° are the rasterization special cases (axis-aligned is RDP's best case and must
be included); 10°/22.5°/37° are generic angles where staircase aliasing is worst.
Benchmark volume: 1050 contours × 2 algorithms × 5 tolerances = 10 500 rows.
Per aggregation cell: 150 contours at level 0, 450 at levels 1–2.

## Storage — canonical GT files; the pipeline starts by loading them

The 30 JSON files are the canonical GT source: written once by the generator, approved
at the shape-review gate, and **committed to git** (small, text, diffable — a file in
gitignored `data/` would pin nothing). `dataset()` starts by loading them, applies the
5 rotation angles to `polygon_xy` (about the canvas centre), and rasterizes every mask
in memory (`cv2.fillPoly` with fixed-point `shift`).

The canonical format is the polygon, not the mask: rotation is applied to the polygon
**before** subpixel rasterization — never to a stored mask, which would corrupt the GT
with resampling artifacts. That is why the PNGs are derived artifacts: committed alongside so anyone can review
the shapes straight from the repo (in a PR or file browser) without running any code,
but never read by the pipeline. Each mask sits on a
per-shape square canvas of `size + 2 × 24 px` margin so distortion never touches the
border.

Step 1 writes 63 files to `performance_test/gt_shapes/` (committed, unlike the
regenerable `data/`): 30 PNG+JSON pairs (60 files) — one per family × size, **0°
rotation only** — for close-up review, plus 3 gallery contact sheets on which all 150
instances (including the 120 rotated variants, which get no individual files) appear as
thumbnails:

```
performance_test/gt_shapes/
  d048/
    rect_d048_a0.png   # GT mask, 0/255 (for IDE viewability; pipeline masks are {0, 1})
    rect_d048_a0.json  # GT description, schema below
    ...                # 10 pairs per folder: one per family, 0° only
  d128/ ...
  d320/ ...
  gallery_d048.png     # contact sheet: 10 families × 5 angles,
  gallery_d128.png     #   mask + GT polygon (red) + corner dots
  gallery_d320.png
```

JSON schema — self-contained GT description; the PNG is derived from `polygon_xy`:

```jsonc
{
  "shape_id": "house_d128_a0",         // matches the PNG basename
  "family": "house",
  "size_px": 128,                      // circumscribed diameter (folder axis)
  "angle_deg": 0.0,
  "canvas_hw": [176, 176],             // size + 2 × 24 margin
  "polygon_xy": [[x, y], ...],         // closed (first == last), float canvas pixel coords
  "corners_xy": [[x, y], ...],         // GT corners for corner_metrics (= vertices without closing dup)
  "params": {"margin_px": 24, "rect_aspect": 1.8, "star_inner_ratio": 0.45}
}
```

Deliberately absent: noise parameters (these files are pure GT) and contours (extracted
at benchmark time from distorted masks). Distorted masks are not stored either — they
are regenerated deterministically from per-record seeds (derived from shape index +
noise level + rep), so any benchmark row can be reproduced in isolation.

Regenerating the GT files is an explicit action (`synth_shapes.py --write-gt`), never a
side effect of a benchmark run. Floats are serialized at full precision with stable key
order, so regenerating unchanged shapes is byte-identical; any git diff under
`gt_shapes/` means the GT changed and existing results are stale.

## Distortion pipeline (step 2, after shape review)

Order: elastic jitter → blur + re-threshold → speckle. Parameters per level:

| Level | Blur σ (px) | Jitter amp (px) | Speckle p | Reads as |
|---|---|---|---|---|
| 0 | – | – | – | clean rasterization (staircase only) |
| 1 | 1.0 | 1.0 | – | decent segmentation net |
| 2 | 2.0 | 2.5 | 0.06 | sloppy segmentation net |

- *Elastic jitter*: smooth random displacement field (Gaussian-smoothed white noise,
  correlation length ~8 px) applied via `cv2.remap` — correlated boundary wobble, not
  salt-and-pepper.
- *Blur + re-threshold at 0.5* — rounds corners; the effect the benchmark is about.
- *Speckle*: pixel flips restricted to a ±2 px boundary band, then 3×3 close/open to
  make the noise blobby; level 2 only.

Exact numbers are tuned at the noise-review gate. Contours are extracted from the
distorted mask with `cv2.findContours` (`RETR_EXTERNAL`, `CHAIN_APPROX_NONE`, largest
contour by area) — integer pixel coordinates, exactly what a real user of the library
feeds the fitter. Tier 0 keeps cv2 extraction; skimage subpixel contours remain Tier 1.

## Build steps
0. create the 10 main shapes one by one and get it reviewed. store them as files.

1. **GT shapes**: unit-scale family definitions, place (scale/rotate/center), rasterize,
   write the 30 stored pairs and the 3 gallery sheets per the layout above.
   ⏸ **Review gate: shapes** — veto proportions, star sharpness, pictogram designs.
2. **Distortion + dataset iterator**: `distort(mask, level, rng)`, `extract_contour`,
   and a `dataset(reps=3)` generator (loads the canonical JSONs, applies rotations,
   rasterizes in memory; yields one record per (shape, level, rep)) as the single
   enumeration point for `run_benchmark.py`.
   Render `preview_noise.png`: one shape per family × 3 levels, GT outline + extracted
   contour overlaid. ⏸ **Review gate: noise levels** — level 2 must read as "bad
   segmentation net", not "absurd".
3. Add `performance_test/data/` and `performance_test/results/` to `.gitignore`
   (`performance_test/gt_shapes/` stays committed); mark step 3 progress in
   [Perf_Test_Plan.md](Perf_Test_Plan.md).
