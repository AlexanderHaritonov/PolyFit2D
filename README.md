# Mask2PolyMin
Geometry processing package to turn noisy raster segmentation masks into clean, optimized, closed polygons.

Given a dense polygon, fit a polyline with a minimal number of segments.

Useful for post‑processing bitmask segmentation outputs from models such as MaskRCNN or YOLO‑Seg, especially when regular or low‑complexity shapes are required:
- constructing larger structures by matching edges
- obtaining simple geometric representations

Unlike common point‑thinning algorithms (Ramer–Douglas–Peucker, Visvalingam–Whyatt, Zhang–Suen), this method:
- minimizes segment count while preserving the raw shape
- does not shrink the area or remove corners
- reconstructs corners with sub-pixel accuracy: vertices are intersections of least-squares fitted lines.

Built for man-made or artificial objects that consist of straight edges, sharp corners, and regular geometric properties.

## Example

python -m venv .venv && source .venv/bin/activate && pip install -r requirements-examples.txt && python example_usage.py

- The input is a dense bitmask produced by a segmentation model.

![input bitmask](step1_bitmap.png)
- A contour is extracted from the bitmask using skimage.measure:

contour = measure.find_contours(bitmap, level=0.5)[0]

![extracted contour](step2_contour.png)

- Mask2PolyMin fits a minimal‑segment polyline to this contour:

segments = FitterToPointsSequence(contour, is_closed=True).fit()

![fitted segments](step3_fitted_segments.png)

- `segments_to_polyline(segments, is_closed=True)` converts the fitted segments to a closed polygon of float (sub-pixel) vertices, ready for GeoJSON/SVG/COCO export:

polygon = segments_to_polyline(segments, is_closed=True)

## Performance
The implementation is optimized, uses NumPy broadcasting.

## future work
- performance tests and comparison
- support for fitting segments using quadratic curves (parabolas) or B‑splines ?
- add possiblity to fit multiple polygons in parallel - on different cores ?


## Running Tests

```bash
.venv/bin/pytest test/
```

Tests run headless by default (no plot windows). To show plots during a test run:

```bash
SHOW_PLOTS=1 .venv/bin/pytest test/
```

In VSCode, right-click the `test/` folder → **Run Tests**. Plots are shown automatically (configured via `.vscode/settings.json`).

Install dev dependencies first if needed: `pip install -r requirements-dev.txt`



