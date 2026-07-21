# COLLINEAR_DIRECTIONS_MIN_DOT / LOCAL_DEFECT_MARGIN sweep — summary and decision

Follow-up to [Fitter_Improvements_Plan.md](Fitter_Improvements_Plan.md) items 1-3, after the
benchmark moved from a fixed-tolerance sweep to noise-matched tolerances
(`run_benchmark.matched_pair`). Looked for separate sweet spots across three slices: all
shapes, simple shapes (arrow, hexagon, house, lshape, rect, star, tab), and complex shapes
(car, plane, ship).

## Method

Two 1D sweeps, not a joint grid: fix `min_dot`, sweep `margin` to find each slice's best
value; then fix each slice's own best margin, sweep `min_dot` to find its best value.
Ranked by **mean** corner recall — median proved flat/uninformative when pooled over
hundreds of contours. Full 1950-contour Tier 0 dataset, plus a follow-up restricted to
`noise_level=4` only (the noisiest tier) to check the finding held under heavy noise, not
just as a pooled average.

Scripts: `collinear_margin_sweep.py`, `collinear_margin_sweep_extend.py`,
`margin_sweep_noise4.py`, plus `plot_collinear_margin_sweep.py` /
`plot_min_dot_tradeoff.py` for the charts. This doc stands in for them once they're
removed.

## Findings

### `LOCAL_DEFECT_MARGIN`

`margin=0.0` is the best value in every slice, both pooled across all noise levels and at
`noise_level=4` alone — recall decreases monotonically as margin rises from 0 toward 9.0
(the original, pre-investigation value), with no recall/precision trade-off (precision is
flat-to-better at margin=0, not worse).

At `margin=0.0`, `_has_severe_local_defect`'s threshold is 0, so it returns `True`
whenever any point in a segment with more than 3 points has nonzero squared deviation
from its fitted line — true for virtually every real (non-degenerate) segment. In effect,
margin=0 means the "no improvement" stop is *always* deferred. The conditional this
parameter gates is vestigial at its own optimal value.

### `COLLINEAR_DIRECTIONS_MIN_DOT`

Recall keeps climbing toward `min_dot=1.0` ("never merge collinear segments") in every
slice, but precision falls sharply approaching that extreme — steeply for simple shapes
(0.434 → 0.397), mildly for complex ones (0.494 → 0.483). `min_dot=0.9` (~25.8° of
allowed divergence, vs. the original 0.98's ~11.5°) captures nearly all of the recall gain
the 1.0 extreme offers, at essentially none of the precision cost. Already applied in
`fit_to_points_sequence.py`.

## Decision (revised)

The recall-only analysis above initially motivated removing `LOCAL_DEFECT_MARGIN` and
`_has_severe_local_defect` entirely (this was implemented briefly). Two follow-up checks
changed the call:

- **Wall time.** Deferring the no-improvement stop unconditionally costs real time at
  every noise level -- worst at the noisiest tiers (noise=4: 84.1ms -> 105.2ms median).
  This wasn't part of the original recall-focused sweep.
- **Simple vs. complex shapes measured separately, with wall time included this time**
  (`margin_4_5_comparison_raw.csv`, `margin_intermediate_raw.csv`,
  `fig_margin_full_tradeoff.png`): the package's main target is simple shapes, and there
  the margin=0 endpoint is not a clean win -- recall improves, but precision gets *worse*
  (0.433 vs. 0.445 at margin=4.5) and wall time gets worse too (70.7ms vs. 53.6ms). Complex
  shapes (car/plane/ship) do improve cleanly on every axis as margin drops toward 0, but
  they're the secondary target.
- The margin/recall/precision/wall-time relationship is **not monotonic for simple shapes**
  in the middle of the range -- both precision and wall time hit their *worst* point
  around margin=1.0 (worse than either margin=0 or a larger margin), so an intermediate
  value can't be picked by interpolating between the endpoints; it has to be read off the
  actual curve.

**Final decision: keep `LOCAL_DEFECT_MARGIN`, set to 2.5** (not removed, not 4.5, not 0).
At margin=2.5, simple shapes get *better* precision (0.439) *and* better wall time (69.2ms)
than the margin-removed state, for a small (~2.6%) recall giveback; complex shapes keep a
real (if partial) share of the recall/precision gain over the old margin=4.5, plus a
meaningful wall-time improvement (85.0ms -> 73.7ms). See `fig_margin_full_tradeoff.png`
for the full curve across every value measured (0, 0.5, 1.0, 1.5, 2.5, 3.5, 4.5).

**Implemented.** `LOCAL_DEFECT_MARGIN = 2.5` and `_has_severe_local_defect` are back in
`fit_to_points_sequence.py`; `_fit()`'s no-improvement branch defers to a severe local
defect exactly as before the removal. Full test suite passes (113/113).
