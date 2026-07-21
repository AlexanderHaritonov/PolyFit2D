"""
A/B comparison of FitterConfig.rank_split_by_max_deviation, backing
Fitter_Improvements_Plan.md item 3.

Runs FitterToPointsSequence directly (not the baselines.mask2polymin wrapper, which has no
way to pass a non-default FitterConfig) across the full Tier 0 dataset, once per flag
setting, at each record's noise-matched tolerance (run_benchmark.matched_pair -- the same
tolerance a real caller would use for that noise level, per the current benchmark
methodology).

Naively, the flag should only matter when max_segments_count is hit before every eligible
segment gets resolved (otherwise every eligible segment is split eventually regardless of
order) -- that's the plan's own framing, and its "known risk" is specifically about that
tight-budget case. In practice the flag changes *something* (n_segments or the polygon
itself) far more often than that: `_fit()`'s early exits (sse_per_segment < tolerance_sq,
and the no-improvement/severe-local-defect check) depend on the SSE trajectory across
splits, which split order affects even without ever hitting the cap. So this script checks
three things, not just aggregate medians (which the plan warns may mask a minority-of-cells
effect):
  1. how often the flag changes the output at all, and how that splits between
     pinned-at-cap and not (the plan's predicted risk zone vs. the rest)
  2. the full per-contour recall/precision delta distribution, not just the median -- a
     regression concentrated in a minority of contours could wash out in a pooled median
  3. per-family recall, to catch a family-concentrated effect (steps 1 and 2 both turned
     out to be about one specific family; worth checking here too)

Run from performance_test/:  python split_ranking_ab.py
"""
from collections import defaultdict

import numpy as np

from mask2polymin import FitterConfig, FitterToPointsSequence
from run_benchmark import matched_pair
from synth_shapes import dataset
from metrics import corner_metrics

MAX_SEGMENTS = FitterConfig().max_segments_count


def fit(contour, tolerance, rank_by_max_deviation):
    config = FitterConfig(tolerance=tolerance, rank_split_by_max_deviation=rank_by_max_deviation)
    polygon, segments = FitterToPointsSequence(contour, is_closed=True, config=config).fit()
    return polygon, len(segments)


def main():
    records = list(dataset())
    print(f"{len(records)} contours, each at its noise-matched tolerance\n")

    by_noise = defaultdict(lambda: defaultdict(list))     # noise_level -> flag -> [(recall, precision, n_segments)]
    by_family = defaultdict(lambda: defaultdict(list))    # family -> flag -> [recall, ...]
    deltas = []            # (contour_id, family, pinned, d_recall, d_precision)
    changed = 0
    changed_and_pinned = 0
    pinned_total = 0

    for r in records:
        _, tol = matched_pair(r["noise_level"])
        poly_off, n_off = fit(r["contour_xy"], tol, False)
        poly_on, n_on = fit(r["contour_xy"], tol, True)

        rec_off, prec_off, _ = corner_metrics(r["gt_corners_xy"], poly_off)
        rec_on, prec_on, _ = corner_metrics(r["gt_corners_xy"], poly_on)
        by_noise[r["noise_level"]][False].append((rec_off, prec_off, n_off))
        by_noise[r["noise_level"]][True].append((rec_on, prec_on, n_on))
        by_family[r["family"]][False].append(rec_off)
        by_family[r["family"]][True].append(rec_on)

        pinned = n_off >= MAX_SEGMENTS or n_on >= MAX_SEGMENTS
        differs = n_off != n_on or not np.array_equal(poly_off, poly_on)
        if pinned:
            pinned_total += 1
        if differs:
            changed += 1
            if pinned:
                changed_and_pinned += 1
        deltas.append((r["contour_id"], r["family"], pinned, rec_on - rec_off, prec_on - prec_off))

    print(f"=== contours where the flag changed the output at all: {changed}/{len(records)} "
          f"({changed / len(records):.1%}) ===")
    print(f"of those, {changed_and_pinned}/{changed} were pinned at the segment cap "
          f"(n_segments >= {MAX_SEGMENTS} in either run)")
    print(f"({pinned_total}/{len(records)} contours were pinned at the cap at all)\n")

    print("=== per-noise-level medians, flag off vs on ===\n")
    print(f"{'noise':>6}  {'n_segments':>15}  {'recall':>15}  {'precision':>15}")
    for level in sorted(by_noise):
        off = np.array(by_noise[level][False])
        on = np.array(by_noise[level][True])
        segs = f"{np.median(off[:, 2]):.1f}->{np.median(on[:, 2]):.1f}"
        rec = f"{np.median(off[:, 0]):.4f}->{np.median(on[:, 0]):.4f}"
        prec = f"{np.median(off[:, 1]):.4f}->{np.median(on[:, 1]):.4f}"
        print(f"{level:>6}  {segs:>15}  {rec:>15}  {prec:>15}")

    d_recall = np.array([d[3] for d in deltas])
    d_precision = np.array([d[4] for d in deltas])
    print(f"\n=== per-contour recall delta (on - off): n={len(deltas)} ===")
    print(f"worse: {int((d_recall < 0).sum())}  same: {int((d_recall == 0).sum())}  "
          f"better: {int((d_recall > 0).sum())}")
    print(f"mean={d_recall.mean():.5f}  min={d_recall.min():.4f}  max={d_recall.max():.4f}  "
          f"sum={d_recall.sum():.2f}")

    print(f"\n=== per-contour precision delta (on - off) ===")
    print(f"worse: {int((d_precision < 0).sum())}  same: {int((d_precision == 0).sum())}  "
          f"better: {int((d_precision > 0).sum())}")
    print(f"mean={d_precision.mean():.5f}  min={d_precision.min():.4f}  max={d_precision.max():.4f}  "
          f"sum={d_precision.sum():.2f}")

    print("\n=== worst 10 recall regressions ===")
    order = np.argsort(d_recall)
    for i in order[:10]:
        cid, fam, pinned, dr, dp = deltas[i]
        if dr >= 0:
            break
        print(f"  {cid:<28} family={fam:<8} pinned={pinned!s:<5} d_recall={dr:+.4f} d_precision={dp:+.4f}")

    print("\n=== pinned-at-cap subset vs rest: mean recall delta ===")
    pinned_deltas = [d[3] for d in deltas if d[2]]
    unpinned_deltas = [d[3] for d in deltas if not d[2]]
    print(f"pinned   (n={len(pinned_deltas):>4}): mean d_recall={np.mean(pinned_deltas):+.5f}")
    print(f"unpinned (n={len(unpinned_deltas):>4}): mean d_recall={np.mean(unpinned_deltas):+.5f}")

    print(f"\n=== per-family recall, flag off vs on ===\n")
    families = sorted(by_family)
    for fam in families:
        off = np.median(by_family[fam][False])
        on = np.median(by_family[fam][True])
        marker = "  <-- CHANGED" if off != on else ""
        print(f"{fam:<10} {off:.4f} -> {on:.4f}{marker}")

    COMPLEX = {"car", "ship", "plane"}
    print(f"\n=== per-contour recall delta, by family (complex = {sorted(COMPLEX)}) ===\n")
    print(f"{'family':<10}{'complex':<9}{'n':>5}  {'worse':>6}  {'same':>6}  {'better':>7}  "
          f"{'mean':>9}  {'sum':>8}")
    by_fam_deltas = defaultdict(list)
    for cid, fam, pinned, dr, dp in deltas:
        by_fam_deltas[fam].append(dr)
    for fam in families:
        arr = np.array(by_fam_deltas[fam])
        worse, same, better = int((arr < 0).sum()), int((arr == 0).sum()), int((arr > 0).sum())
        print(f"{fam:<10}{str(fam in COMPLEX):<9}{len(arr):>5}  {worse:>6}  {same:>6}  "
              f"{better:>7}  {arr.mean():>9.5f}  {arr.sum():>8.3f}")

    complex_arr = np.array([dr for _, fam, _, dr, _ in deltas if fam in COMPLEX])
    simple_arr = np.array([dr for _, fam, _, dr, _ in deltas if fam not in COMPLEX])
    print(f"\n{'complex':<10}{'':<9}{len(complex_arr):>5}  "
          f"{int((complex_arr < 0).sum()):>6}  {int((complex_arr == 0).sum()):>6}  "
          f"{int((complex_arr > 0).sum()):>7}  {complex_arr.mean():>9.5f}  {complex_arr.sum():>8.3f}")
    print(f"{'simple':<10}{'':<9}{len(simple_arr):>5}  "
          f"{int((simple_arr < 0).sum()):>6}  {int((simple_arr == 0).sum()):>6}  "
          f"{int((simple_arr > 0).sum()):>7}  {simple_arr.mean():>9.5f}  {simple_arr.sum():>8.3f}")


if __name__ == "__main__":
    main()
