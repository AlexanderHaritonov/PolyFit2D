"""
Detailed simple-vs-complex breakdown of summarized_csvs/defect_margin_ab_raw.csv (see
defect_margin_ab.py). Separates the pooled mean/median from defect_margin_ab.py's own summary
into: per-noise-level trend, paired per-contour win/tie/loss counts (does False actually help
or hurt on a contour-by-contour basis, or is the pooled mean driven by a few outliers), and a
mechanism sanity check (False must never produce MORE segments than True for the same contour --
it only ever stops the split loop earlier).
"""
import csv
import statistics
from collections import defaultdict
from pathlib import Path

RAW = Path(__file__).parent / "summarized_csvs" / "defect_margin_ab_raw.csv"

NUMERIC_FIELDS = ["n_segments", "corner_recall", "corner_precision", "corner_loc_err", "wall_time_ms"]


def load(path: Path) -> list[dict]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["noise_level"] = int(r["noise_level"])
        r["apply_local_defect_margin"] = r["apply_local_defect_margin"] == "True"
        r["n_segments"] = int(r["n_segments"])
        for field in NUMERIC_FIELDS[1:]:
            r[field] = float(r[field])
    return rows


def pair_by_contour(rows: list[dict]) -> dict[str, tuple[dict, dict]]:
    """contour_id -> (True-variant row, False-variant row), only where both exist."""
    by_id: dict[str, dict] = defaultdict(dict)
    for r in rows:
        by_id[r["contour_id"]][r["apply_local_defect_margin"]] = r
    return {cid: (v[True], v[False]) for cid, v in by_id.items() if True in v and False in v}


def win_tie_loss(pairs: list[tuple[dict, dict]], metric: str, lower_is_better: bool) -> tuple[int, int, int]:
    """Count contours where the False variant is better / tied / worse than True on `metric`."""
    win = tie = loss = 0
    for t, f in pairs:
        tv, fv = t[metric], f[metric]
        better = (fv < tv) if lower_is_better else (fv > tv)
        worse = (fv > tv) if lower_is_better else (fv < tv)
        win += better
        loss += worse
        tie += not (better or worse)
    return win, tie, loss


def report(shape_class: str, rows: list[dict], pairs: list[tuple[dict, dict]]) -> None:
    print(f"\n{'=' * 78}\n{shape_class.upper()} shapes -- {len(pairs)} contours\n{'=' * 78}")

    t_rows = [r for r in rows if r["apply_local_defect_margin"] is True]
    f_rows = [r for r in rows if r["apply_local_defect_margin"] is False]
    print(f"{'metric':<18}{'True mean':>11}{'True med':>10}{'False mean':>12}{'False med':>11}{'mean delta':>12}")
    for metric in NUMERIC_FIELDS:
        tm, tmed = statistics.fmean(r[metric] for r in t_rows), statistics.median(r[metric] for r in t_rows)
        fm, fmed = statistics.fmean(r[metric] for r in f_rows), statistics.median(r[metric] for r in f_rows)
        pct = (fm - tm) / tm * 100 if tm else float("nan")
        print(f"{metric:<18}{tm:>11.4f}{tmed:>10.4f}{fm:>12.4f}{fmed:>11.4f}{pct:>+11.1f}%")

    print(f"\n  Per noise level (mean recall, mean wall_time_ms):")
    print(f"  {'level':<7}{'n':>5}{'recall_T':>10}{'recall_F':>10}{'d_recall':>10}{'time_T':>9}{'time_F':>9}{'d_time':>9}")
    for level in sorted({r["noise_level"] for r in rows}):
        tl = [r for r in t_rows if r["noise_level"] == level]
        fl = [r for r in f_rows if r["noise_level"] == level]
        if not tl or not fl:
            continue
        rt, rf = statistics.fmean(r["corner_recall"] for r in tl), statistics.fmean(r["corner_recall"] for r in fl)
        tt, tf = statistics.fmean(r["wall_time_ms"] for r in tl), statistics.fmean(r["wall_time_ms"] for r in fl)
        print(f"  {level:<7}{len(tl):>5}{rt:>10.4f}{rf:>10.4f}{rf - rt:>+10.4f}{tt:>9.2f}{tf:>9.2f}{(tf - tt) / tt * 100:>+8.1f}%")

    w, tie, l = win_tie_loss(pairs, "corner_recall", lower_is_better=False)
    print(f"\n  Paired recall:     False better={w:>4}  tie={tie:>4}  False worse={l:>4}  (of {len(pairs)})")
    w, tie, l = win_tie_loss(pairs, "wall_time_ms", lower_is_better=True)
    print(f"  Paired wall_time:  False faster={w:>4}  tie={tie:>4}  False slower={l:>4}  (of {len(pairs)})")

    more_segs = sum(1 for t, f in pairs if f["n_segments"] > t["n_segments"])
    fewer_segs = sum(1 for t, f in pairs if f["n_segments"] < t["n_segments"])
    print(f"  Sanity (mechanism): False has MORE segments than True in {more_segs} contours "
          f"(expected ~0 -- False only ever stops earlier); fewer in {fewer_segs}.")


def main() -> None:
    rows = load(RAW)
    all_pairs = pair_by_contour(rows)
    for shape_class in ("complex", "simple"):
        class_rows = [r for r in rows if r["shape_class"] == shape_class]
        class_pairs = [(t, f) for t, f in all_pairs.values() if t["shape_class"] == shape_class]
        report(shape_class, class_rows, class_pairs)


if __name__ == "__main__":
    main()
