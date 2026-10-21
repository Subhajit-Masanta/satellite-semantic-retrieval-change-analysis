"""Multi-year change on the Indian sites (yearly composites 2019-2025).

    run.bat experiments\\run_series.py

There is no ground truth for these sites, so this script does NOT report
accuracy. It reports what the multi-year test removes compared with a plain
first-year vs last-year comparison, the first year each change is seen, and
the area per change type. Known construction dates are the sanity check.

Saves results/series_results.json, results/series_log.txt and figures.
"""

import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt               # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import change_types as CT            # noqa: E402
from src import time_series as TS             # noqa: E402

OUT = ROOT / "results"
FIG = OUT / "figs"
LOG = open(OUT / "series_log.txt", "w", encoding="utf-8")


def say(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True); LOG.write(line + "\n"); LOG.flush()


def rgb_of(arr):
    """(5,H,W) reflectance -> uint8 RGB for display (fixed scale, all years alike)."""
    x = np.clip(arr[[2, 1, 0]] / 3000.0, 0, 1) ** 0.8
    return (np.transpose(x, (1, 2, 0)) * 255).astype(np.uint8)


def analyse(site_dir):
    years, cube, valid = TS.load_series(site_dir)
    z, _ = TS.normalise(cube, valid)
    det = TS.detect(z)
    mb, ma = TS.before_after(cube, valid, det["first"], det["lasting"])
    i1 = CT.indices(mb[1], mb[2], mb[3], mb[4])
    i2 = CT.indices(ma[1], ma[2], ma[3], ma[4])
    types = CT.classify(i1, i2)
    types[~det["lasting"]] = 0
    return years, cube, det, types


def main():
    say("=" * 70); say("MULTI-YEAR CHANGE, INDIAN SITES"); say("=" * 70)
    R = {}
    k3 = np.ones((3, 3), np.uint8)
    for mfile in sorted((ROOT / "data" / "india").glob("*/meta.json")):
        meta = json.loads(mfile.read_text())
        key = meta["site"]
        years, cube, det, types = analyse(mfile.parent)
        two = cv2.morphologyEx(det["two_mask"].astype(np.uint8), cv2.MORPH_OPEN, k3) > 0
        last = cv2.morphologyEx(det["lasting"].astype(np.uint8), cv2.MORPH_OPEN, k3) > 0
        kept = float((two & last).sum()) / max(int(two.sum()), 1)
        fy = np.array(years)[det["first"][last]] if last.any() else np.array([])
        hist = {str(y): int((fy == y).sum()) for y in years}
        types[~last] = 0
        R[key] = {"name": meta["name"], "years": years,
                  "two_date_pct": round(100 * float(two.mean()), 2),
                  "lasting_pct": round(100 * float(last.mean()), 2),
                  "two_date_alarms_that_last_pct": round(100 * kept, 1),
                  "two_date_alarms_dropped_pct": round(100 * (1 - kept), 1),
                  "lasting_not_in_two_date_pct": round(100 * float((last & ~two).sum()) / max(int(last.sum()), 1), 1),
                  "first_year_pixels": hist,
                  "area_ha": CT.area_table(types),
                  "lasting_area_ha": round(float(last.sum()) * 0.01, 1)}
        top = max(hist, key=hist.get) if fy.size else "-"
        say(f"{key:22s} two-date {R[key]['two_date_pct']:5.2f}%  lasting {R[key]['lasting_pct']:5.2f}%  "
            f"two-date alarms dropped {R[key]['two_date_alarms_dropped_pct']:5.1f}%  "
            f"most common first year {top}")
        say(f"{'':22s} area (ha): {R[key]['area_ha']}")

        if key in ("jewar_airport_up", "navi_mumbai_airport", "vit_vellore"):
            fig = plt.figure(figsize=(14, 7.6), facecolor="white")
            for i, y in enumerate(years):
                a = fig.add_subplot(2, len(years), i + 1)
                a.imshow(rgb_of(cube[i])); a.set_title(str(y), fontsize=10); a.axis("off")
            base = rgb_of(cube[-1])
            p1 = base.copy(); p1[two] = (255, 40, 40)
            p2 = base.copy(); p2[last] = (255, 40, 40)
            fmap = np.where(last, np.array(years)[det["first"]], np.nan)
            tcol = (0.45 * base + 0.55 * CT.colour(types)).astype(np.uint8)
            tcol[types == 0] = base[types == 0]
            for j, (im, t) in enumerate(((p1, f"first vs last year: {R[key]['two_date_pct']}% flagged"),
                                         (p2, f"lasting change only: {R[key]['lasting_pct']}% flagged"),
                                         (None, "first year the change is seen"),
                                         (tcol, "change type (from indices)"))):
                a = fig.add_subplot(2, 4, 5 + j)
                if im is None:
                    a.imshow(base); h = a.imshow(fmap, cmap="viridis", vmin=years[2], vmax=years[-2])
                    fig.colorbar(h, ax=a, fraction=0.046, ticks=years[2:-1])
                else:
                    a.imshow(im)
                a.set_title(t, fontsize=10); a.axis("off")
            fig.suptitle(meta["name"], fontsize=12)
            plt.tight_layout(); fig.savefig(FIG / f"series_{key}.png", dpi=75); plt.close(fig)

    d = [v["two_date_alarms_dropped_pct"] for v in R.values()]
    R["_summary"] = {"mean_two_date_alarms_dropped_pct": round(float(np.mean(d)), 1),
                     "min": min(d), "max": max(d), "n_sites": len(d)}
    say(f"\nacross {len(d)} sites: on average {np.mean(d):.1f}% of the two-date alarms do not "
        f"last (range {min(d)} to {max(d)}%) and are dropped by the multi-year test")
    (OUT / "series_results.json").write_text(json.dumps(R, indent=2))
    say(f"Saved {OUT / 'series_results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
