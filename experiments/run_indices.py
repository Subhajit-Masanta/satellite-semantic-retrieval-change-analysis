"""Does the spectral-index change score find OSCD's labelled (urban) change?

    run.bat experiments\\run_indices.py

OSCD labels are mostly new buildings and roads, so the right test for our
'built-up gained' score is: does it overlap the ground truth? Thresholds are
picked on the 14 TRAIN cities and reported on the 10 TEST cities.

Saves results/indices_results.json and results/indices_log.txt.
"""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

os.environ["RUN_LOG"] = "indices_log.txt"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import run_oscd as X                       # noqa: E402
from src import change_types as CT         # noqa: E402
from src import metrics as M               # noqa: E402

say, OUT = X.say, X.OUT
R = {}


def bands(city, date, lab_shape):
    cdir = X.find_dir(city)
    h, w = lab_shape
    out = {}
    for name, b in (("green", "B03"), ("red", "B04"), ("nir", "B08"), ("swir", "B11")):
        a = X.read_band(cdir, date, b)
        if a.shape[:2] != (h, w):
            a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
        out[name] = a[:h, :w]
    return out


def score(masks, cities, names):
    per = [M.change_metrics(masks[c], cities[c]["label"]) for c in names]
    r = X.micro(per)
    neg = sum((cities[c]["label"] == 0).sum() for c in names)
    r["fpr_pct"] = round(100 * sum(p["fp"] for p in per) / neg, 3)
    return r


say("=" * 70); say("SPECTRAL INDEX CHANGE ON OSCD"); say("=" * 70)
cities = {c: X.load_city(c) for c in X.TRAIN + X.TEST}
I1, I2 = {}, {}
for c in cities:
    shp = cities[c]["label"].shape
    I1[c] = CT.indices(**bands(c, 1, shp))
    I2[c] = CT.indices(**bands(c, 2, shp))

k3 = np.ones((3, 3), np.uint8)
scores = {
    "NDBI up (built-up index)": lambda c: CT.robust_z(CT.centred_delta(I1[c]["ndbi"], I2[c]["ndbi"])),
    "NDVI down (vegetation lost)": lambda c: -CT.robust_z(CT.centred_delta(I1[c]["ndvi"], I2[c]["ndvi"])),
    "Built score (NDBI up and NDVI down)": lambda c: CT.built_score(I1[c], I2[c]),
    "Any index moved (max of the three)": lambda c: np.max(np.abs(np.stack([
        CT.robust_z(CT.centred_delta(I1[c][k], I2[c][k])) for k in ("ndvi", "ndwi", "ndbi")])), 0),
}
say("\n[1] Index change scores, top p% of pixels per image (p picked on TRAIN)")
R["scores"] = {}
best_maps = {}
for name, fn in scores.items():
    S = {c: fn(c) for c in cities}
    pcs = [90, 93, 95, 97, 98]
    f = [score({c: cv2.morphologyEx(((S[c] >= np.percentile(S[c], p)) * 255).astype(np.uint8),
                                    cv2.MORPH_OPEN, k3) for c in X.TRAIN}, cities, X.TRAIN)["f1"]
         for p in pcs]
    pb = pcs[int(np.argmax(f))]
    masks = {c: cv2.morphologyEx(((S[c] >= np.percentile(S[c], pb)) * 255).astype(np.uint8),
                                 cv2.MORPH_OPEN, k3) for c in cities}
    te, tr = score(masks, cities, X.TEST), score(masks, cities, X.TRAIN)
    R["scores"][name] = {"top_percent": 100 - pb, "test": te, "train": tr}
    best_maps[name] = (S, masks)
    say(f"   {name:38s} top {100-pb:2d}%  TEST P={te['precision']:.3f} R={te['recall']:.3f} "
        f"F1={te['f1']:.3f} IoU={te['iou']:.3f} FPR={te['fpr_pct']:.2f}%  (train F1={tr['f1']:.3f})")
pick = max(R["scores"], key=lambda k: R["scores"][k]["train"]["f1"])
R["picked_on_train"] = pick
say(f"   best on TRAIN: {pick}")
say("   for reference on the same test cities: CVA 3 bands F1 0.353, ours 13 bands F1 0.338")

# combine with the 13-band magnitude mask from the earlier run, if it is there
m13 = OUT / "masks_cva13.npz"
if m13.exists():
    Z = np.load(m13)
    comb_or = {c: cv2.bitwise_or(Z[c], best_maps[pick][1][c]) for c in X.TEST}
    comb_and = {c: cv2.bitwise_and(Z[c], best_maps[pick][1][c]) for c in X.TEST}
    R["with_13band"] = {"either": score(comb_or, cities, X.TEST),
                        "both": score(comb_and, cities, X.TEST)}
    for k, v in R["with_13band"].items():
        say(f"   13-band mask and index mask, {k:6s}: P={v['precision']:.3f} R={v['recall']:.3f} "
            f"F1={v['f1']:.3f} FPR={v['fpr_pct']:.2f}%")

say("\n[2] Which type do we give to the pixels OSCD marks as changed? (TEST cities)")
tot = {k: 0 for k in CT.TYPES}
out_of_gt = {k: 0 for k in CT.TYPES}
for c in X.TEST:
    t = CT.classify(I1[c], I2[c])
    gt = cities[c]["label"] > 0
    for k in CT.TYPES:
        tot[k] += int(((t == k) & gt).sum())
        out_of_gt[k] += int(((t == k) & ~gt).sum())
n_gt = sum(tot.values())
R["types_inside_ground_truth"] = {CT.TYPES[k][0]: round(100 * tot[k] / n_gt, 2) for k in CT.TYPES}
n_bg = sum(out_of_gt.values())
R["types_outside_ground_truth"] = {CT.TYPES[k][0]: round(100 * out_of_gt[k] / n_bg, 2) for k in CT.TYPES}
say(f"   {'type':34s} {'inside labelled change':>24s} {'on unchanged pixels':>22s}")
for k in CT.TYPES:
    say(f"   {CT.TYPES[k][0]:34s} {100*tot[k]/n_gt:22.1f}%  {100*out_of_gt[k]/n_bg:20.1f}%")
typed = {c: ((CT.classify(I1[c], I2[c]) == 1) * 255).astype(np.uint8) for c in X.TEST}
R["type1_as_detector"] = score(typed, cities, X.TEST)
v = R["type1_as_detector"]
say(f"   'built-up gained' label used as a detector: P={v['precision']:.3f} R={v['recall']:.3f} "
    f"F1={v['f1']:.3f} FPR={v['fpr_pct']:.2f}%")

(OUT / "indices_results.json").write_text(json.dumps(R, indent=2))
say(f"\nSaved {OUT / 'indices_results.json'}")
