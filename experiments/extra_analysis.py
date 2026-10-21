"""Follow-up analysis on OSCD. Run AFTER run_oscd.py.

    python experiments/extra_analysis.py

A. Does embedding distance find LARGE changes better than small ones?
B. Gate C at MATCHED thresholds (suppression on/off, same threshold).
C. Tile-size ablation: 128 px tiles instead of 64 px.

Writes results/extra_results.json and results/extra_log.txt.
"""

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

os.environ["RUN_LOG"] = "extra_log.txt"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import run_oscd as X                         # noqa: E402
from src import change_detect as cd          # noqa: E402
from src import metrics as M                 # noqa: E402
from src.embedder import Embedder            # noqa: E402

say, OUT = X.say, X.OUT
E = {}

say("=" * 70)
say("OSCD FOLLOW-UP ANALYSIS")
say("=" * 70)
cities = {c: X.load_city(c) for c in X.TRAIN + X.TEST}

# ---------------------------------------------------------------- A
say("\n[A] Embedding distance vs size of the change (64 px tiles)")
S = np.load(OUT / "tile_scores.npy")
names = json.loads((OUT / "results.json").read_text())["tile_change"]["score_order"]
frac = []
for c, d in cities.items():
    h, w = d["label"].shape
    for (y, x) in X.tile_grid(h, w):
        frac.append(float((d["label"][y:y + X.TILE, x:x + X.TILE] > 0).mean()))
frac = np.array(frac)
assert len(frac) == S.shape[1], (len(frac), S.shape)
unchanged = frac == 0
E["by_change_size"] = {}
say(f"   {'changed tiles are those with':30s} {'n':>5s}  " + "  ".join(f"{n[:24]:>24s}" for n in names))
for lo in (0.0001, 0.05, 0.15, 0.30, 0.50):
    pos = frac >= lo
    keep = pos | unchanged                   # compare against truly unchanged tiles
    row = {}
    for n, s in zip(names, S[1:]):
        auc, _ = X.roc_auc(s[keep], pos[keep])
        row[n] = round(auc, 4)
    E["by_change_size"][str(lo)] = {"n_changed": int(pos.sum()), "auc": row}
    say(f"   >= {100*lo:5.1f}% of the tile changed   {int(pos.sum()):5d}  "
        + "  ".join(f"{row[n]:24.3f}" for n in names))

# ---------------------------------------------------------------- B
say("\n[B] Gate C at matched thresholds, TEST cities")
say(f"   {'thr':>4s} | {'FPR% off':>8s} {'FPR% on':>8s} | {'Rec off':>7s} {'Rec on':>7s} "
    f"{'kept':>5s} | {'Prec off':>8s} {'Prec on':>8s} | {'F1 off':>6s} {'F1 on':>6s}")
E["gate_c_matched"] = {}
for thr in (20, 30, 40, 50, 60, 80):
    res = {}
    for sup in (False, True):
        per = []
        for c in X.TEST:
            d = cities[c]
            m = cd.analyse_pair(d["img1"], d["img2"], suppress=sup, threshold=thr)[0]
            per.append(M.change_metrics(m, d["label"]))
        r = X.micro(per)
        neg = sum((cities[c]["label"] == 0).sum() for c in X.TEST)
        r["fpr_pct"] = round(100 * sum(p["fp"] for p in per) / neg, 3)
        res[sup] = r
    kept = res[True]["recall"] / res[False]["recall"] if res[False]["recall"] else 0
    E["gate_c_matched"][str(thr)] = {
        "off": res[False], "on": res[True], "recall_kept": round(kept, 3),
        "false_alarm_half": res[True]["fpr_pct"] < res[False]["fpr_pct"],
        "sensitivity_half": kept >= 0.8}
    say(f"   {thr:4d} | {res[False]['fpr_pct']:8.2f} {res[True]['fpr_pct']:8.2f} | "
        f"{res[False]['recall']:7.3f} {res[True]['recall']:7.3f} {100*kept:4.0f}% | "
        f"{res[False]['precision']:8.3f} {res[True]['precision']:8.3f} | "
        f"{res[False]['f1']:6.3f} {res[True]['f1']:6.3f}")
passed = [t for t, v in E["gate_c_matched"].items()
          if v["false_alarm_half"] and v["sensitivity_half"]]
say(f"   both halves pass at thresholds: {passed or 'none'}")

# which part of suppression does the work? (threshold 40, test cities)
say("\n   which part of suppression matters (threshold 40, TEST):")
from src.suppress import radiometric_normalise, usable_mask, drop_small_regions  # noqa: E402
variants = {
    "nothing": lambda a, b: cd.change_mask(a, b, 40),
    "brightness matching only": lambda a, b: cd.change_mask(a, radiometric_normalise(a, b), 40),
    "+ cloud/shadow mask": lambda a, b: cv2.bitwise_and(
        cd.change_mask(a, radiometric_normalise(a, b), 40), usable_mask(a, b)),
    "+ remove small specks (full)": lambda a, b: drop_small_regions(cv2.bitwise_and(
        cd.change_mask(a, radiometric_normalise(a, b), 40), usable_mask(a, b))),
}
E["suppression_parts"] = {}
for name, fn in variants.items():
    per = [M.change_metrics(fn(cities[c]["img1"], cities[c]["img2"]), cities[c]["label"])
           for c in X.TEST]
    r = X.micro(per)
    neg = sum((cities[c]["label"] == 0).sum() for c in X.TEST)
    r["fpr_pct"] = round(100 * sum(p["fp"] for p in per) / neg, 3)
    E["suppression_parts"][name] = r
    say(f"   {name:30s} P={r['precision']:.3f} R={r['recall']:.3f} "
        f"F1={r['f1']:.3f} FPR={r['fpr_pct']:.2f}%")
masked = [100 * float((usable_mask(cities[c]["img1"], cities[c]["img2"]) == 0).mean())
          for c in X.TEST]
E["cloud_mask_pct_per_test_city"] = dict(zip(X.TEST, [round(v, 2) for v in masked]))
say(f"   % of each test city removed by the cloud/shadow mask: {E['cloud_mask_pct_per_test_city']}")

# ---------------------------------------------------------------- C
say("\n[C] Tile-size ablation: 128 px tiles")
T = 128
tiles, tags, fr = [], [], []
for c, d in cities.items():
    h, w = d["label"].shape
    for y in range(0, h - T + 1, T):
        for x in range(0, w - T + 1, T):
            fr.append(float((d["label"][y:y + T, x:x + T] > 0).mean()))
            for date in (1, 2):
                tiles.append(d[f"img{date}"][y:y + T, x:x + T])
                s = (slice(y, y + T), slice(x, x + T))
                b = d["bands"][date]
                ndvi = (b["B08"][s] - b["B04"][s]) / (b["B08"][s] + b["B04"][s] + 1e-6)
                ndwi = (b["B03"][s] - b["B08"][s]) / (b["B03"][s] + b["B08"][s] + 1e-6)
                tags.append({"water": float((ndwi > 0.1).mean()) >= 0.30,
                             "vegetation": float((ndvi > 0.5).mean()) >= 0.70,
                             "builtup": float(((ndvi < 0.2) & (ndwi < 0)).mean()) >= 0.70})
fr = np.array(fr)
say(f"   {len(tiles)} tiles of {T}x{T} px ({len(fr)} locations)")
E["tile128"] = {"n_tiles": len(tiles)}
i1 = np.arange(0, len(tiles), 2)
for backend in ("openclip", "remoteclip"):
    e = Embedder(backend=backend)
    t0 = time.perf_counter()
    v = np.concatenate([e.encode_images(tiles[i:i + 64]) for i in range(0, len(tiles), 64)])
    aps = {}
    for key, text in X.QUERIES.items():
        order = np.argsort(-(v @ e.encode_text(text)[0]))
        rel = {i for i, t in enumerate(tags) if t[key]}
        aps[key] = {"ap": round(M.average_precision(order.tolist(), rel), 4),
                    "p10": M.precision_at_k(order, rel, 10), "n_relevant": len(rel)}
    sims = v[i1] @ v[i1 + 1].T
    rank = (sims > np.diag(sims)[:, None]).sum(1)
    dist = 1 - np.sum(v[i1] * v[i1 + 1], 1)
    auc5, _ = X.roc_auc(dist, fr >= 0.05)
    E["tile128"][backend] = {"retrieval": aps,
                             "mAP": round(float(np.mean([a["ap"] for a in aps.values()])), 4),
                             "recall@1": round(float((rank < 1).mean()), 4),
                             "recall@5": round(float((rank < 5).mean()), 4),
                             "change_auc": round(auc5, 4),
                             "embed_s": round(time.perf_counter() - t0, 1)}
    say(f"   {backend:10s} mAP={E['tile128'][backend]['mAP']:.3f} "
        f"{ {k: (a['p10'], a['ap']) for k, a in aps.items()} } "
        f"recall@1={E['tile128'][backend]['recall@1']:.3f} change AUC={auc5:.3f}")
prev = {k: round(float(np.mean([t[k] for t in tags])), 4) for k in X.QUERIES}
E["tile128"]["prevalence"] = prev
E["tile128"]["changed_share"] = round(float((fr >= 0.05).mean()), 4)
say(f"   random mAP would be about {np.mean(list(prev.values())):.3f} (tag prevalence {prev})")

(OUT / "extra_results.json").write_text(json.dumps(E, indent=2, default=str))
say(f"\nSaved {OUT / 'extra_results.json'}")
