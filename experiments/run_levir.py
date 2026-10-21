"""LEVIR-CD experiment (0.5 m per pixel, building change).

    python experiments/run_levir.py

Asks the open question from OSCD: does the embedding idea fail everywhere, or
only at 10 m resolution? We train nothing, so we use only the VAL split (to
pick variants and thresholds) and the TEST split (to report).

Saves results/levir_results.json and results/levir_log.txt.
"""

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

os.environ.setdefault("RUN_LOG", "levir_log.txt")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import patch_models as PM                       # noqa: E402
from src import change_detect as cd             # noqa: E402
from src import metrics as M                    # noqa: E402
from src.suppress import radiometric_normalise  # noqa: E402

DATA = ROOT / "data" / "LEVIR-CD"
OUT = ROOT / "results"
LOG = open(OUT / os.environ["RUN_LOG"], "w", encoding="utf-8")
TILE = 256
CHANGED = 0.05
R = {"tile_px": TILE, "device": PM.DEV}


def say(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    LOG.write(line + "\n"); LOG.flush()


def auc(s, y):
    from sklearn.metrics import roc_auc_score
    return round(float(roc_auc_score(y, s)), 4)


def load_split(name):
    """Works for both layouts: <root>/<split>/A/*.png, or one merged A/ folder
    whose files are named <split>_<n>.png (how the mirror we used unzips)."""
    cands = [p.parent for p in DATA.rglob("A") if p.is_dir()]
    root = next((r for r in cands if r.name == name), cands[0])
    files = sorted((root / "A").glob("*.png"))
    if root.name != name:
        files = [f for f in files if f.name.startswith(name + "_")]
    out = []
    for pa in files:
        a, b = cv2.imread(str(pa)), cv2.imread(str(root / "B" / pa.name))
        l = cv2.imread(str(root / "label" / pa.name), cv2.IMREAD_GRAYSCALE)
        out.append({"name": pa.stem, "img1": a, "img2": b,
                    "label": ((l > 127) * 255).astype(np.uint8)})
    return out


def tiles_of(pairs):
    T1, T2, fr, where = [], [], [], []
    for pi, p in enumerate(pairs):
        h, w = p["label"].shape
        for y in range(0, h - TILE + 1, TILE):
            for x in range(0, w - TILE + 1, TILE):
                T1.append(p["img1"][y:y + TILE, x:x + TILE])
                T2.append(p["img2"][y:y + TILE, x:x + TILE])
                fr.append(float((p["label"][y:y + TILE, x:x + TILE] > 0).mean()))
                where.append((pi, y, x))
    return np.stack(T1), np.stack(T2), np.array(fr), where


def micro(masks, pairs):
    per = [M.change_metrics(m, p["label"]) for m, p in zip(masks, pairs)]
    a = M.aggregate(per)
    neg = sum(int((p["label"] == 0).sum()) for p in pairs)
    return {"precision": round(a["micro_precision"], 4), "recall": round(a["micro_recall"], 4),
            "f1": round(a["micro_f1"], 4), "iou": round(a["micro_iou"], 4),
            "fpr_pct": round(100 * sum(p["fp"] for p in per) / neg, 3)}


def main():
    t_all = time.perf_counter()
    say("=" * 70); say("LEVIR-CD EXPERIMENT RUN"); say("=" * 70)
    say(f"device: {PM.DEV}")
    val, test = load_split("val"), load_split("test")
    for nm, sp in (("val", val), ("test", test)):
        tot = sum(p["label"].size for p in sp); ch = sum(int((p["label"] > 0).sum()) for p in sp)
        R[f"n_{nm}"] = len(sp); R[f"changed_pct_{nm}"] = round(100 * ch / tot, 3)
        say(f"   {nm}: {len(sp)} pairs of {sp[0]['label'].shape[1]}x{sp[0]['label'].shape[0]} px, "
            f"{100*ch/tot:.2f}% of pixels changed")
    V1, V2, vfr, vwhere = tiles_of(val)
    S1, S2, sfr, swhere = tiles_of(test)
    vy, sy = vfr >= CHANGED, sfr >= CHANGED
    R["tiles"] = {"val": len(V1), "test": len(S1), "val_changed": int(vy.sum()),
                  "test_changed": int(sy.sum())}
    say(f"   tiles of {TILE} px: val {len(V1)} ({int(vy.sum())} changed), "
        f"test {len(S1)} ({int(sy.sum())} changed)")

    # ------------------------------------------------ 1. tile-level change
    say("\n[1] Tile-level change signal (AUC; pick on VAL, report on TEST)")
    R["tile"] = {}
    heat_test, heat_val, best = {}, {}, None
    for mname, factory in PM.MODELS:
        try:
            t0 = time.perf_counter()
            fwd, mean, std = factory()
        except Exception as exc:                            # noqa: BLE001
            say(f"   {mname}: could not load ({type(exc).__name__}: {str(exc)[:90]})")
            continue
        ev = PM.encode_pairs(V1, V2, fwd, mean, std)
        es = PM.encode_pairs(S1, S2, fwd, mean, std, keep_cls=True)
        vv, sv = PM.variants(ev), PM.variants(es)
        res = {k: {"val": auc(vv[k], vy), "test": auc(sv[k], sy)} for k in vv}
        pick = max((k for k in res if k != "whole-tile embedding"), key=lambda k: res[k]["val"])
        sims = es["v1"] @ es["v2"].T
        rank = (sims > np.diag(sims)[:, None]).sum(1)
        R["tile"][mname] = {"grid": es["grid"], "variants": res, "picked_on_val": pick,
                            "same_place_recall@1": round(float((rank < 1).mean()), 4),
                            "same_place_recall@5": round(float((rank < 5).mean()), 4),
                            "seconds": round(time.perf_counter() - t0, 1)}
        say(f"   {mname:20s} grid {es['grid']}x{es['grid']}  ({time.perf_counter()-t0:.0f}s)  "
            f"same place found first: {100*(rank<1).mean():.1f}% of {len(rank)}")
        for k in res:
            say(f"      {k:24s} AUC val={res[k]['val']:.3f} test={res[k]['test']:.3f}"
                + ("  <- picked on val" if k == pick else ""))
        key = "cen" if pick.startswith("centred") else "raw"
        g = es["grid"]
        heat_val[mname], heat_test[mname] = ev[key].reshape(-1, g, g), es[key].reshape(-1, g, g)
        if best is None or res[pick]["val"] > R["tile"][best]["variants"][R["tile"][best]["picked_on_val"]]["val"]:
            best = mname
        del fwd
        if PM.DEV == "cuda":
            torch.cuda.empty_cache()

    def tile_ref(T1, T2, norm):
        out = []
        for a, b in zip(T1, T2):
            if norm:
                b = radiometric_normalise(a, b)
            out.append(float(cv2.absdiff(a, b).mean()))
        return np.array(out)
    R["tile_reference"] = {}
    for nm, norm in (("pixel difference (raw)", False), ("pixel difference (brightness matched)", True)):
        R["tile_reference"][nm] = {"val": auc(tile_ref(V1, V2, norm), vy),
                                   "test": auc(tile_ref(S1, S2, norm), sy)}
        say(f"   reference: {nm:38s} AUC val={R['tile_reference'][nm]['val']:.3f} "
            f"test={R['tile_reference'][nm]['test']:.3f}")
    R["best_model_on_val"] = best

    # ------------------------------------------------ 2. pixel-level maps
    say("\n[2] Pixel-level change maps (thresholds picked on VAL)")
    R["pixel"] = {}

    def report(name, masks):
        r = micro(masks, test)
        R["pixel"][name] = r
        say(f"   {name:40s} P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} "
            f"IoU={r['iou']:.3f} FPR={r['fpr_pct']:.2f}%")
        return r

    report("Pixel difference + Otsu", [cd.baseline_pixel_otsu(p["img1"], p["img2"]) for p in test])
    report("Change Vector Analysis", [cd.baseline_cva(p["img1"], p["img2"]) for p in test])
    report("SSIM differencing", [cd.baseline_ssim(p["img1"], p["img2"]) for p in test])

    grid = [20, 30, 40, 50, 60, 80, 100]

    def bright(p, t):
        return cd.change_mask(p["img1"], radiometric_normalise(p["img1"], p["img2"]), t)
    f1s = [micro([bright(p, t) for p in val], val)["f1"] for t in grid]
    tb = grid[int(np.argmax(f1s))]
    R["picked_threshold_brightness"] = tb
    bmask_test = [bright(p, tb) for p in test]
    report(f"Ours, brightness matching (thr {tb})", bmask_test)
    f1s = [micro([cd.analyse_pair(p["img1"], p["img2"], True, t)[0] for p in val], val)["f1"] for t in grid]
    tf = grid[int(np.argmax(f1s))]
    report(f"Ours, all three removal steps (thr {tf})",
           [cd.analyse_pair(p["img1"], p["img2"], True, tf)[0] for p in test])

    if best:
        say(f"   patch heat map from: {best} ({R['tile'][best]['picked_on_val']})")

        def stitch(pairs, where, maps):
            hs = [np.zeros(p["label"].shape, np.float32) for p in pairs]
            for (pi, y, x), m in zip(where, maps):
                hs[pi][y:y + TILE, x:x + TILE] = cv2.resize(m, (TILE, TILE), interpolation=cv2.INTER_LINEAR)
            return hs
        hv, ht = stitch(val, vwhere, heat_val[best]), stitch(test, swhere, heat_test[best])
        allv = np.concatenate([h[::4, ::4].ravel() for h in hv])
        qs = [0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97]
        f1s = [micro([((h >= np.quantile(allv, q)) * 255).astype(np.uint8) for h in hv], val)["f1"] for q in qs]
        q1 = qs[int(np.argmax(f1s))]; t1 = float(np.quantile(allv, q1))
        heat_masks = [((h >= t1) * 255).astype(np.uint8) for h in ht]
        report(f"Patch heat map alone (top {100*(1-q1):.0f}%)", heat_masks)
        bval = [bright(p, tb) for p in val]
        qs = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
        f1s = [micro([cv2.bitwise_and(m, ((h >= np.quantile(allv, q)) * 255).astype(np.uint8))
                      for m, h in zip(bval, hv)], val)["f1"] for q in qs]
        q2 = qs[int(np.argmax(f1s))]; t2 = float(np.quantile(allv, q2))
        comb = [cv2.bitwise_and(m, ((h >= t2) * 255).astype(np.uint8)) for m, h in zip(bmask_test, ht)]
        report(f"Ours, brightness + patch heat (q {q2})", comb)
        R["heat"] = {"model": best, "heat_quantile": q1, "combined_quantile": q2}
        # keep a few examples for figures
        per = [M.change_metrics(m, p["label"])["f1"] for m, p in zip(comb, test)]
        order = np.argsort(per)
        keep = {"good": int(order[-1]), "median": int(order[len(order) // 2]), "bad": int(order[0])}
        R["examples"] = {k: {"name": test[i]["name"], "f1": round(float(per[i]), 3)} for k, i in keep.items()}
        np.savez_compressed(OUT / "levir_examples.npz", **{
            f"{k}_{part}": arr for k, i in keep.items() for part, arr in (
                ("img1", test[i]["img1"]), ("img2", test[i]["img2"]), ("label", test[i]["label"]),
                ("heat", ht[i]), ("bright", bmask_test[i]), ("comb", comb[i]),
                ("cva", cd.baseline_cva(test[i]["img1"], test[i]["img2"])))})

    R["runtime_s"] = round(time.perf_counter() - t_all, 1)
    (OUT / "levir_results.json").write_text(json.dumps(R, indent=2))
    say(f"\nTotal runtime: {R['runtime_s']} s")
    say(f"Saved {OUT / 'levir_results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
