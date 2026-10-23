"""SYSU-CD experiment (0.5 m per pixel, Hong Kong 2007 vs 2014, many change types).

    python experiments/run_sysu.py

LEVIR-CD showed the embedding idea works on sharp images, but nearly every
LEVIR change is a new building. SYSU-CD mixes new buildings, roads, vegetation
loss, sea reclamation and suburban growth, with binary change masks (no type
labels). Same protocol as LEVIR: train nothing, pick on VAL, report on TEST.
Each SYSU pair is 256 x 256 px, so one pair = one tile.

Reads data/SYSU-CD/{val,test}-*.parquet (Hugging Face mirror ericyu/SYSU_CD).
Saves results/sysu_results.json and results/sysu_log.txt, and writes a fixed
choice of test pairs as PNGs to data/SYSU-CD/{A,B,label}/ for the demo.
"""

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
os.environ["RUN_LOG"] = "sysu_log.txt"          # run_levir opens this on import
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import patch_models as PM                       # noqa: E402
import run_levir as RL                          # noqa: E402
import run_levir_text as RT                     # noqa: E402
from src import change_detect as cd             # noqa: E402
from src.suppress import radiometric_normalise  # noqa: E402

DATA = ROOT / "data" / "SYSU-CD"
OUT = ROOT / "results"
CHANGED = RL.CHANGED                             # a tile counts as changed if >= 5% of it changed
DEMO_N = 12
# CLIP ViT-B/16 was run once (results/sysu_log_run1_with_clip_b16.txt: patch mean,
# test AUC 0.746) and is not repeated; the two DINOv2 models are still to do.
# SYSU_MODELS="CLIP ViT-B/16,DINOv2 ViT-S/14" picks other models by name.
MODELS = [m for m in PM.MODELS if m[0] in os.environ.get(
    "SYSU_MODELS", "CLIP ViT-B/32,RemoteCLIP ViT-B/32").split(",")]
say, auc = RL.say, RL.auc
R = {"tile_px": 256, "changed_if_fraction_at_least": CHANGED, "device": PM.DEV}
PROMPTS = RT.PROMPTS + [
    "a satellite image of a new road",
    "a satellite image of a construction site",
    "a satellite image of land reclaimed from the sea",
    "a satellite image of a city with buildings and roads",
]


def load_split(name):
    import pyarrow.parquet as pq
    pairs = []
    for f in sorted(DATA.glob(f"{name}-*.parquet")):
        t = pq.read_table(f).to_pydict()
        for a, b, l in zip(t["imageA"], t["imageB"], t["label"]):
            dec = lambda x, flag: cv2.imdecode(np.frombuffer(x["bytes"], np.uint8), flag)
            lab = dec(l, cv2.IMREAD_GRAYSCALE)
            pairs.append({"img1": dec(a, cv2.IMREAD_COLOR), "img2": dec(b, cv2.IMREAD_COLOR),
                          "label": ((lab > 127) * 255).astype(np.uint8)})
    return pairs


def export_demo(test):
    """Fixed, label-based choice (never score-based): a seeded random sample of
    pairs with 10-60% labelled change, plus 3 with almost none."""
    fr = np.array([float((p["label"] > 0).mean()) for p in test])
    rng = np.random.default_rng(0)
    mid = rng.permutation(np.where((fr >= 0.10) & (fr <= 0.60))[0])[:DEMO_N - 3]
    quiet = rng.permutation(np.where(fr < 0.002)[0])[:3]
    names = []
    for i in list(mid) + list(quiet):
        n = f"test_{i}"
        for sub, key in (("A", "img1"), ("B", "img2"), ("label", "label")):
            (DATA / sub).mkdir(exist_ok=True)
            cv2.imwrite(str(DATA / sub / f"{n}.png"), test[i][key])
        names.append(n)
    (DATA / "demo_pick.json").write_text(json.dumps(names))
    say(f"   demo: wrote {len(names)} test pairs to data/SYSU-CD/A, B, label")


def main():
    t_all = time.perf_counter()
    say("=" * 70); say("SYSU-CD EXPERIMENT RUN"); say("=" * 70)
    val, test = load_split("val"), load_split("test")
    R["val_files"] = sorted(f.name for f in DATA.glob("val-*.parquet"))   # 2 of 3 shards is enough to pick settings
    for nm, sp in (("val", val), ("test", test)):
        fr = np.array([float((p["label"] > 0).mean()) for p in sp])
        R[f"n_{nm}"] = len(sp)
        R[f"changed_pct_pixels_{nm}"] = round(100 * float(fr.mean()), 3)
        R[f"changed_tiles_{nm}"] = int((fr >= CHANGED).sum())
        say(f"   {nm}: {len(sp)} pairs, {100*fr.mean():.1f}% of pixels changed, "
            f"{int((fr >= CHANGED).sum())} tiles count as changed")
    export_demo(test)

    V1, V2 = np.stack([p["img1"] for p in val]), np.stack([p["img2"] for p in val])
    S1, S2 = np.stack([p["img1"] for p in test]), np.stack([p["img2"] for p in test])
    vy = np.array([(p["label"] > 0).mean() >= CHANGED for p in val])
    sy = np.array([(p["label"] > 0).mean() >= CHANGED for p in test])
    R["random_precision"] = round(float(sy.mean()), 4)

    # ------------------------------------------------ 1. tile-level change, image only
    say("\n[1] Tile-level change signal (AUC; pick on VAL, report on TEST)")
    R["tile"], heat_val, heat_test, best = {}, {}, {}, None
    for mname, factory in MODELS:
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
        R["tile"][mname] = {"variants": res, "picked_on_val": pick,
                            "same_place_recall@1": round(float((rank < 1).mean()), 4),
                            "same_place_recall@5": round(float((rank < 5).mean()), 4),
                            "seconds": round(time.perf_counter() - t0, 1)}
        say(f"   {mname:20s} ({time.perf_counter()-t0:.0f}s)  same place found first: "
            f"{100*(rank<1).mean():.1f}% of {len(rank)}")
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
    R["best_model_on_val"] = best
    R["not_in_this_run"] = {"CLIP ViT-B/16": "run 1 log: patch mean picked on val, AUC val 0.677 "
                                             "test 0.746, same place first 11.2%",
                            "DINOv2 ViT-S/14, ViT-B/14": "not run yet"}

    def tile_ref(T1, T2, norm):
        return np.array([float(cv2.absdiff(a, radiometric_normalise(a, b) if norm else b).mean())
                         for a, b in zip(T1, T2)])
    R["tile_reference"] = {}
    for nm, norm in (("pixel difference (raw)", False), ("pixel difference (brightness matched)", True)):
        R["tile_reference"][nm] = {"val": auc(tile_ref(V1, V2, norm), vy),
                                   "test": auc(tile_ref(S1, S2, norm), sy)}
        say(f"   reference: {nm:38s} AUC val={R['tile_reference'][nm]['val']:.3f} "
            f"test={R['tile_reference'][nm]['test']:.3f}")

    # ------------------------------------------------ 2. tile-level change from a text query
    say("\n[2] Text query change: sim(date 2, query) - sim(date 1, query)")
    R["text"] = {}
    for name, remote in (("OpenCLIP ViT-B/32", False), ("RemoteCLIP ViT-B/32", True)):
        model, tok = RT.load(remote)
        ev1, ev2, es1, es2 = (RT.embed_images(model, X) for X in (V1, V2, S1, S2))
        T = RT.embed_text(model, tok, PROMPTS)
        bare = RT.embed_text(model, tok, [RT.BARE])[0]
        res = {}
        for i, p in enumerate(PROMPTS):
            q2 = T[i] - bare
            res[p] = {"plain": {"val": auc(ev2 @ T[i] - ev1 @ T[i], vy),
                                "test": auc(es2 @ T[i] - es1 @ T[i], sy)},
                      "vs_bare_land": {"val": auc(ev2 @ q2 - ev1 @ q2, vy),
                                       "test": auc(es2 @ q2 - es1 @ q2, sy)}}
        p, k = max(((p, k) for p in res for k in res[p]), key=lambda pk: res[pk[0]][pk[1]]["val"])
        q = T[PROMPTS.index(p)] - (bare if k == "vs_bare_land" else 0)
        top = np.argsort(-(es2 @ q - es1 @ q))[:50]
        R["text"][name] = {"per_prompt": res, "picked_on_val": {"prompt": p, "variant": k},
                           "picked_test_auc": res[p][k]["test"],
                           "picked_test_precision@50": round(float(sy[top].mean()), 4)}
        say(f"   {name}: picked '{p}' ({k}) val {res[p][k]['val']:.3f} test {res[p][k]['test']:.3f}"
            f"  P@50 {sy[top].mean():.2f}")
        del model
        if PM.DEV == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------ 3. pixel-level maps
    say("\n[3] Pixel-level change maps (thresholds picked on VAL)")
    R["pixel"] = {}

    def report(name, masks):
        r = RL.micro(masks, test)
        R["pixel"][name] = r
        say(f"   {name:40s} P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} "
            f"IoU={r['iou']:.3f} FPR={r['fpr_pct']:.2f}%")
    report("Pixel difference + Otsu", [cd.baseline_pixel_otsu(p["img1"], p["img2"]) for p in test])
    report("Change Vector Analysis", [cd.baseline_cva(p["img1"], p["img2"]) for p in test])
    grid = [20, 30, 40, 50, 60, 80, 100]

    def bright(p, t):
        return cd.change_mask(p["img1"], radiometric_normalise(p["img1"], p["img2"]), t)
    f1s = [RL.micro([bright(p, t) for p in val], val)["f1"] for t in grid]
    tb = grid[int(np.argmax(f1s))]
    R["picked_threshold_brightness"] = tb
    report(f"Ours, brightness matching (thr {tb})", [bright(p, tb) for p in test])
    if best:
        up = lambda h: cv2.resize(h, (256, 256), interpolation=cv2.INTER_LINEAR)
        hv, ht = [up(h) for h in heat_val[best]], [up(h) for h in heat_test[best]]
        allv = np.concatenate([h[::4, ::4].ravel() for h in hv])
        qs = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
        f1s = [RL.micro([((h >= np.quantile(allv, q)) * 255).astype(np.uint8) for h in hv], val)["f1"]
               for q in qs]
        t1 = float(np.quantile(allv, qs[int(np.argmax(f1s))]))
        report(f"Ours, patch heat map ({best})", [((h >= t1) * 255).astype(np.uint8) for h in ht])

    R["runtime_s"] = round(time.perf_counter() - t_all, 1)
    (OUT / "sysu_results.json").write_text(json.dumps(R, indent=1))
    say(f"\nsaved results/sysu_results.json  ({R['runtime_s']:.0f}s)")


if __name__ == "__main__":
    main()
