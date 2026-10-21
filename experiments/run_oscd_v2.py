"""OSCD round 2: can we fix the change-detection result?

    python experiments/run_oscd_v2.py

A. All 13 Sentinel-2 bands for the pixel comparison (we used 3 before).
B. Patch-level features instead of one embedding per tile, with more models
   (CLIP ViT-B/32, RemoteCLIP, CLIP ViT-B/16, DINOv2).
C. Patch-feature heat map used to clean the pixel change map.

Every choice (which variant, which threshold) is made on the 14 TRAIN cities
and reported on the 10 TEST cities.

Saves results/oscd_v2_results.json and results/oscd_v2_log.txt.
"""

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

os.environ["RUN_LOG"] = "oscd_v2_log.txt"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import run_oscd as X                          # noqa: E402
from src import change_detect as cd           # noqa: E402
from src import metrics as M                  # noqa: E402
from src.suppress import radiometric_normalise  # noqa: E402

say, OUT, TILE = X.say, X.OUT, X.TILE
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BANDS13 = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A",
           "B09", "B10", "B11", "B12"]
R = {"device": DEV}


def micro_fpr(masks, cities, names):
    per = [M.change_metrics(masks[c], cities[c]["label"]) for c in names]
    r = X.micro(per)
    neg = sum((cities[c]["label"] == 0).sum() for c in names)
    r["fpr_pct"] = round(100 * sum(p["fp"] for p in per) / neg, 3)
    return r


def auc(scores, labels):
    return round(X.roc_auc(scores, labels)[0], 4)


# ============================================================== load data
say("=" * 70)
say("OSCD ROUND 2")
say("=" * 70)
say(f"device: {DEV}" + (f" ({torch.cuda.get_device_name(0)})" if DEV == "cuda" else ""))
cities = {c: X.load_city(c) for c in X.TRAIN + X.TEST}

# ---------------------------------------------------------------- A. 13 bands
say("\n[A] Pixel comparison with more bands")


def load_stack(city, bands):
    cdir = X.find_dir(city)
    h, w = cities[city]["label"].shape
    out = {}
    for d in (1, 2):
        arrs = []
        for b in bands:
            a = X.read_band(cdir, d, b)
            if a.shape[:2] != (h, w):
                a = cv2.resize(a[:h, :w] if a.shape[0] >= h and a.shape[1] >= w else a,
                               (w, h), interpolation=cv2.INTER_LINEAR)
            arrs.append(a[:h, :w])
        out[d] = np.dstack(arrs).astype(np.float32)
    return out[1], out[2]


def band_diff(city, bands):
    """Standardised multi-band difference after matching each band of date 2 to
    date 1 (mean and spread). One number per pixel: how many 'typical spreads'
    the pixel moved, averaged over the bands."""
    a, b = load_stack(city, bands)
    m1, s1 = a.mean((0, 1)), a.std((0, 1)) + 1e-6
    m2, s2 = b.mean((0, 1)), b.std((0, 1)) + 1e-6
    bn = (b - m2) / s2 * s1 + m1
    d = (a - bn) / s1
    return np.sqrt((d ** 2).mean(axis=2))


def clean(mask):
    k = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)


R["bands"] = {}
band_maps = {}
for name, bands in (("3 bands (B02,B03,B04)", ["B02", "B03", "B04"]),
                    ("4 bands (+B08 infrared)", ["B02", "B03", "B04", "B08"]),
                    ("10 bands (no 60 m bands)", ["B02", "B03", "B04", "B05", "B06", "B07",
                                                 "B08", "B8A", "B11", "B12"]),
                    ("13 bands (all)", BANDS13)):
    t0 = time.perf_counter()
    D = {c: band_diff(c, bands) for c in cities}
    grid = [0.8, 1.0, 1.2, 1.5, 1.8, 2.1, 2.5, 3.0]
    f1s = [micro_fpr({c: clean(((D[c] >= t) * 255).astype(np.uint8)) for c in X.TRAIN},
                     cities, X.TRAIN)["f1"] for t in grid]
    t_best = grid[int(np.argmax(f1s))]
    masks = {c: clean(((D[c] >= t_best) * 255).astype(np.uint8)) for c in cities}
    te, tr = micro_fpr(masks, cities, X.TEST), micro_fpr(masks, cities, X.TRAIN)
    R["bands"][name] = {"threshold": t_best, "test": te, "train": tr,
                        "train_sweep": dict(zip(map(str, grid), f1s))}
    band_maps[name] = D
    say(f"   {name:26s} thr={t_best:<4} TEST P={te['precision']:.3f} R={te['recall']:.3f} "
        f"F1={te['f1']:.3f} IoU={te['iou']:.3f} FPR={te['fpr_pct']:.2f}%   "
        f"(train F1={tr['f1']:.3f}, {time.perf_counter()-t0:.0f}s)")
    if name.startswith("13"):
        np.savez_compressed(OUT / "masks_13band.npz", **{c: masks[c] for c in X.TEST})
best_band = max(R["bands"], key=lambda k: R["bands"][k]["train"]["f1"])
R["bands_best_on_train"] = best_band
say(f"   best on TRAIN: {best_band}  (for reference CVA on 3 bands had test F1 0.353)")

# ------------------------------------------------- B. patch-level features
say("\n[B] Patch-level features")
T1, T2, loc_city, loc_yx, frac = [], [], [], [], []
for c, d in cities.items():
    h, w = d["label"].shape
    for (y, x) in X.tile_grid(h, w):
        T1.append(d["img1"][y:y + TILE, x:x + TILE])
        T2.append(d["img2"][y:y + TILE, x:x + TILE])
        loc_city.append(c); loc_yx.append((y, x))
        frac.append(float((d["label"][y:y + TILE, x:x + TILE] > 0).mean()))
T1, T2 = np.stack(T1), np.stack(T2)
frac = np.array(frac)
ylab = frac >= X.CHANGED_TILE_FRAC
is_test = np.array([c in X.TEST for c in loc_city])
say(f"   {len(T1)} tile pairs; changed: {int(ylab.sum())} "
    f"(train {int(ylab[~is_test].sum())}, test {int(ylab[is_test].sum())})")

CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
IMNET_MEAN, IMNET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def prep(batch, mean, std, size=224):
    x = torch.from_numpy(np.ascontiguousarray(batch[..., ::-1])).permute(0, 3, 1, 2)
    x = x.float().div(255).to(DEV)
    x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False).clamp(0, 1)
    m = torch.tensor(mean, device=DEV).view(1, 3, 1, 1)
    s = torch.tensor(std, device=DEV).view(1, 3, 1, 1)
    return (x - m) / s


def make_clip(arch, pretrained=None, remote=False):
    import open_clip
    if remote:
        model, _, _ = open_clip.create_model_and_transforms(arch)
        from huggingface_hub import hf_hub_download
        ck = hf_hub_download("chendelong/RemoteCLIP", f"RemoteCLIP-{arch}.pt")
        model.load_state_dict(torch.load(ck, map_location="cpu"))
    else:
        model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
    model = model.to(DEV).eval()
    model.visual.output_tokens = True

    def fwd(x):
        out = model.visual(x)
        pooled, tokens = out if isinstance(out, (tuple, list)) else (out, None)
        return pooled.float(), tokens.float()
    return fwd, CLIP_MEAN, CLIP_STD


def make_dino(name):
    model = torch.hub.load("facebookresearch/dinov2", name, verbose=False).to(DEV).eval()

    def fwd(x):
        o = model.forward_features(x)
        return o["x_norm_clstoken"].float(), o["x_norm_patchtokens"].float()
    return fwd, IMNET_MEAN, IMNET_STD


MODELS = [
    ("CLIP ViT-B/32", lambda: make_clip("ViT-B-32", "laion2b_s34b_b79k")),
    ("RemoteCLIP ViT-B/32", lambda: make_clip("ViT-B-32", remote=True)),
    ("CLIP ViT-B/16", lambda: make_clip("ViT-B-16", "laion2b_s34b_b88k")),
    ("DINOv2 ViT-S/14", lambda: make_dino("dinov2_vits14")),
    ("DINOv2 ViT-B/14", lambda: make_dino("dinov2_vitb14")),
]


def nrm(t):
    return t / t.norm(dim=-1, keepdim=True).clamp_min(1e-6)


R["patch"] = {}
heat = {}
for mname, factory in MODELS:
    try:
        t0 = time.perf_counter()
        fwd, mean, std = factory()
    except Exception as exc:                               # noqa: BLE001
        say(f"   {mname}: could not load ({type(exc).__name__}: {str(exc)[:90]})")
        continue
    cls_d, raw_maps, cen_maps = [], [], []
    with torch.no_grad(), torch.autocast(DEV, enabled=(DEV == "cuda")):
        for i in range(0, len(T1), 64):
            c1, k1 = fwd(prep(T1[i:i + 64], mean, std))
            c2, k2 = fwd(prep(T2[i:i + 64], mean, std))
            cls_d.append((1 - (nrm(c1) * nrm(c2)).sum(-1)).cpu().numpy())
            raw_maps.append((1 - (nrm(k1) * nrm(k2)).sum(-1)).cpu().numpy())
            # centred: remove each tile's own average token first, so a change
            # that hits the WHOLE tile (season, light) cancels out and only
            # local change is left
            k1c, k2c = k1 - k1.mean(1, keepdim=True), k2 - k2.mean(1, keepdim=True)
            cen_maps.append((1 - (nrm(k1c) * nrm(k2c)).sum(-1)).cpu().numpy())
    cls_d = np.concatenate(cls_d)
    raw_maps, cen_maps = np.concatenate(raw_maps), np.concatenate(cen_maps)
    g = int(round(np.sqrt(raw_maps.shape[1])))
    secs = time.perf_counter() - t0

    def top(m, share=0.1):
        k = max(1, int(round(m.shape[1] * share)))
        return np.sort(m, axis=1)[:, -k:].mean(1)

    variants = {
        "whole-tile embedding": cls_d,
        "patch mean": raw_maps.mean(1),
        "patch max": raw_maps.max(1),
        "patch top-10%": top(raw_maps),
        "centred patch mean": cen_maps.mean(1),
        "centred patch max": cen_maps.max(1),
        "centred patch top-10%": top(cen_maps),
    }
    res = {v: {"train": auc(s[~is_test], ylab[~is_test]),
               "test": auc(s[is_test], ylab[is_test]),
               "all": auc(s, ylab)} for v, s in variants.items()}
    pick = max((v for v in res if v != "whole-tile embedding"), key=lambda v: res[v]["train"])
    R["patch"][mname] = {"grid": g, "seconds": round(secs, 1), "variants": res,
                         "picked_on_train": pick}
    say(f"   {mname:20s} patch grid {g}x{g}, {secs:.0f}s")
    for v in variants:
        mark = "  <- picked on train" if v == pick else ""
        say(f"      {v:24s} AUC train={res[v]['train']:.3f} test={res[v]['test']:.3f} "
            f"all={res[v]['all']:.3f}{mark}")
    heat[mname] = (cen_maps if pick.startswith("centred") else raw_maps).reshape(-1, g, g)
    del fwd
    if DEV == "cuda":
        torch.cuda.empty_cache()

# pixel-difference reference on the same split
ref = {}
for nm, fn in (("pixel difference (raw)", lambda a, b: float(cv2.absdiff(a, b).mean())),):
    ref[nm] = np.array([fn(a, b) for a, b in zip(T1, T2)])
nd = []
for c, d in cities.items():
    n2 = radiometric_normalise(d["img1"], d["img2"])
    h, w = d["label"].shape
    for (y, x) in X.tile_grid(h, w):
        nd.append(float(cv2.absdiff(d["img1"][y:y + TILE, x:x + TILE], n2[y:y + TILE, x:x + TILE]).mean()))
ref["pixel difference (brightness matched)"] = np.array(nd)
b13 = band_maps["13 bands (all)"]
ref["13-band difference (tile mean)"] = np.array(
    [float(b13[c][y:y + TILE, x:x + TILE].mean()) for c, (y, x) in zip(loc_city, loc_yx)])
R["patch_reference"] = {}
for nm, s in ref.items():
    R["patch_reference"][nm] = {"train": auc(s[~is_test], ylab[~is_test]),
                                "test": auc(s[is_test], ylab[is_test]), "all": auc(s, ylab)}
    say(f"   reference: {nm:38s} AUC train={R['patch_reference'][nm]['train']:.3f} "
        f"test={R['patch_reference'][nm]['test']:.3f} all={R['patch_reference'][nm]['all']:.3f}")

# --------------------------------------- C. heat map + pixel change map
say("\n[C] Patch heat map used to clean the pixel change map")
if heat:
    best_model = max(heat, key=lambda k: R["patch"][k]["variants"][R["patch"][k]["picked_on_train"]]["train"])
    hm = heat[best_model]
    say(f"   model picked on TRAIN: {best_model} ({R['patch'][best_model]['picked_on_train']})")
    city_heat = {c: np.zeros(cities[c]["label"].shape, np.float32) for c in cities}
    for k, (c, (y, x)) in enumerate(zip(loc_city, loc_yx)):
        city_heat[c][y:y + TILE, x:x + TILE] = cv2.resize(hm[k], (TILE, TILE),
                                                          interpolation=cv2.INTER_LINEAR)
    pix = {c: (b13[c] >= R["bands"]["13 bands (all)"]["threshold"]) for c in cities}
    allh = np.concatenate([city_heat[c][::4, ::4].ravel() for c in X.TRAIN])
    R["heat"] = {"model": best_model}
    # heat map alone
    qs = [0.80, 0.85, 0.90, 0.93, 0.95, 0.97]
    f1s = [micro_fpr({c: ((city_heat[c] >= np.quantile(allh, q)) * 255).astype(np.uint8)
                      for c in X.TRAIN}, cities, X.TRAIN)["f1"] for q in qs]
    tq = float(np.quantile(allh, qs[int(np.argmax(f1s))]))
    te = micro_fpr({c: ((city_heat[c] >= tq) * 255).astype(np.uint8) for c in X.TEST}, cities, X.TEST)
    R["heat"]["heat_only"] = te
    say(f"   heat map alone              TEST P={te['precision']:.3f} R={te['recall']:.3f} "
        f"F1={te['f1']:.3f} FPR={te['fpr_pct']:.2f}%")
    # 13-band pixels kept only where the heat map agrees
    qs = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
    f1s = [micro_fpr({c: clean(((pix[c] & (city_heat[c] >= np.quantile(allh, q))) * 255).astype(np.uint8))
                      for c in X.TRAIN}, cities, X.TRAIN)["f1"] for q in qs]
    qb = qs[int(np.argmax(f1s))]
    tq = float(np.quantile(allh, qb))
    comb = {c: clean(((pix[c] & (city_heat[c] >= tq)) * 255).astype(np.uint8)) for c in cities}
    te = micro_fpr(comb, cities, X.TEST)
    R["heat"]["bands13_and_heat"] = {**te, "quantile_picked": qb,
                                     "train_sweep": dict(zip(map(str, qs), f1s))}
    say(f"   13 bands AND heat (q={qb})    TEST P={te['precision']:.3f} R={te['recall']:.3f} "
        f"F1={te['f1']:.3f} FPR={te['fpr_pct']:.2f}%")
    np.savez_compressed(OUT / "masks_13band_heat.npz", **{c: comb[c] for c in X.TEST})
    np.savez_compressed(OUT / "heat_test.npz", **{c: city_heat[c] for c in X.TEST})

(OUT / "oscd_v2_results.json").write_text(json.dumps(R, indent=2, default=str))
say(f"\nSaved {OUT / 'oscd_v2_results.json'}")
