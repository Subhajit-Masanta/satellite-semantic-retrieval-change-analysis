"""Real experiments on the OSCD dataset (24 Sentinel-2 city pairs).

    python experiments/run_oscd.py

Everything printed here is also saved to results/run_log.txt, and every number
in the report comes from results/results.json written by this script.

Thresholds are chosen on the 14 TRAIN cities and reported on the 10 TEST
cities, so no number in the test table was tuned on the data it is scored on.
"""

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import change_detect as cd            # noqa: E402
from src import metrics as M                   # noqa: E402
from src.embedder import Embedder              # noqa: E402
from src.index_builder import SharedIndex      # noqa: E402

DATA = ROOT / "data" / "OSCD"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
LOG = open(OUT / os.environ.get("RUN_LOG", "run_log.txt"), "w", encoding="utf-8")

TILE = int(os.environ.get("TILE_PX", 64))   # 64 px x 10 m = 640 m on the ground
CHANGED_TILE_FRAC = 0.05      # a tile counts as "changed" if >=5% of it changed

TRAIN = ["aguasclaras", "bercy", "bordeaux", "nantes", "paris", "rennes",
         "saclay_e", "abudhabi", "cupertino", "pisa", "beihai", "hongkong",
         "beirut", "mumbai"]
TEST = ["brasilia", "montpellier", "norcia", "rio", "saclay_w", "valencia",
        "dubai", "lasvegas", "milano", "chongqing"]

QUERIES = {
    "water": "a satellite image of a river, sea or water body",
    "vegetation": "a satellite image of green vegetation, forest or farmland",
    "builtup": "a satellite image of dense urban buildings and roads",
}


def say(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    LOG.write(line + "\n")
    LOG.flush()


# ------------------------------------------------------------------ loading
def find_dir(name):
    # the labels folders reuse the city names, so ask for the one with imagery
    hits = [p for p in DATA.rglob(name)
            if p.is_dir() and (p / "imgs_1_rect").is_dir()]
    return hits[0] if hits else None


def read_band(city_dir, date, band):
    p = city_dir / f"imgs_{date}_rect" / f"{band}.tif"
    arr = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(p)
    return arr.astype(np.float32)


def load_city(city):
    cdir = find_dir(city)
    bands = {d: {b: read_band(cdir, d, b) for b in ("B02", "B03", "B04", "B08")}
             for d in (1, 2)}
    # one stretch for BOTH dates, so brightness differences between the dates
    # are kept (they are one of the confounds we want to test against)
    rgb = {}
    for i, b in enumerate(("B02", "B03", "B04")):          # -> BGR order
        both = np.concatenate([bands[1][b].ravel(), bands[2][b].ravel()])
        lo, hi = np.percentile(both, [2, 98])
        for d in (1, 2):
            ch = np.clip((bands[d][b] - lo) / max(hi - lo, 1e-6) * 255, 0, 255)
            rgb.setdefault(d, []).append(ch.astype(np.uint8))
    img1, img2 = (np.dstack(rgb[d]) for d in (1, 2))

    label = None
    for p in DATA.rglob("cm.png"):
        if p.parent.parent.name == city:
            label = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            break
    if label is None:
        raise FileNotFoundError(f"no label for {city}")
    label = ((label > 127) * 255).astype(np.uint8)
    h = min(img1.shape[0], img2.shape[0], label.shape[0])
    w = min(img1.shape[1], img2.shape[1], label.shape[1])
    img1, img2, label = img1[:h, :w], img2[:h, :w], label[:h, :w]
    for d in (1, 2):
        for b in bands[d]:
            bands[d][b] = bands[d][b][:h, :w]

    dates = {}
    dfile = cdir / "dates.txt"
    if dfile.exists():
        for line in dfile.read_text().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                v = v.strip()
                dates[k.strip()] = f"{v[:4]}-{v[4:6]}-{v[6:8]}" if len(v) == 8 else v

    geo = {}
    try:
        import rasterio
        with rasterio.open(cdir / "imgs_1_rect" / "B04.tif") as ds:
            geo = {"crs": str(ds.crs) if ds.crs else None,
                   "bounds": [round(x, 2) for x in ds.bounds],
                   "dtype": ds.dtypes[0], "driver": ds.driver}
    except Exception as exc:                                # noqa: BLE001
        geo = {"error": type(exc).__name__}
    gj = cdir / f"{city}.geojson"
    if gj.exists():                      # footprint of the scene in lon/lat
        try:
            feat = json.loads(gj.read_text())["features"][0]
            ring = np.array(feat["geometry"]["coordinates"][0], dtype=float)
            geo["footprint_lonlat"] = [round(float(ring[:, 0].min()), 4),
                                       round(float(ring[:, 1].min()), 4),
                                       round(float(ring[:, 0].max()), 4),
                                       round(float(ring[:, 1].max()), 4)]
        except Exception:                                   # noqa: BLE001
            pass

    return {"city": city, "img1": img1, "img2": img2, "label": label,
            "bands": bands, "dates": dates, "geo": geo}


def tile_grid(h, w):
    return [(y, x) for y in range(0, h - TILE + 1, TILE)
            for x in range(0, w - TILE + 1, TILE)]


def spectral_tags(bands, y, x):
    """Automatic relevance labels from standard spectral indices.
    NDVI marks vegetation, NDWI marks water. Used ONLY to score retrieval."""
    s = (slice(y, y + TILE), slice(x, x + TILE))
    b3, b4, b8 = bands["B03"][s], bands["B04"][s], bands["B08"][s]
    ndvi = (b8 - b4) / (b8 + b4 + 1e-6)
    ndwi = (b3 - b8) / (b3 + b8 + 1e-6)
    return {"water": float((ndwi > 0.1).mean()) >= 0.30,
            "vegetation": float((ndvi > 0.5).mean()) >= 0.70,
            "builtup": float(((ndvi < 0.2) & (ndwi < 0.0)).mean()) >= 0.70}


# ------------------------------------------------------------------ helpers
def roc_auc(scores, labels):
    from sklearn.metrics import average_precision_score, roc_auc_score
    return (float(roc_auc_score(labels, scores)),
            float(average_precision_score(labels, scores)))


def micro(per):
    a = M.aggregate(per)
    return {k.replace("micro_", ""): round(a[k], 4)
            for k in ("micro_precision", "micro_recall", "micro_f1", "micro_iou")}


def fpr(pred, truth):
    return M.false_positive_rate(pred, truth)


# ===================================================================== main
def main():
    t_all = time.perf_counter()
    say("=" * 70)
    say("OSCD EXPERIMENT RUN")
    say("=" * 70)
    R = {"tile_px": TILE, "changed_tile_frac": CHANGED_TILE_FRAC}

    # ------------------------------------------------------ 1. load dataset
    say("\n[1] Loading dataset")
    cities = {}
    for c in TRAIN + TEST:
        try:
            cities[c] = load_city(c)
        except Exception as exc:                            # noqa: BLE001
            say(f"   !! {c}: {type(exc).__name__}: {exc}")
    inv = []
    say(f"   {'city':13s} {'split':5s} {'size (px)':11s} {'date 1':11s} {'date 2':11s} changed%")
    for c, d in cities.items():
        h, w = d["label"].shape
        pc = 100 * float((d["label"] > 0).mean())
        row = {"city": c, "split": "train" if c in TRAIN else "test",
               "height": h, "width": w, "date_1": d["dates"].get("date_1"),
               "date_2": d["dates"].get("date_2"), "changed_pct": round(pc, 2),
               "geo": d["geo"]}
        inv.append(row)
        say(f"   {c:13s} {row['split']:5s} {w}x{h:<7d} {str(row['date_1']):11s} "
            f"{str(row['date_2']):11s} {pc:5.2f}")
    R["inventory"] = inv
    tot_px = sum(r["height"] * r["width"] for r in inv)
    ch_px = sum((cities[r["city"]]["label"] > 0).sum() for r in inv)
    R["dataset"] = {"n_cities": len(inv), "total_pixels": int(tot_px),
                    "changed_pixels": int(ch_px),
                    "changed_pct": round(100 * ch_px / tot_px, 3)}
    say(f"   {len(inv)} cities, {tot_px:,} pixels, "
        f"{R['dataset']['changed_pct']}% of pixels labelled changed")
    say(f"   example geo metadata ({inv[0]['city']}): {inv[0]['geo']}")

    # ----------------------------------------------------------- 2. tiling
    say("\n[2] Tiling and metadata extraction")
    tiles, meta = [], []
    for c, d in cities.items():
        h, w = d["label"].shape
        for (y, x) in tile_grid(h, w):
            frac = float((d["label"][y:y + TILE, x:x + TILE] > 0).mean())
            for date in (1, 2):
                img = d[f"img{date}"][y:y + TILE, x:x + TILE]
                tiles.append(img)
                meta.append({"tile_id": f"{c}_d{date}_y{y}_x{x}", "city": c,
                             "split": "train" if c in TRAIN else "test",
                             "date_slot": date,
                             "date": d["dates"].get(f"date_{date}"),
                             "sensor": "Sentinel-2 MSI", "gsd_m": 10,
                             "x": x, "y": y, "tile_px": TILE,
                             "changed_frac": round(frac, 4),
                             **spectral_tags(d["bands"][date], y, x)})
    say(f"   {len(tiles)} tiles of {TILE}x{TILE} px "
        f"({len(tiles)//2} locations x 2 dates)")
    say("   metadata kept for one tile:")
    say("   " + json.dumps(meta[0]))
    R["n_tiles"] = len(tiles)
    R["example_metadata"] = meta[0]
    for k in QUERIES:
        R.setdefault("prevalence", {})[k] = round(
            float(np.mean([m[k] for m in meta])), 4)
    say(f"   share of tiles tagged by spectral index: {R['prevalence']}")

    # ------------------------------------------------- 3. embed and index
    say("\n[3] Embedding with frozen encoders and building the shared index")
    emb, idx, R["system"] = {}, {}, {}
    for backend in ("remoteclip", "openclip"):
        e = Embedder(backend=backend)
        t0 = time.perf_counter()
        v = np.concatenate([e.encode_images(tiles[i:i + 64])
                            for i in range(0, len(tiles), 64)])
        t_embed = time.perf_counter() - t0
        index = SharedIndex(v, meta, e.provenance())
        lat = index.timed_search(v[0], k=10)
        emb[backend], idx[backend] = (v, e), index
        R["system"][backend] = {**index.stats(), "embed_time_s": round(t_embed, 1),
                                "tiles_per_s": round(len(tiles) / t_embed, 1),
                                "query_latency_ms": round(lat, 3)}
        say(f"   {backend:10s} dim={v.shape[1]} embed={t_embed:6.1f}s "
            f"({len(tiles)/t_embed:5.1f} tiles/s) index={index.backend} "
            f"build={index.build_time_s*1000:.1f}ms store={index.storage_mb:.2f}MB "
            f"query={lat:.2f}ms")
    np.save(OUT / "embeddings_remoteclip.npy", emb["remoteclip"][0])

    # ------------------------------------------------- 4. text retrieval
    say("\n[4] Text retrieval (relevance = spectral-index tag)")
    R["retrieval"] = {}
    rng = np.random.default_rng(0)
    for backend in ("openclip", "remoteclip"):
        v, e = emb[backend]
        R["retrieval"][backend] = {}
        for key, text in QUERIES.items():
            q = e.encode_text(text)[0]
            order = np.argsort(-(v @ q))
            rel = {i for i, m in enumerate(meta) if m[key]}
            p5 = M.precision_at_k(order, rel, 5)
            p10 = M.precision_at_k(order, rel, 10)
            p20 = M.precision_at_k(order, rel, 20)
            ap = M.average_precision(order.tolist(), rel)
            R["retrieval"][backend][key] = {"p5": p5, "p10": p10, "p20": p20,
                                            "ap": round(ap, 4), "n_relevant": len(rel),
                                            "top10": [int(i) for i in order[:10]]}
            say(f"   {backend:10s} {key:10s} P@5={p5:.2f} P@10={p10:.2f} "
                f"P@20={p20:.2f} AP={ap:.3f}  (relevant tiles: {len(rel)})")
    R["retrieval"]["random"] = {}
    for key in QUERIES:
        rel = {i for i, m in enumerate(meta) if m[key]}
        aps, p10s = [], []
        for _ in range(20):
            order = rng.permutation(len(meta))
            aps.append(M.average_precision(order.tolist(), rel))
            p10s.append(M.precision_at_k(order, rel, 10))
        R["retrieval"]["random"][key] = {"p10": round(float(np.mean(p10s)), 3),
                                         "ap": round(float(np.mean(aps)), 4)}
        say(f"   {'random':10s} {key:10s} P@10={np.mean(p10s):.2f} AP={np.mean(aps):.3f}")
    for backend in ("random", "openclip", "remoteclip"):
        R["retrieval"][backend]["mAP"] = round(float(np.mean(
            [R["retrieval"][backend][k]["ap"] for k in QUERIES])), 4)
    say(f"   mAP: random={R['retrieval']['random']['mAP']} "
        f"openclip={R['retrieval']['openclip']['mAP']} "
        f"remoteclip={R['retrieval']['remoteclip']['mAP']}")

    # ---------------------------------- 5. same-place search across dates
    say("\n[5] Image search across dates (find the same place 2+ years later)")
    R["cross_date"] = {}
    i1 = np.array([i for i, m in enumerate(meta) if m["date_slot"] == 1])
    i2 = i1 + 1                                   # tiles were appended in pairs
    for backend in ("openclip", "remoteclip"):
        v = emb[backend][0]
        sims = v[i1] @ v[i2].T
        rank = (sims > np.diag(sims)[:, None]).sum(1)
        r = {f"recall@{k}": round(float((rank < k).mean()), 4) for k in (1, 5, 10)}
        R["cross_date"][backend] = r
        say(f"   {backend:10s} {r}   (random recall@1 = {1/len(i1):.4f})")
    R["cross_date"]["n_candidates"] = int(len(i1))

    # ------------------------- 6. shared-space change signal, tile level
    say("\n[6] Tile-level change signal from the SAME embeddings")
    y = np.array([meta[i]["changed_frac"] >= CHANGED_TILE_FRAC for i in i1])
    raw_diff, norm_diff = [], []
    for i in i1:
        a, b = tiles[i], tiles[i + 1]
        raw_diff.append(float(cv2.absdiff(a, b).mean()))
    for c, d in cities.items():                   # normalise per city, then tile
        n2 = cd.radiometric_normalise(d["img1"], d["img2"])
        h, w = d["label"].shape
        for (yy, xx) in tile_grid(h, w):
            norm_diff.append(float(cv2.absdiff(
                d["img1"][yy:yy + TILE, xx:xx + TILE],
                n2[yy:yy + TILE, xx:xx + TILE]).mean()))
    scores = {"pixel difference (raw)": np.array(raw_diff),
              "pixel difference (normalised)": np.array(norm_diff)}
    for backend in ("openclip", "remoteclip"):
        v = emb[backend][0]
        scores[f"embedding distance ({backend})"] = 1 - np.sum(v[i1] * v[i2], 1)
    R["tile_change"] = {"n_tiles": int(len(y)), "n_changed": int(y.sum()),
                        "changed_share": round(float(y.mean()), 4), "methods": {}}
    say(f"   {len(y)} locations, {int(y.sum())} changed "
        f"({100*y.mean():.1f}%) -> a random scorer gets AP = {y.mean():.3f}")
    for name, s in scores.items():
        auc, ap = roc_auc(s, y)
        R["tile_change"]["methods"][name] = {"auc": round(auc, 4), "ap": round(ap, 4)}
        say(f"   {name:36s} AUC={auc:.3f}  AP={ap:.3f}")
    np.save(OUT / "tile_scores.npy",
            np.vstack([y.astype(float)] + [scores[k] for k in scores]))
    R["tile_change"]["score_order"] = list(scores)

    # -------------------------------------- 7. pixel-level change, cities
    say("\n[7] Pixel-level change maps")
    grid = [20, 30, 40, 50, 60, 80]

    def run_ours(c, thr, sup):
        d = cities[c]
        return cd.analyse_pair(d["img1"], d["img2"], suppress=sup, threshold=thr)[0]

    best = {}
    for sup in (False, True):
        f1s = []
        for thr in grid:
            per = [M.change_metrics(run_ours(c, thr, sup), cities[c]["label"])
                   for c in TRAIN if c in cities]
            f1s.append(micro(per)["f1"])
        best[sup] = grid[int(np.argmax(f1s))]
        say(f"   threshold sweep on TRAIN (suppress={sup}): "
            + ", ".join(f"{t}:{f:.3f}" for t, f in zip(grid, f1s))
            + f"  -> picked {best[sup]}")
        R.setdefault("threshold_sweep", {})[str(sup)] = dict(zip(map(str, grid), f1s))
    R["picked_threshold"] = {"no_suppression": best[False], "suppression": best[True]}

    # embedding gate: drop pixel change inside tiles the embedding calls unchanged
    v = emb["remoteclip"][0]
    dist = 1 - np.sum(v[i1] * v[i2], 1)
    loc = {(meta[i]["city"], meta[i]["y"], meta[i]["x"]): k for k, i in enumerate(i1)}

    def gate(c, mask, tau):
        out = mask.copy()
        h, w = mask.shape
        for (yy, xx) in tile_grid(h, w):
            if dist[loc[(c, yy, xx)]] < tau:
                out[yy:yy + TILE, xx:xx + TILE] = 0
        out[(h // TILE) * TILE:, :] = 0
        out[:, (w // TILE) * TILE:] = 0
        return out

    taus = np.round(np.quantile(dist, [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]), 4)
    sup_train = {c: run_ours(c, best[True], True) for c in TRAIN if c in cities}
    f1s = [micro([M.change_metrics(gate(c, m, t), cities[c]["label"])
                  for c, m in sup_train.items()])["f1"] for t in taus]
    tau = float(taus[int(np.argmax(f1s))])
    say("   embedding-gate sweep on TRAIN: "
        + ", ".join(f"{t:.3f}:{f:.3f}" for t, f in zip(taus, f1s)) + f"  -> picked {tau:.3f}")
    R["picked_tau"] = tau

    methods = {
        "Pixel difference + Otsu": lambda c: cd.baseline_pixel_otsu(
            cities[c]["img1"], cities[c]["img2"]),
        "Change Vector Analysis": lambda c: cd.baseline_cva(
            cities[c]["img1"], cities[c]["img2"]),
        "SSIM differencing": lambda c: cd.baseline_ssim(
            cities[c]["img1"], cities[c]["img2"]),
        "Ours, no suppression": lambda c: run_ours(c, best[False], False),
        "Ours, with suppression": lambda c: run_ours(c, best[True], True),
        "Ours, suppression + embedding gate": lambda c: gate(
            c, run_ours(c, best[True], True), tau),
    }
    R["pixel"] = {}
    masks_test = {}
    for split, names in (("test", TEST), ("train", TRAIN)):
        say(f"   --- {split.upper()} cities ---")
        say(f"   {'method':36s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'IoU':>6s} {'FPR%':>6s}")
        R["pixel"][split] = {}
        for name, fn in methods.items():
            per, fprs, t0 = [], [], time.perf_counter()
            per_city = {}
            for c in names:
                if c not in cities:
                    continue
                m = fn(c)
                cm = M.change_metrics(m, cities[c]["label"])
                per.append(cm)
                fprs.append(fpr(m, cities[c]["label"]))
                per_city[c] = {k: round(cm[k], 4) for k in ("precision", "recall", "f1", "iou")}
                if split == "test":
                    masks_test.setdefault(name, {})[c] = m
            r = micro(per)
            fp = sum(p["fp"] for p in per)
            neg = sum((cities[c]["label"] == 0).sum() for c in names if c in cities)
            r["fpr_pct"] = round(100 * fp / neg, 3)
            r["time_s"] = round(time.perf_counter() - t0, 2)
            r["per_city"] = per_city
            R["pixel"][split][name] = r
            say(f"   {name:36s} {r['precision']:6.3f} {r['recall']:6.3f} "
                f"{r['f1']:6.3f} {r['iou']:6.3f} {r['fpr_pct']:6.2f}")

    # --------------------------------------- 8. Gate C on real imagery
    say("\n[8] Gate C on real data (TEST cities)")
    a = R["pixel"]["test"]["Ours, no suppression"]
    b = R["pixel"]["test"]["Ours, with suppression"]
    kept = b["recall"] / a["recall"] if a["recall"] else 0.0
    R["gate_c"] = {"fpr_no_suppression_pct": a["fpr_pct"],
                   "fpr_suppression_pct": b["fpr_pct"],
                   "recall_no_suppression": a["recall"],
                   "recall_suppression": b["recall"],
                   "recall_retained": round(kept, 4),
                   "false_alarm_half": b["fpr_pct"] < a["fpr_pct"],
                   "sensitivity_half": kept >= 0.8}
    say(f"   false-alarm rate: {a['fpr_pct']}% -> {b['fpr_pct']}%   "
        f"({'PASS' if R['gate_c']['false_alarm_half'] else 'FAIL'})")
    say(f"   recall kept     : {a['recall']} -> {b['recall']} = {100*kept:.0f}%   "
        f"({'PASS' if R['gate_c']['sensitivity_half'] else 'FAIL'}, need 80%)")

    R["total_runtime_s"] = round(time.perf_counter() - t_all, 1)
    say(f"\nTotal runtime: {R['total_runtime_s']} s")
    (OUT / "results.json").write_text(json.dumps(R, indent=2, default=str))
    say(f"Saved {OUT / 'results.json'}")

    # keep what the figure script needs
    np.savez_compressed(
        OUT / "masks_test.npz",
        **{f"{n}||{c}": m for n, d in masks_test.items() for c, m in d.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
