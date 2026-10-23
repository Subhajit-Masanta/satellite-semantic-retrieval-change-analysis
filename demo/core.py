"""Shared code for the demo: loading sites, building the index, searching.

One frozen CLIP model (ViT-B/32) gives, for every 64 px tile,
  * one embedding  -> used for text search and image search
  * a 7x7 grid of patch features -> date 1 vs date 2 gives the change heat map
Both come out of the same forward pass and are stored in one index.
"""

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "demo_index"
TILE = 64            # Sentinel-2 sites (10 m per pixel)
LEVIR_TILE = 256     # LEVIR-CD sites (0.5 m per pixel)
LEVIR_N = 14         # how many LEVIR-CD pairs to show in the demo
SYSU_DIR = ROOT / "data" / "SYSU-CD"   # 256 px pairs at 0.5 m, same tile size as LEVIR-CD
WEB_DIR = ROOT / "demo_data"           # small copies for the website (demo/make_web_data.py)

# OSCD folder name -> (place, country). OSCD tiles also get real GPS from the
# original georeferenced bands; LEVIR-CD and SYSU-CD publish no coordinates.
OSCD_PLACES = {
    "abudhabi": ("Abu Dhabi", "UAE"), "aguasclaras": ("Aguas Claras, Brasilia", "Brazil"),
    "beihai": ("Beihai", "China"), "beirut": ("Beirut", "Lebanon"),
    "bercy": ("Bercy, Paris", "France"), "bordeaux": ("Bordeaux", "France"),
    "brasilia": ("Brasilia", "Brazil"), "chongqing": ("Chongqing", "China"),
    "cupertino": ("Cupertino, California", "USA"), "dubai": ("Dubai", "UAE"),
    "hongkong": ("Hong Kong", "China"), "lasvegas": ("Las Vegas, Nevada", "USA"),
    "milano": ("Milan", "Italy"), "montpellier": ("Montpellier", "France"),
    "mumbai": ("Mumbai", "India"), "nantes": ("Nantes", "France"),
    "norcia": ("Norcia", "Italy"), "paris": ("Paris", "France"), "pisa": ("Pisa", "Italy"),
    "rennes": ("Rennes", "France"), "rio": ("Rio de Janeiro", "Brazil"),
    "saclay_e": ("Saclay east, Paris region", "France"),
    "saclay_w": ("Saclay west, Paris region", "France"), "valencia": ("Valencia", "Spain"),
}


# Collections that publish no coordinates: one approximate marker per region.
REGION_CENTRES = {
    "LEVIR-CD (0.5 m)": ("Austin area, Texas", "USA", 30.27, -97.74),
    "SYSU-CD (0.5 m)": ("Hong Kong", "China", 22.32, 114.17),
}


def res_label(gsd):
    """Resolution as shown in the app, e.g. '10 m' or '0.5 m'."""
    return f"{gsd:g} m"
MODEL_NAME, MODEL_TAG = "ViT-B-32", "laion2b_s34b_b79k"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


# ------------------------------------------------------------------ sites
def _stretch(a, b):
    """Same 2-98% stretch for both dates so real brightness differences stay."""
    out1, out2 = [], []
    for i in range(a.shape[0]):
        both = np.concatenate([a[i].ravel(), b[i].ravel()])
        lo, hi = np.percentile(both[both > 0] if (both > 0).any() else both, [2, 98])
        f = lambda x: np.clip((x - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        out1.append(f(a[i].astype(np.float32))); out2.append(f(b[i].astype(np.float32)))
    return np.dstack(out1), np.dstack(out2)


def list_sites():
    """Every site folder that is on disk, India first then OSCD cities."""
    sites = []
    sentinel = [(d, "India (Sentinel-2)") for d in sorted((ROOT / "data" / "india").glob("*/meta.json"))]
    sentinel += [(d, "World (Sentinel-2)") for d in sorted((ROOT / "data" / "world").glob("*/meta.json"))]
    for d, coll in sentinel:
        m = json.loads(d.read_text())
        sites.append({"key": m["site"], "name": m["name"], "collection": coll,
                      "dir": str(d.parent), "date1": m["t1"]["date"], "date2": m["t2"]["date"],
                      "scene1": m["t1"]["scene"], "scene2": m["t2"]["scene"], "meta": m,
                      "tile": TILE, "gsd": 10.0, "sensor": "Sentinel-2 MSI",
                      "place": m["name"], "country": m.get("country", "India"), "gps": True})
    for d in sorted((ROOT / "data" / "isro").glob("*/meta.json")):
        # bands are green, red, NIR: shown as B,G,R this is standard false colour
        m = json.loads(d.read_text())
        sites.append({"key": m["site"], "name": m["name"], "collection": "ISRO LISS-IV (5 m)",
                      "dir": str(d.parent), "date1": m["t1"]["date"], "date2": m["t2"]["date"],
                      "scene1": m["t1"]["scene"], "scene2": m["t2"]["scene"], "meta": m,
                      "tile": TILE, "gsd": float(m["gsd_m"]), "sensor": m["sensor"],
                      "place": m["name"], "country": "India", "gps": True})
    # full OSCD in data/ (local), or the 6-city copy in demo_data/ (website)
    seen = set()
    oscd = [d for base in (ROOT / "data" / "OSCD", WEB_DIR / "OSCD") for d in sorted(base.rglob("imgs_1_rect"))]
    for d in oscd:
        city = d.parent
        if city.name in seen:
            continue
        seen.add(city.name)
        dates = {}
        if (city / "dates.txt").exists():
            for line in (city / "dates.txt").read_text().splitlines():
                if ":" in line:
                    k, v = [x.strip() for x in line.split(":", 1)]
                    dates[k] = f"{v[:4]}-{v[4:6]}-{v[6:8]}" if len(v) == 8 else v
        place, country = OSCD_PLACES.get(city.name, (city.name, "?"))
        sites.append({"key": "oscd_" + city.name, "name": f"{place} (OSCD)",
                      "collection": "OSCD (Sentinel-2)", "dir": str(city),
                      "date1": dates.get("date_1", "?"), "date2": dates.get("date_2", "?"),
                      "scene1": city.name + "/imgs_1_rect", "scene2": city.name + "/imgs_2_rect",
                      "meta": None, "tile": TILE, "gsd": 10.0, "sensor": "Sentinel-2 MSI",
                      "place": place, "country": country, "gps": True})
    for name in levir_pick():
        sites.append({"key": "levir_" + name, "name": f"LEVIR-CD {name.replace('_', ' ')}",
                      "collection": "LEVIR-CD (0.5 m)", "dir": str(levir_root()),
                      "date1": "earlier", "date2": "later", "scene1": f"A/{name}.png",
                      "scene2": f"B/{name}.png", "meta": None, "file": name + ".png",
                      "tile": LEVIR_TILE, "gsd": 0.5, "sensor": "Google Earth (LEVIR-CD)",
                      "place": "Austin area, Texas (exact spot not published)", "country": "USA",
                      "gps": False})
    for name in sysu_pick():
        sites.append({"key": "sysu_" + name, "name": f"SYSU-CD Hong Kong {name.replace('_', ' ')}",
                      "collection": "SYSU-CD (0.5 m)", "dir": str(SYSU_DIR),
                      "date1": "2007", "date2": "2014", "scene1": f"A/{name}.png",
                      "scene2": f"B/{name}.png", "meta": None, "file": name + ".png",
                      "tile": LEVIR_TILE, "gsd": 0.5, "sensor": "Aerial (SYSU-CD)",
                      "place": "Hong Kong (exact spot not published)", "country": "China",
                      "gps": False})
    return sites


def sysu_pick():
    """SYSU-CD test pairs written by experiments/run_sysu.py (a fixed choice
    by label, not by score)."""
    keep = SYSU_DIR / "demo_pick.json"
    return json.loads(keep.read_text()) if keep.exists() else []


def levir_root():
    for a in (ROOT / "data" / "LEVIR-CD").rglob("A"):
        if a.is_dir():
            return a.parent
    return None


def levir_pick():
    """A fixed choice of LEVIR-CD test pairs for the demo: the ones with the
    most labelled change, plus a few with little or none so the demo also shows
    what 'nothing changed' looks like. Saved so the choice never moves."""
    root = levir_root()
    if root is None:
        return []
    keep = root / "demo_pick.json"
    if keep.exists():
        return json.loads(keep.read_text())
    frac = []
    for f in sorted((root / "label").glob("test_*.png")):
        l = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        frac.append((float((l > 127).mean()), f.stem))
    frac.sort(reverse=True)
    quiet = [n for fr, n in frac if fr < 0.002]
    pick = [n for _, n in frac[:LEVIR_N - 3]] + quiet[:3]
    keep.write_text(json.dumps(pick))
    return pick


def load_site(site):
    """Returns dict: img1, img2 (uint8 BGR), b1, b2 (float bands B,G,R,NIR),
    label (or None), geo (rasterio transform + crs, or None)."""
    d = Path(site["dir"])
    geo, label = None, None
    if site.get("file"):                      # PNG pairs: LEVIR-CD and SYSU-CD
        i1 = cv2.imread(str(d / "A" / site["file"]))
        i2 = cv2.imread(str(d / "B" / site["file"]))
        l = cv2.imread(str(d / "label" / site["file"]), cv2.IMREAD_GRAYSCALE)
        return {"img1": i1, "img2": i2,
                "b1": np.transpose(i1, (2, 0, 1)).astype(np.float32),
                "b2": np.transpose(i2, (2, 0, 1)).astype(np.float32),
                "label": (l > 127).astype(np.uint8) * 255, "geo": None}
    if site["collection"].startswith(("India", "World", "ISRO")):
        import rasterio
        with rasterio.open(d / "t1.tif") as ds:
            a = ds.read().astype(np.float32)
            geo = {"transform": ds.transform, "crs": str(ds.crs)}
        with rasterio.open(d / "t2.tif") as ds:
            b = ds.read().astype(np.float32)
    else:
        def rd(date):
            arrs = [cv2.imread(str(d / f"imgs_{date}_rect" / f"{bn}.tif"), cv2.IMREAD_UNCHANGED)
                    for bn in ("B02", "B03", "B04", "B08")]
            h = min(x.shape[0] for x in arrs); w = min(x.shape[1] for x in arrs)
            return np.stack([x[:h, :w] for x in arrs]).astype(np.float32)
        a, b = rd(1), rd(2)
        geo = _oscd_geo(d, a.shape[1:])
        cms = [d / "cm" / "cm.png"] + list((ROOT / "data" / "OSCD").rglob("cm.png"))
        for p in cms:
            if p.exists() and p.parent.parent.name == d.name:
                label = (cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8) * 255
                break
    h, w = min(a.shape[1], b.shape[1]), min(a.shape[2], b.shape[2])
    a, b = a[:, :h, :w], b[:, :h, :w]
    if label is not None:
        label = label[:h, :w]
    s1, s2 = _stretch(a[:3], b[:3])           # B,G,R -> already BGR order
    return {"img1": s1, "img2": s2, "b1": a, "b2": b, "label": label, "geo": geo}


def _oscd_geo(city_dir, shape):
    """The *_rect bands have no coordinates, but the original imgs_1 bands of
    the same size do. Returns the geo dict, or None if no size match."""
    try:
        import rasterio
        for f in sorted((Path(city_dir) / "imgs_1").glob("*B02.tif")):
            with rasterio.open(f) as ds:
                if ds.crs and (ds.height, ds.width) == tuple(shape):
                    return {"transform": ds.transform, "crs": str(ds.crs)}
    except Exception:                                    # noqa: BLE001
        pass
    return None


def tile_grid(h, w, tile=TILE):
    return [(y, x) for y in range(0, h - tile + 1, tile) for x in range(0, w - tile + 1, tile)]


def tile_lonlat(geo, y, x, tile=TILE):
    """Longitude / latitude of a tile centre (India sites only)."""
    if not geo:
        return None
    from rasterio.warp import transform
    X, Y = geo["transform"] * (x + tile / 2, y + tile / 2)
    lon, lat = transform(geo["crs"], "EPSG:4326", [X], [Y])
    return round(float(lon[0]), 5), round(float(lat[0]), 5)


# ------------------------------------------------------------------ model
class Clip:
    def __init__(self):
        import open_clip
        import torch
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=MODEL_TAG)
        self.model = model.to(self.dev).eval()
        self.model.visual.output_tokens = True
        self.tok = open_clip.get_tokenizer(MODEL_NAME)

    def _prep(self, batch):
        torch = self.torch
        x = torch.from_numpy(np.ascontiguousarray(batch[..., ::-1])).permute(0, 3, 1, 2)
        x = x.float().div(255).to(self.dev)
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bicubic",
                                            align_corners=False).clamp(0, 1)
        m = torch.tensor(CLIP_MEAN, device=self.dev).view(1, 3, 1, 1)
        s = torch.tensor(CLIP_STD, device=self.dev).view(1, 3, 1, 1)
        return (x - m) / s

    def images(self, batch):
        """-> (embeddings (n,512) unit length, patch tokens (n,49,width) unit length)"""
        torch = self.torch
        with torch.no_grad(), torch.autocast(self.dev, enabled=(self.dev == "cuda")):
            pooled, tokens = self.model.visual(self._prep(batch))
        pooled, tokens = pooled.float(), tokens.float()
        pooled = pooled / pooled.norm(dim=-1, keepdim=True)
        tokens = tokens / tokens.norm(dim=-1, keepdim=True)
        return pooled.cpu().numpy(), tokens.cpu().numpy()

    def text(self, query):
        torch = self.torch
        with torch.no_grad():
            t = self.model.encode_text(self.tok([query]).to(self.dev)).float()
        t = t / t.norm(dim=-1, keepdim=True)
        return t.cpu().numpy()[0]


# ------------------------------------------------------------------ index
def build_index(progress=print):
    clip = Clip()
    sites = list_sites()
    vecs, meta, heats, loc_meta = [], [], [], []
    for si, site in enumerate(sites):
        s = load_site(site)
        h, w = s["img1"].shape[:2]
        T = site["tile"]
        grid = tile_grid(h, w, T)
        t1 = np.stack([s["img1"][y:y + T, x:x + T] for y, x in grid])
        t2 = np.stack([s["img2"][y:y + T, x:x + T] for y, x in grid])
        e1, k1, e2, k2 = [], [], [], []
        for i in range(0, len(grid), 64):
            a, b = clip.images(t1[i:i + 64]); e1.append(a); k1.append(b)
            a, b = clip.images(t2[i:i + 64]); e2.append(a); k2.append(b)
        e1, e2 = np.concatenate(e1), np.concatenate(e2)
        k1, k2 = np.concatenate(k1), np.concatenate(k2)
        heat = 1 - (k1 * k2).sum(-1)                       # (n, 49) patch change
        g = int(round(np.sqrt(heat.shape[1])))
        top = np.sort(heat, axis=1)[:, -max(1, heat.shape[1] // 10):].mean(1)
        for j, (y, x) in enumerate(grid):
            ll = tile_lonlat(s["geo"], y, x, T)
            for slot, e, date, scene in ((1, e1, site["date1"], site["scene1"]),
                                         (2, e2, site["date2"], site["scene2"])):
                vecs.append(e[j])
                meta.append({"site": site["key"], "site_name": site["name"],
                             "place": site["place"], "country": site["country"],
                             "collection": site["collection"], "slot": slot, "date": date,
                             "scene": scene, "y": y, "x": x, "loc": len(loc_meta),
                             "lonlat": ll, "sensor": site["sensor"], "gsd_m": site["gsd"],
                             "tile": T})
            loc_meta.append({"site": site["key"], "y": y, "x": x, "lonlat": ll, "tile": T,
                             "embedding_distance": float(1 - e1[j] @ e2[j]),
                             "patch_score": float(top[j])})
            heats.append(heat[j].reshape(g, g))
        progress(f"[{si + 1}/{len(sites)}] {site['name']}: {len(grid)} tiles x 2 dates")
    INDEX_DIR.mkdir(exist_ok=True)
    n_eurosat = build_eurosat(clip, progress)
    np.save(INDEX_DIR / "vectors.npy", np.stack(vecs).astype(np.float32))
    np.save(INDEX_DIR / "heat.npy", np.stack(heats).astype(np.float32))
    (INDEX_DIR / "meta.json").write_text(json.dumps(meta))
    (INDEX_DIR / "locations.json").write_text(json.dumps(loc_meta))
    (INDEX_DIR / "model.json").write_text(json.dumps({
        "model": f"OpenCLIP {MODEL_NAME} ({MODEL_TAG})", "licence": "MIT",
        "source": "https://github.com/mlfoundations/open_clip",
        "tile_px": TILE, "levir_tile_px": LEVIR_TILE, "n_vectors": len(vecs),
        "n_sites": len(sites), "n_eurosat_patches": n_eurosat}, indent=2))
    progress(f"index saved: {len(vecs)} tile vectors, {len(loc_meta)} places, {len(sites)} sites")
    return len(vecs)


def load_index():
    v = np.load(INDEX_DIR / "vectors.npy")
    heat = np.load(INDEX_DIR / "heat.npy")
    meta = json.loads((INDEX_DIR / "meta.json").read_text())
    locs = json.loads((INDEX_DIR / "locations.json").read_text())
    info = json.loads((INDEX_DIR / "model.json").read_text())
    return v, heat, meta, locs, info


# ------------------------------------------------------------- EuroSAT
EUROSAT_DIRS = (ROOT / "data" / "EuroSAT" / "2750", WEB_DIR / "EuroSAT")
# the prompts after tuning on half of EuroSAT (results/eurosat_prompt_tuning_results.json)
EUROSAT_PROMPTS = {
    "AnnualCrop": "annual crop land", "Forest": "a forest",
    "HerbaceousVegetation": "scrubland and grass", "Highway": "a highway or road",
    "Industrial": "industrial buildings", "Pasture": "pasture land",
    "PermanentCrop": "permanent crop land", "Residential": "residential buildings",
    "River": "a river", "SeaLake": "deep blue water, sea or lake",
}


def eurosat_files():
    """(base dir, [(relative path, class)]) from the full set if it is on disk,
    else from the website copy."""
    for base in EUROSAT_DIRS:
        if base.is_dir():
            return base, [(f"{p.parent.name}/{p.name}", p.parent.name)
                          for p in sorted(base.glob("*/*.jpg"))]
    return None, []


def build_eurosat(clip, progress=print):
    base, files = eurosat_files()
    if not files:
        return 0
    vecs = []
    for i in range(0, len(files), 256):
        batch = np.stack([cv2.imread(str(base / f)) for f, _ in files[i:i + 256]])
        vecs.append(clip.images(batch)[0])
    np.save(INDEX_DIR / "eurosat_vectors.npy", np.concatenate(vecs).astype(np.float16))
    (INDEX_DIR / "eurosat_meta.json").write_text(json.dumps(files))
    progress(f"EuroSAT: {len(files)} labelled patches")
    return len(files)


def load_eurosat():
    """-> (vectors, [(path, class)], base dir) for the patches whose image is
    on disk, or None."""
    f = INDEX_DIR / "eurosat_vectors.npy"
    base, _ = eurosat_files()
    if not f.exists() or base is None:
        return None
    v = np.load(f).astype(np.float32)
    files = json.loads((INDEX_DIR / "eurosat_meta.json").read_text())
    keep = [i for i, (p, _) in enumerate(files) if (base / p).exists()]
    return v[keep], [files[i] for i in keep], base


def search(vectors, query_vec, keep, k):
    """keep = boolean mask of rows allowed by the filters."""
    scores = vectors @ query_vec
    scores = np.where(keep, scores, -np.inf)
    order = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in order if np.isfinite(scores[i])]


def lonlat_to_px(geo, lon, lat):
    """Inverse of tile_lonlat: image column, row of a lon/lat point."""
    from rasterio.warp import transform
    X, Y = transform("EPSG:4326", geo["crs"], [lon], [lat])
    col, row = ~geo["transform"] * (X[0], Y[0])
    return int(round(col)), int(round(row))


def coverage(locs):
    """Per site with GPS: the lat/lon box of its tile centres and its centre."""
    box = {}
    for l in locs:
        if l.get("lonlat"):
            lon, lat = l["lonlat"]
            b = box.setdefault(l["site"], [lat, lat, lon, lon])
            b[0], b[1] = min(b[0], lat), max(b[1], lat)
            b[2], b[3] = min(b[2], lon), max(b[3], lon)
    return {k: {"lat_min": b[0], "lat_max": b[1], "lon_min": b[2], "lon_max": b[3],
                "lat": (b[0] + b[1]) / 2, "lon": (b[2] + b[3]) / 2} for k, b in box.items()}


def gps_lookup(locs, lat, lon, radius_km):
    """Tiles whose centre lies within radius_km of (lat, lon), nearest first,
    plus the nearest tile overall as (loc index, km)."""
    pts = [i for i, l in enumerate(locs) if l.get("lonlat")]
    if not pts:
        return [], None
    ll = np.array([locs[i]["lonlat"] for i in pts], np.float64)       # lon, lat
    p1, p2 = np.radians(lat), np.radians(ll[:, 1])
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(ll[:, 0] - lon) / 2) ** 2)
    km = 2 * 6371.0 * np.arcsin(np.sqrt(a))
    order = np.argsort(km)
    return ([(pts[j], float(km[j])) for j in order if km[j] <= radius_km],
            (pts[order[0]], float(km[order[0]])))


# ---------------------------------------------------------------- change
def brightness_match(a, b):
    """Match every band of date 2 to the mean and spread of date 1."""
    m1, s1 = a.mean((1, 2), keepdims=True), a.std((1, 2), keepdims=True) + 1e-6
    m2, s2 = b.mean((1, 2), keepdims=True), b.std((1, 2), keepdims=True) + 1e-6
    return (b - m2) / s2 * s1 + m1, s1


def pixel_change(b1, b2, top_percent=3.0):
    """Multi-band difference after brightness matching; keeps the strongest
    top_percent of pixels (the setting that did best in our OSCD tests)."""
    bn, s1 = brightness_match(b1, b2)
    mag = np.sqrt((((b1 - bn) / s1) ** 2).sum(0))
    mask = (mag >= np.percentile(mag, 100 - top_percent)).astype(np.uint8) * 255
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)), mag


def site_heat(heat, locs, site_key, shape):
    """Stitch the per-tile patch heat maps into one map for the whole site."""
    out = np.zeros(shape, np.float32)
    for i, l in enumerate(locs):
        if l["site"] == site_key:
            T = l.get("tile", TILE)
            out[l["y"]:l["y"] + T, l["x"]:l["x"] + T] = cv2.resize(
                heat[i], (T, T), interpolation=cv2.INTER_LINEAR)
    return out
