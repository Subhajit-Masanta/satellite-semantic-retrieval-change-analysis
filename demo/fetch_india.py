"""Download Sentinel-2 image pairs of Indian sites (no account needed).

    python demo/fetch_india.py

Source: Copernicus Sentinel-2 L2A, read from the public cloud-optimised GeoTIFF
mirror on AWS (found through the Element84 Earth Search catalogue). Only the
small window around each site is downloaded, not the whole 110 km scene.

Each site gets data/india/<site>/t1.tif and t2.tif (4 bands: blue, green, red,
near infrared, 10 m, georeferenced) plus meta.json with scene ids and dates.
"""

import json
import sys
from pathlib import Path

import numpy as np
import requests
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "india"
STAC = "https://earth-search.aws.element84.com/v1/search"
BANDS = ["blue", "green", "red", "nir"]
HALF_DEG = 0.035                      # about 7.5 km box
T1 = "2019-01-01T00:00:00Z/2019-03-31T23:59:59Z"
T2 = "2025-01-01T00:00:00Z/2025-03-31T23:59:59Z"

SITES = {
    "jewar_airport_up": ("Noida International Airport, Jewar, Uttar Pradesh", 77.610, 28.170),
    "navi_mumbai_airport": ("Navi Mumbai International Airport, Maharashtra", 73.070, 18.990),
    "amaravati_ap": ("Amaravati capital region, Andhra Pradesh", 80.510, 16.510),
    "dholera_gujarat": ("Dholera Special Investment Region, Gujarat", 72.190, 22.250),
    "hyderabad_west": ("Hyderabad west (Kokapet / Financial District), Telangana", 78.330, 17.400),
    "bengaluru_north": ("Bengaluru north (Devanahalli), Karnataka", 77.680, 13.220),
    "ayodhya_up": ("Ayodhya airport area, Uttar Pradesh", 82.150, 26.750),
    "gift_city_gujarat": ("GIFT City, Gandhinagar, Gujarat", 72.680, 23.160),
    "vit_vellore": ("VIT Vellore campus and Katpadi, Tamil Nadu", 79.1559, 12.9692),
    "vellore_city": ("Vellore city and fort, Tamil Nadu", 79.1325, 12.9165),
}


def search(bbox, when, tile=None):
    body = {"collections": ["sentinel-2-l2a"], "bbox": bbox, "datetime": when,
            "query": {"eo:cloud_cover": {"lt": 10}}, "limit": 50}
    feats = requests.post(STAC, json=body, timeout=60).json().get("features", [])
    if tile:
        feats = [f for f in feats if f["properties"].get("grid:code") == tile]
    return sorted(feats, key=lambda f: f["properties"].get("eo:cloud_cover", 99))


def read_window(feat, bounds_utm):
    arrs, prof = [], None
    for b in BANDS:
        with rasterio.open(feat["assets"][b]["href"]) as ds:
            win = from_bounds(*bounds_utm, transform=ds.transform).round_offsets().round_lengths()
            arrs.append(ds.read(1, window=win))
            if prof is None:
                prof = {"crs": ds.crs, "transform": ds.window_transform(win)}
    h = min(a.shape[0] for a in arrs); w = min(a.shape[1] for a in arrs)
    return np.stack([a[:h, :w] for a in arrs]), prof


def main():
    env = rasterio.Env(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                       GDAL_HTTP_MAX_RETRY="3", GDAL_HTTP_RETRY_DELAY="1")
    with env:
        for key, (name, lon, lat) in SITES.items():
            d = OUT / key
            if (d / "meta.json").exists():
                print(f"{key}: already there"); continue
            bbox = [lon - HALF_DEG, lat - HALF_DEG, lon + HALF_DEG, lat + HALF_DEG]
            try:
                f1s = search(bbox, T1)
                if not f1s:
                    print(f"{key}: no clear scene for date 1"); continue
                ok = False
                for f1 in f1s:                      # try scenes until the box fits fully
                    tile = f1["properties"].get("grid:code")
                    f2s = search(bbox, T2, tile)
                    if not f2s:
                        continue
                    f2 = f2s[0]
                    with rasterio.open(f1["assets"]["red"]["href"]) as ds:
                        b = transform_bounds("EPSG:4326", ds.crs, *bbox)
                        b = tuple(round(v / 10) * 10 for v in b)       # snap to the 10 m grid
                        full = (b[0] >= ds.bounds.left and b[2] <= ds.bounds.right
                                and b[1] >= ds.bounds.bottom and b[3] <= ds.bounds.top)
                    if not full:
                        continue
                    a1, prof = read_window(f1, b)
                    a2, _ = read_window(f2, b)
                    ok = True
                    break
                if not ok:
                    print(f"{key}: no scene pair covers the whole box"); continue
                h, w = min(a1.shape[1], a2.shape[1]), min(a1.shape[2], a2.shape[2])
                a1, a2 = a1[:, :h, :w], a2[:, :h, :w]
                if (a1 == 0).mean() > 0.05 or (a2 == 0).mean() > 0.05:
                    print(f"{key}: window falls off the scene edge, skipped"); continue
                d.mkdir(parents=True, exist_ok=True)
                for nm, arr in (("t1", a1), ("t2", a2)):
                    with rasterio.open(d / f"{nm}.tif", "w", driver="GTiff", height=h, width=w,
                                       count=4, dtype="uint16", compress="deflate", **prof) as out:
                        out.write(arr.astype("uint16"))
                        out.descriptions = tuple(BANDS)
                meta = {"site": key, "name": name, "lon": lon, "lat": lat, "bbox_lonlat": bbox,
                        "crs": str(prof["crs"]), "bounds_utm": b, "height": h, "width": w,
                        "gsd_m": 10, "sensor": "Sentinel-2 MSI (L2A)", "bands": BANDS,
                        "source": "Copernicus Sentinel-2, AWS open data COG mirror",
                        "t1": {"scene": f1["id"], "date": f1["properties"]["datetime"][:10],
                               "cloud_pct": f1["properties"].get("eo:cloud_cover")},
                        "t2": {"scene": f2["id"], "date": f2["properties"]["datetime"][:10],
                               "cloud_pct": f2["properties"].get("eo:cloud_cover")}}
                (d / "meta.json").write_text(json.dumps(meta, indent=2))
                print(f"{key}: {w}x{h} px  {meta['t1']['date']} -> {meta['t2']['date']}  "
                      f"({f1['id']}, {f2['id']})")
            except Exception as exc:                          # noqa: BLE001
                print(f"{key}: FAILED {type(exc).__name__}: {str(exc)[:120]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
