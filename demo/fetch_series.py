"""Download a YEARLY image series for every demo site (no account needed).

    run.bat demo\\fetch_series.py              (all sites)
    run.bat demo\\fetch_series.py key1 key2    (only these sites)

For each site and each year 2019-2025 we take up to three clear Sentinel-2
scenes from the dry season (January to April) and combine them into one
composite: for every pixel, the median of the scenes where that pixel is not
cloud or cloud shadow. One odd scene (haze, a passing cloud, a wet field) can
then not show up as change.

Output: data/india/<site>/series/<year>.tif with 6 bands
        blue, green, red, nir, swir16, n_valid   (reflectance x 10000, 10 m)
plus    data/india/<site>/series/series.json with the scenes used.
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parents[1]
SITE_DIRS = (ROOT / "data" / "india", ROOT / "data" / "world")
STAC = "https://earth-search.aws.element84.com/v1/search"
YEARS = list(range(2019, 2026))
PER_YEAR = 3
BANDS = ["blue", "green", "red", "nir", "swir16"]
BAD_SCL = {0, 1, 3, 8, 9, 10, 11}     # no data, saturated, shadow, clouds, cirrus, snow
ENV = dict(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
           GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="1")


def scenes_for(meta, year):
    grid = "MGRS-" + meta["t1"]["scene"].split("_")[1]
    # the site's own season (fetch_india.py), else January to April
    m0, m1 = meta.get("months", ["01-01", "04-30"])
    body = {"collections": ["sentinel-2-l2a"], "bbox": meta["bbox_lonlat"],
            "datetime": f"{year}-{m0}T00:00:00Z/{year}-{m1}T23:59:59Z",
            "query": {"eo:cloud_cover": {"lt": 20}, "grid:code": {"eq": grid}}, "limit": 100}
    feats = requests.post(STAC, json=body, timeout=90).json().get("features", [])
    feats.sort(key=lambda f: f["properties"].get("eo:cloud_cover", 99))
    out, seen = [], set()
    for f in feats:                                  # one scene per calendar day
        day = f["properties"]["datetime"][:10]
        if day not in seen:
            seen.add(day); out.append(f)
        if len(out) == PER_YEAR:
            break
    return out


def read(feat, band, bounds, shape, nearest=False):
    with rasterio.open(feat["assets"][band]["href"]) as ds:
        win = from_bounds(*bounds, transform=ds.transform)
        return ds.read(1, window=win, out_shape=shape,
                       resampling=Resampling.nearest if nearest else Resampling.bilinear)


def offset_of(feat, raw_blue, ok):
    """Scenes processed since 2022 can carry a +1000 offset. The catalogue flag
    for this turned out to be unreliable, so we decide from the pixels: with
    the offset in place no value can be below about 1000, without it the dark
    pixels of the blue band sit at a few hundred."""
    try:
        new = float(str(feat["properties"].get("s2:processing_baseline", "0"))) >= 4.0
    except ValueError:
        new = False
    if not new or not ok.any():
        return 0
    return 1000 if float(np.percentile(raw_blue[ok], 0.5)) >= 950 else 0


def one(job):
    site_dir, meta, year = job
    out = site_dir / "series" / f"{year}.tif"
    if out.exists():
        return f"{meta['site']} {year}: already there"
    try:
        with rasterio.Env(**ENV):
            feats = scenes_for(meta, year)
            if not feats:
                return f"{meta['site']} {year}: no clear scene"
            shape = (meta["height"], meta["width"])
            b = tuple(meta["bounds_utm"])
            stack, used = [], []
            for f in feats:
                scl = read(f, "scl", b, shape, nearest=True)
                ok = ~np.isin(scl, list(BAD_SCL))
                arr = np.stack([read(f, bn, b, shape).astype(np.float32) for bn in BANDS])
                ok &= arr.max(0) > 0
                off = offset_of(f, arr[0], ok)
                arr -= off
                arr[:, ~ok] = np.nan
                stack.append(arr)
                used.append({"scene": f["id"], "date": f["properties"]["datetime"][:10],
                             "cloud_pct": f["properties"].get("eo:cloud_cover"),
                             "offset_removed": off, "usable_pct": round(100 * float(ok.mean()), 1)})
            stack = np.stack(stack)                              # (scenes, bands, H, W)
            nvalid = np.isfinite(stack[:, 0]).sum(0).astype(np.float32)
            with np.errstate(all="ignore"):
                med = np.nanmedian(stack, axis=0)
            med = np.nan_to_num(np.clip(med, 0, 65535), nan=0.0)
            with rasterio.open(site_dir / "t1.tif") as ref:
                prof = {"crs": ref.crs, "transform": ref.transform}
            out.parent.mkdir(exist_ok=True)
            with rasterio.open(out, "w", driver="GTiff", height=shape[0], width=shape[1],
                               count=6, dtype="uint16", compress="deflate", **prof) as dst:
                dst.write(np.concatenate([med, nvalid[None]]).astype("uint16"))
                dst.descriptions = tuple(BANDS + ["n_valid"])
            (out.parent / f"{year}.json").write_text(json.dumps(used, indent=2))
            return (f"{meta['site']} {year}: {len(used)} scenes "
                    f"({', '.join(u['date'][5:] for u in used)}), "
                    f"pixels with no clear view: {100 * float((nvalid == 0).mean()):.2f}%")
    except Exception as exc:                                     # noqa: BLE001
        return f"{meta['site']} {year}: FAILED {type(exc).__name__}: {str(exc)[:110]}"


def site_metas():
    only = set(sys.argv[1:])
    return [m for d in SITE_DIRS for m in sorted(d.glob("*/meta.json"))
            if not only or m.parent.name in only]


def main():
    jobs = []
    for mfile in site_metas():
        meta = json.loads(mfile.read_text())
        jobs += [(mfile.parent, meta, y) for y in YEARS]
    print(f"{len(jobs)} site-years to fetch")
    with ThreadPoolExecutor(max_workers=8) as pool:
        for msg in pool.map(one, jobs):
            print(msg, flush=True)
    for mfile in site_metas():
        d = mfile.parent / "series"
        years = {p.stem: json.loads(p.read_text()) for p in sorted(d.glob("20*.json"))}
        (d / "series.json").write_text(json.dumps(years, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
