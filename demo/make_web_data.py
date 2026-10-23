"""Make the small data copies that go to GitHub for the deployed demo.

    python demo/make_web_data.py

The full datasets stay in data/ (not in git). The website gets, in demo_data/:
  * 6 OSCD cities, bands re-saved with lossless DEFLATE compression, plus one
    original band per city for its GPS, the dates and the change mask
  * 200 EuroSAT patches per class (a fixed random choice), for "Check the search"
Pixel values are copied unchanged, so the website shows the same as local.
"""

import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402

WEB = core.ROOT / "demo_data"
OSCD_WEB = ["beirut", "dubai", "lasvegas", "mumbai", "hongkong", "abudhabi"]
EUROSAT_PER_CLASS = 200


def copy_tif(src, dst):
    import rasterio
    with rasterio.open(src) as r:
        prof = r.profile
        prof.update(compress="deflate", predictor=2, zlevel=9)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst, "w", **prof) as o:
            o.write(r.read())


def oscd():
    for city in OSCD_WEB:
        src = next(d.parent for d in (core.ROOT / "data" / "OSCD").rglob("imgs_1_rect")
                   if d.parent.name == city)
        dst = WEB / "OSCD" / city
        for date in (1, 2):
            for band in ("B02", "B03", "B04", "B08"):
                copy_tif(src / f"imgs_{date}_rect" / f"{band}.tif", dst / f"imgs_{date}_rect" / f"{band}.tif")
        geo = sorted((src / "imgs_1").glob("*B02.tif"))[0]               # GPS of the city
        copy_tif(geo, dst / "imgs_1" / geo.name)
        shutil.copy(src / "dates.txt", dst / "dates.txt")
        cm = next(p for p in (core.ROOT / "data" / "OSCD").rglob("cm.png") if p.parent.parent.name == city)
        (dst / "cm").mkdir(exist_ok=True)
        shutil.copy(cm, dst / "cm" / "cm.png")
        size = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file())
        print(f"OSCD {city}: {size / 1e6:.1f} MB")


def eurosat():
    src = core.ROOT / "data" / "EuroSAT" / "2750"
    rng = np.random.default_rng(0)
    for cls in sorted(p.name for p in src.iterdir() if p.is_dir()):
        files = sorted((src / cls).glob("*.jpg"))
        out = WEB / "EuroSAT" / cls
        out.mkdir(parents=True, exist_ok=True)
        for i in sorted(rng.choice(len(files), EUROSAT_PER_CLASS, replace=False)):
            shutil.copy(files[i], out / files[i].name)
    size = sum(f.stat().st_size for f in (WEB / "EuroSAT").rglob("*.jpg"))
    print(f"EuroSAT: {EUROSAT_PER_CLASS} per class, {size / 1e6:.1f} MB")


if __name__ == "__main__":
    oscd()
    eurosat()
