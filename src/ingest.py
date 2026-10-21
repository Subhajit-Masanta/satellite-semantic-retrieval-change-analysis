"""Stage 1 -- ingestion and tiling.   OWNER: Person A

Reads imagery (GeoTIFF/COG where rasterio is available, ordinary rasters
otherwise), cuts it into fixed tiles, and keeps geo-metadata beside every tile
so provenance survives all the way into the reported results.

PS SIH26227 requires that georeferencing and acquisition metadata be preserved
and that common geospatial formats (GeoTIFF, COG) be ingested.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:                       # rasterio is optional: needed only for real GeoTIFF/COG
    import rasterio
    from rasterio.windows import Window
    HAVE_RASTERIO = True
except ImportError:        # pragma: no cover - depends on environment
    HAVE_RASTERIO = False


@dataclass
class Tile:
    """One image tile plus everything needed to trace it back to its source."""

    image: np.ndarray                     # HxWx3 uint8, BGR
    source: str                           # source scene path / product id
    row: int                              # tile position in the grid
    col: int
    x: int                                # pixel offset within the source scene
    y: int
    date: str | None = None               # acquisition date, ISO 8601
    sensor: str | None = None             # e.g. "Sentinel-2", "LEVIR-CD"
    crs: str | None = None                # coordinate reference system
    bounds: tuple[float, float, float, float] | None = None   # geographic extent
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def tile_id(self) -> str:
        return f"{Path(self.source).stem}__r{self.row}c{self.col}"

    def metadata(self) -> dict[str, Any]:
        """Everything except the pixels -- this is what goes in the index table."""
        d = asdict(self)
        d.pop("image")
        d["tile_id"] = self.tile_id
        return d


def _tile_array(img, source, tile_size, stride, **meta) -> list[Tile]:
    h, w = img.shape[:2]
    tiles = []
    for r, y in enumerate(range(0, h - tile_size + 1, stride)):
        for c, x in enumerate(range(0, w - tile_size + 1, stride)):
            tiles.append(Tile(image=img[y:y + tile_size, x:x + tile_size],
                              source=str(source), row=r, col=c, x=x, y=y, **meta))
    return tiles


def load_tiles(path, tile_size=256, stride=None, date=None, sensor=None) -> list[Tile]:
    """Read one scene and return its tiles.

    stride defaults to tile_size (no overlap). Use a smaller stride for
    overlapping tiles if retrieval recall needs it.
    """
    stride = stride or tile_size
    path = Path(path)

    if HAVE_RASTERIO and path.suffix.lower() in {".tif", ".tiff"}:
        with rasterio.open(path) as ds:
            # read first three bands as BGR; real multispectral work keeps all bands
            count = min(3, ds.count)
            arr = ds.read(list(range(1, count + 1)))            # (bands, H, W)
            arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype != np.uint8:                            # scale to 8-bit
                lo, hi = np.percentile(arr, [2, 98])
                arr = np.clip((arr - lo) / max(hi - lo, 1e-6) * 255, 0, 255)
            img = np.ascontiguousarray(arr.astype(np.uint8)[:, :, ::-1])
            return _tile_array(img, path, tile_size, stride, date=date, sensor=sensor,
                               crs=str(ds.crs), bounds=tuple(ds.bounds))

    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return _tile_array(img, path, tile_size, stride, date=date, sensor=sensor)


def load_pair(path_t1, path_t2, tile_size=256, stride=None,
              date_t1=None, date_t2=None, sensor=None):
    """Load a bitemporal pair and return aligned tile lists (t1[i] matches t2[i]).

    LEVIR-CD and OSCD ship pre-aligned pairs, which is deliberate: general
    multi-sensor registration is out of scope (see PROJECT_PLAN.md).
    """
    t1 = load_tiles(path_t1, tile_size, stride, date=date_t1, sensor=sensor)
    t2 = load_tiles(path_t2, tile_size, stride, date=date_t2, sensor=sensor)
    if len(t1) != len(t2):
        raise ValueError(f"pair does not tile to the same grid: {len(t1)} vs {len(t2)}")
    return t1, t2
