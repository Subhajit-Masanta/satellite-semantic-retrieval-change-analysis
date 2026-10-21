"""Change TYPE from spectral indices.   (PS SIH26227 section 2.2.2)

Three standard indices, each a simple ratio of two Sentinel-2 bands:

    NDVI = (nir - red)   / (nir + red)      high on healthy vegetation
    NDWI = (green - nir) / (green + nir)    high on open water
    NDBI = (swir - nir)  / (swir + nir)     high on built-up and bare ground

We compare each index between an earlier and a later image. Before comparing,
the scene-wide median shift of each index is removed, so a season that makes
the WHOLE scene greener or drier does not get reported as change.

No training. The thresholds are plain numbers on index differences.
"""

import numpy as np

TYPES = {
    0: ("no change", (0, 0, 0)),
    1: ("built-up or bare ground gained", (230, 50, 50)),
    2: ("vegetation lost", (240, 150, 40)),
    3: ("vegetation gained", (60, 180, 75)),
    4: ("water gained", (50, 110, 230)),
    5: ("water lost", (70, 200, 220)),
}


def indices(green, red, nir, swir):
    e = 1e-6
    g, r, n, s = (x.astype(np.float32) for x in (green, red, nir, swir))
    return {"ndvi": (n - r) / (n + r + e), "ndwi": (g - n) / (g + n + e),
            "ndbi": (s - n) / (s + n + e)}


def centred_delta(a, b, valid=None):
    """later minus earlier, with the scene-wide median shift removed."""
    d = b - a
    ref = d[valid] if valid is not None and valid.any() else d
    return d - np.median(ref)


def robust_z(d, valid=None):
    ref = d[valid] if valid is not None and valid.any() else d
    mad = np.median(np.abs(ref - np.median(ref))) * 1.4826 + 1e-6
    return d / mad


def built_score(i1, i2, valid=None):
    """How strongly a pixel moved towards 'built-up / bare': NDBI up and NDVI
    down, each measured in units of the scene's own typical variation."""
    return (robust_z(centred_delta(i1["ndbi"], i2["ndbi"], valid), valid)
            - robust_z(centred_delta(i1["ndvi"], i2["ndvi"], valid), valid)) / 2.0


def classify(i1, i2, valid=None, t_veg=0.15, t_water=0.15, t_built=0.07):
    """Per-pixel change type (see TYPES). Water is decided first, then
    built-up, then vegetation, so one pixel gets one label."""
    dv = centred_delta(i1["ndvi"], i2["ndvi"], valid)
    dw = centred_delta(i1["ndwi"], i2["ndwi"], valid)
    db = centred_delta(i1["ndbi"], i2["ndbi"], valid)
    out = np.zeros(dv.shape, np.uint8)
    water1, water2 = i1["ndwi"] > 0.0, i2["ndwi"] > 0.0
    out[(dv > t_veg) & ~water2] = 3
    out[(dv < -t_veg) & ~water2] = 2
    out[(db > t_built) & (dv < -0.05) & (i2["ndvi"] < 0.35) & ~water2] = 1
    out[(dw > t_water) & water2 & ~water1] = 4
    out[(dw < -t_water) & water1 & ~water2] = 5
    if valid is not None:
        out[~valid] = 0
    return out


def colour(types):
    rgb = np.zeros(types.shape + (3,), np.uint8)
    for k, (_, c) in TYPES.items():
        rgb[types == k] = c
    return rgb


def area_table(types, gsd_m=10.0):
    """Area of each change type in hectares (1 ha = 10,000 m2)."""
    px_ha = gsd_m * gsd_m / 10000.0
    return {TYPES[k][0]: round(float((types == k).sum()) * px_ha, 1)
            for k in TYPES if k != 0}
