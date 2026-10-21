"""Change over MANY dates instead of two.   (PS SIH26227 section 2.2.2 / 2.2.3)

Input: one cloud-free composite per year (see demo/fetch_series.py).

Two-date change asks "is 2025 different from 2019?". That also fires on a field
that happened to be wet in one of the two images. The multi-year test asks a
stricter question: "is there ONE year after which the pixel is different and
STAYS different?". It also tells us WHEN: the first year the change is seen.

For every pixel and every possible split year we compare the mean of the years
before with the mean of the years after, and divide by how much the pixel
wobbles inside the two groups. A real, lasting change gives a big step and a
small wobble. A one-off difference gives a big wobble.
"""

import json
from pathlib import Path

import numpy as np

BANDS = ["blue", "green", "red", "nir", "swir16"]


def load_series(site_dir):
    """-> years (list), cube (Y, 5, H, W) float32 reflectance, valid (Y, H, W)."""
    import rasterio
    d = Path(site_dir) / "series"
    years, arrs, valid = [], [], []
    for f in sorted(d.glob("20*.tif")):
        with rasterio.open(f) as ds:
            a = ds.read().astype(np.float32)
        years.append(int(f.stem)); arrs.append(a[:5]); valid.append(a[5] > 0)
    return years, np.stack(arrs), np.stack(valid)


def normalise(cube, valid):
    """Bring every year onto the same brightness scale: each band of each year
    is matched to the middle value and spread of the many-year median image."""
    ref = np.nanmedian(np.where(valid[:, None], cube, np.nan), axis=0)       # (5, H, W)
    out = np.empty_like(cube)
    scale = np.empty(cube.shape[1], np.float32)
    for b in range(cube.shape[1]):
        r = ref[b][np.isfinite(ref[b])]
        rm, rs = np.median(r), np.percentile(r, 84) - np.percentile(r, 16) + 1e-6
        scale[b] = rs
        for y in range(cube.shape[0]):
            v = cube[y, b][valid[y]]
            m, s = np.median(v), np.percentile(v, 84) - np.percentile(v, 16) + 1e-6
            out[y, b] = ((cube[y, b] - m) / s * rs + rm - rm) / rs           # in units of spread
    return out, np.nan_to_num(ref)


def noise_level(z):
    """Typical size of a pixel's difference between two neighbouring years.
    Most pixels do not change from one year to the next, so the middle value of
    this difference is the scene's own noise (weather, crops, haze)."""
    d = np.sqrt(((z[1:] - z[:-1]) ** 2).mean(1))            # (Y-1, H, W)
    return float(np.median(d))


def step_change(z, min_side=2, max_gap=2, floor=None):
    """z: (Y, B, H, W) normalised cube.

    For every pixel we look for a 'before' group of years and an 'after' group.
    Up to max_gap years in between may be skipped: construction takes time, and
    the half-built years belong to neither group.

    Returns
      two_date : size of the first-year vs last-year difference
      step     : size of the best before/after step
      stat     : step divided by the wobble inside the two groups
      first    : index of the first year that is no longer 'before'
    """
    Y = z.shape[0]
    floor = noise_level(z) if floor is None else floor
    two_date = np.sqrt(((z[-1] - z[0]) ** 2).mean(0))
    best_stat = np.full(z.shape[2:], -1, np.float32)
    best_step = np.zeros(z.shape[2:], np.float32)
    first = np.zeros(z.shape[2:], np.int16)
    for k in range(min_side, Y - min_side + 1):
        for gap in range(0, max_gap + 1):
            if Y - (k + gap) < min_side:
                break
            before, after = z[:k], z[k + gap:]
            mb, ma = before.mean(0), after.mean(0)
            step = np.sqrt(((ma - mb) ** 2).mean(0))
            n = before.shape[0] + after.shape[0]
            wob = np.sqrt((((before - mb) ** 2).sum(0) + ((after - ma) ** 2).sum(0)).mean(0)
                          / max(n - 2, 1))
            stat = step / (wob + floor)
            better = stat > best_stat
            best_stat[better], best_step[better], first[better] = stat[better], step[better], k
    return two_date, best_step, best_stat, first


def null_stat(z, floor, n_shuffles=6, q=99.0, seed=0, sub=4):
    """What 'step score' do we get when there is NO lasting change by
    construction? We shuffle the order of the years (same shuffle for every
    pixel) and score again. A real step is broken up by the shuffle, so these
    scores show what chance alone produces. The q-th percentile becomes the
    bar a pixel must clear."""
    rng = np.random.default_rng(seed)
    zs = z[:, :, ::sub, ::sub]
    vals = []
    Y = z.shape[0]
    ident = np.arange(Y)
    for _ in range(n_shuffles):
        for _try in range(20):
            perm = rng.permutation(Y)
            # skip shuffles that keep the years nearly in order (or reversed)
            if abs(np.corrcoef(perm, ident)[0, 1]) < 0.35:
                break
        _, _, st, _ = step_change(zs[perm], floor=floor)
        vals.append(st.ravel())
    return float(np.percentile(np.concatenate(vals), q))


def detect(z, strength=2.0, q=99.0):
    """Both maps use the SAME size cut: `strength` times the scene's own
    year-to-year noise, so the cut adapts to each site.

      two_mask : first year vs last year differ by more than the cut
      lasting  : a before/after step larger than the cut whose score is higher
                 than what shuffled (no-change) years reach q% of the time
    """
    noise = noise_level(z)
    floor = 0.5 * noise
    two_date, step, stat, first = step_change(z, floor=floor)
    cut = strength * noise
    bar = null_stat(z, floor, q=q)
    return {"two_date": two_date, "two_mask": two_date >= cut, "step": step, "stat": stat,
            "first": first, "lasting": (step >= cut) & (stat >= bar),
            "cut": float(cut), "noise": noise, "stat_bar": bar}


def before_after(cube, valid, first, lasting):
    """Mean reflectance before and after each pixel's own change year (for the
    change-type indices). Pixels with no lasting change use first vs last half."""
    Y = cube.shape[0]
    k = np.where(lasting, first, Y // 2)
    idx = np.arange(Y)[:, None, None]
    wb = (idx < k[None]) & valid
    wa = (idx >= k[None]) & valid
    nb, na = np.maximum(wb.sum(0), 1), np.maximum(wa.sum(0), 1)
    mb = (cube * wb[:, None]).sum(0) / nb
    ma = (cube * wa[:, None]).sum(0) / na
    return mb, ma


def series_info(site_dir):
    f = Path(site_dir) / "series" / "series.json"
    return json.loads(f.read_text()) if f.exists() else {}
