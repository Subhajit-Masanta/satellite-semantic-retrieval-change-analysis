"""Stage 5 -- false-alarm suppression.   OWNER: Person C

PS SIH26227 section 2.2.3: season, illumination, view angle, cloud, haze, snow,
shadow, radiometric inconsistency and imperfect co-registration are CONFOUNDS,
not change. The PS explicitly prefers analytically useful precision over
indiscriminate change recall.

This module is where the novelty claim is won or lost -- Gate C tests it.

Two bugs found while validating this logic, both worth knowing about:
  1. A plain brightness gate deletes bright REAL change. New concrete and metal
     roofing are bright, so "mask anything bright" throws away exactly the
     construction the project is looking for. Fixed by also requiring low
     texture and a large area fraction.
  2. Masking must run on the RAW acquisitions, not the radiometrically
     normalised one. Normalisation pulls a bright cloud back under the
     detection threshold, so the cloud is silently never flagged.
"""

import cv2
import numpy as np


def radiometric_normalise(img_t1, img_t2):
    """Match t2's per-channel mean and std to t1.

    A pure illumination or sensor-gain difference then cancels out instead of
    appearing as change across the whole tile.
    """
    out = np.zeros_like(img_t2)
    for c in range(3):
        s1, m1 = float(img_t1[:, :, c].std()), float(img_t1[:, :, c].mean())
        s2, m2 = float(img_t2[:, :, c].std()), float(img_t2[:, :, c].mean())
        scaled = (img_t2[:, :, c].astype(np.float32) - m2) / (s2 + 1e-6) * s1 + m1
        out[:, :, c] = np.clip(scaled, 0, 255).astype(np.uint8)
    return out


def cloud_shadow_mask(img, bright=248, dark=18, min_area_frac=0.03,
                      texture_thresh=12.0, buffer_px=25):
    """Mark pixels USABLE (255) vs cloud/shadow-contaminated (0).

    Cloud is separated from bright change by two extra cues beyond brightness:
    cloud is locally SMOOTH (low texture) and covers a LARGE FRACTION of the
    tile, whereas a new roof is small and edge-rich. The area test is a
    fraction of tile area, not a pixel count, so it keeps its meaning when the
    tile size changes. Detected cores are then BUFFERED outward -- a cloud's
    feathered edge and cast shadow are not smooth enough to be caught directly
    but are just as unusable, and buffering a cloud mask is standard practice.

    >>> TUNE min_area_frac ON REAL DATA. <<<
    It is the single most important knob here: it is the cloud/change decision
    boundary and therefore sets the precision-sensitivity tradeoff directly.
    Too high and clouds leak through as false change; too low and large real
    structures get masked as cloud. The default is a starting point, not a
    calibrated value -- sweep it on LEVIR-CD/OSCD and report the sweep in the
    ablation (Section 6).

    Honest limitations, state these in the paper:
      * RGB-only heuristic standing in for a real cloud-mask product such as
        the Sentinel-2 SCL band.
      * A very large, perfectly flat bright roof is genuinely indistinguishable
        from cloud by these cues and will be masked. Nothing here reads
        spectral signature.
      * Separation relies on clouds being LARGER than the built structures in
        the scene. In a tile dominated by one huge bright building that
        assumption breaks.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    f = gray.astype(np.float32)
    mean = cv2.blur(f, (9, 9))
    texture = np.sqrt(np.maximum(cv2.blur(f * f, (9, 9)) - mean * mean, 0))

    cloud_like = ((gray >= bright) & (texture < texture_thresh)).astype(np.uint8)
    shadow_like = ((gray <= dark) & (texture < texture_thresh)).astype(np.uint8)

    # A cloud is ONE object. Scene noise showing through its semi-transparent
    # edge splits the candidate region into fragments, none of which clears the
    # area test on its own, so the cloud escapes entirely. Close the candidates
    # first to heal those noise-induced gaps before measuring area.
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    cloud_like = cv2.morphologyEx(cloud_like, cv2.MORPH_CLOSE, close_k)
    shadow_like = cv2.morphologyEx(shadow_like, cv2.MORPH_CLOSE, close_k)

    min_area = min_area_frac * gray.size
    bad = np.zeros_like(gray, dtype=np.uint8)
    for candidate in (cloud_like, shadow_like):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
        for i in range(1, n):                       # skip background
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                bad[labels == i] = 1

    if buffer_px:
        k = 2 * buffer_px + 1
        bad = cv2.dilate(bad, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    return ((1 - bad) * 255).astype(np.uint8)


def usable_mask(img_t1, img_t2):
    """A pixel is usable only if it is usable in BOTH acquisitions.

    Pass the RAW images here, never the normalised one -- see module docstring.
    """
    return cv2.bitwise_and(cloud_shadow_mask(img_t1), cloud_shadow_mask(img_t2))


def confidence(mask, valid_mask):
    """Share of flagged pixels sitting in usable (non-contaminated) area.

    Reported per candidate so an analyst can triage: PS 2.2.5 asks for a ranked
    review queue with confidence attached.
    """
    flagged = mask > 0
    if not flagged.any():
        return 0.0
    return float((flagged & (valid_mask > 0)).sum() / flagged.sum())


def drop_small_regions(mask, min_area_frac=0.0005):
    """Remove speckle. Co-registration error produces thin edge slivers that are
    never real change."""
    min_area = min_area_frac * mask.size
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out
