"""Stage 4b -- change typing.   OWNER: Person C

PS SIH26227 section 2.2.2 asks that supported change types be classified:
construction, clearance, water-extent variation, road development.

These are INTERPRETABLE RULES over colour, texture and shape -- not a trained
classifier. Say so plainly in the paper; a rule base that a reviewer can read
is a legitimate choice, but presenting it as learned classification would not be.
"""

import cv2
import numpy as np

CHANGE_TYPES = ("construction", "clearance", "water", "vegetation", "road", "unknown")


def _region_features(img_t1, img_t2, region_mask):
    """Descriptive statistics for one changed region, before and after."""
    m = region_mask > 0
    hsv1 = cv2.cvtColor(img_t1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img_t2, cv2.COLOR_BGR2HSV)
    g1 = cv2.cvtColor(img_t1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g2 = cv2.cvtColor(img_t2, cv2.COLOR_BGR2GRAY).astype(np.float32)

    contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    elongation = 0.0
    if contours:
        c = max(contours, key=cv2.contourArea)
        if len(c) >= 5:
            (_, _), (w, h), _ = cv2.minAreaRect(c)
            if min(w, h) > 1e-6:
                elongation = max(w, h) / min(w, h)

    return {
        "brightness_before": float(g1[m].mean()),
        "brightness_after": float(g2[m].mean()),
        "texture_before": float(g1[m].std()),
        "texture_after": float(g2[m].std()),
        "hue_after": float(np.median(hsv2[:, :, 0][m])),
        "sat_after": float(np.median(hsv2[:, :, 1][m])),
        "green_before": float(((hsv1[:, :, 0] > 35) & (hsv1[:, :, 0] < 85))[m].mean()),
        "green_after": float(((hsv2[:, :, 0] > 35) & (hsv2[:, :, 0] < 85))[m].mean()),
        "area_frac": float(m.mean()),
        "elongation": elongation,
    }


def classify_region(img_t1, img_t2, region_mask):
    """Label one changed region. Returns (type, confidence, features)."""
    f = _region_features(img_t1, img_t2, region_mask)
    d_bright = f["brightness_after"] - f["brightness_before"]
    d_texture = f["texture_after"] - f["texture_before"]

    # water: dark, smooth, low saturation, blue-ish hue after
    if (f["brightness_after"] < 70 and f["texture_after"] < 25
            and 90 <= f["hue_after"] <= 130):
        return "water", 0.7, f

    # road: strongly elongated, brighter, smooth
    if f["elongation"] >= 4.0 and d_bright > 10 and f["texture_after"] < 45:
        return "road", 0.6, f

    # vegetation: green fraction moved substantially either way
    if abs(f["green_after"] - f["green_before"]) > 0.35:
        return "vegetation", 0.65, f

    # construction: got brighter AND more textured (roofs, edges, plant)
    if d_bright > 12 and d_texture > -5:
        return "construction", 0.7, f

    # clearance: vegetation removed -> brighter, smoother, less green
    if d_bright > 8 and d_texture < -8 and f["green_before"] > f["green_after"]:
        return "clearance", 0.6, f

    return "unknown", 0.3, f


def classify_mask(img_t1, img_t2, mask, min_area_frac=0.0005):
    """Split a change mask into regions and label each one.

    Returns a list of dicts, one per region -- the rows of the analyst's
    review queue.
    """
    min_area = min_area_frac * mask.size
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), 8)

    out = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        region = (labels == i).astype(np.uint8) * 255
        kind, conf, feats = classify_region(img_t1, img_t2, region)
        out.append({
            "region_id": i,
            "type": kind,
            "confidence": conf,
            "area_px": int(area),
            "area_frac": round(area / mask.size, 5),
            "centroid": (float(centroids[i][0]), float(centroids[i][1])),
            "bbox": (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                     int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])),
            "features": feats,
        })
    out.sort(key=lambda r: -r["area_px"])
    return out


def summarise(regions):
    """Counts per change type -- the headline line of a change report."""
    counts = {t: 0 for t in CHANGE_TYPES}
    for r in regions:
        counts[r["type"]] = counts.get(r["type"], 0) + 1
    return {k: v for k, v in counts.items() if v}
