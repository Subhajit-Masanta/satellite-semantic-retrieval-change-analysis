"""
Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery
Using a Shared Foundation-Model Embedding Space

BCSE407L Computer Vision RIL  |  Aligned to SIH26227 (MoD / Indian Army DGIS)

DRAFT SKELETON. Each section below is owned by one team member and will move
into its own file. Everything here runs inference only -- nothing is trained.

Owners:  P1 = ingest/embed/index   P2 = retrieval   P3 = change+suppress   P4 = metrics
"""

import time

import cv2
import numpy as np


# ============================================================================
# P1 -- ingest.py : GeoTIFF/COG reading and tiling
# ============================================================================

def load_tiles(path, tile_size=256):
    """Read an image (later: GeoTIFF/COG via rasterio) and cut it into tiles.

    Real version keeps geo-metadata alongside every tile so provenance
    survives into the results: CRS, bounds, acquisition date, source scene.
    """
    img = cv2.imread(path)
    h, w = img.shape[:2]
    tiles, meta = [], []
    for y in range(0, h - tile_size + 1, tile_size):
        for x in range(0, w - tile_size + 1, tile_size):
            tiles.append(img[y:y + tile_size, x:x + tile_size])
            meta.append({"source": path, "x": x, "y": y})
    return tiles, meta


# ============================================================================
# P1 -- embedder.py : frozen encoder, tile -> vector
# ============================================================================

def embed_tiles(tiles):
    """Encode tiles with a FROZEN RemoteCLIP/CLIP encoder. Inference only.

    Placeholder: returns random unit vectors so P2 and P3 can build against
    this interface before the real encoder is wired in. Swapping in the real
    model must not change the shape contract -- (n_tiles, dim), L2-normalised.
    """
    rng = np.random.default_rng(0)
    vecs = rng.random((len(tiles), 512), dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def embed_text(query):
    """Encode a free-text query into the SAME space as the tile vectors.

    Placeholder until the real text encoder is wired in.
    """
    rng = np.random.default_rng(abs(hash(query)) % (2 ** 32))
    v = rng.random(512, dtype=np.float32)
    return v / np.linalg.norm(v)


# ============================================================================
# P1 -- index_builder.py : the shared index (Stages 3 AND 4 both read this)
# ============================================================================

def build_index(vectors):
    """Build the searchable index. Real version uses FAISS.

    Report build time and storage footprint -- the PS requires both in the
    evaluation report.
    """
    t0 = time.time()
    index = {"vectors": vectors, "n": len(vectors)}
    index["build_time_s"] = time.time() - t0
    index["storage_mb"] = vectors.nbytes / 1e6
    return index


# ============================================================================
# P2 -- retrieval.py : semantic search over the shared index
# ============================================================================

def search(index, query_vec, k=5):
    """Rank every tile against a query vector by cosine similarity."""
    scores = index["vectors"] @ query_vec
    top = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in top]


def search_by_text(index, query, k=5):
    return search(index, embed_text(query), k)


def search_by_image(index, tile_idx, k=5):
    """Image-to-image search: find tiles similar to one the analyst picked."""
    return search(index, index["vectors"][tile_idx], k + 1)[1:]


# ============================================================================
# P3 -- change_detect.py : change signal from the SAME embeddings
# ============================================================================

def embedding_change_score(vec_t1, vec_t2):
    """Cosine distance in the shared space. This IS the novelty:
    the vectors that answer search queries also produce the change signal."""
    return float(1.0 - np.dot(vec_t1, vec_t2))


def change_mask(img_t1, img_t2, threshold=40):
    """Pixel-level refinement of the change region."""
    g1 = cv2.cvtColor(img_t1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img_t2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(g1, g2)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


# ============================================================================
# P3 -- suppress.py : the false-alarm suppression that Gate C tests
# ============================================================================

def radiometric_normalise(img_t1, img_t2):
    """Match t2's brightness/contrast to t1 so a pure illumination shift
    does not masquerade as change. Gate C lives or dies here."""
    out = np.zeros_like(img_t2)
    for c in range(3):
        s1, m1 = img_t1[:, :, c].std(), img_t1[:, :, c].mean()
        s2, m2 = img_t2[:, :, c].std(), img_t2[:, :, c].mean()
        scaled = (img_t2[:, :, c].astype(float) - m2) / (s2 + 1e-6) * s1 + m1
        out[:, :, c] = np.clip(scaled, 0, 255).astype(np.uint8)
    return out


def cloud_shadow_mask(img, bright=248, dark=18, min_area_frac=0.04,
                      texture_thresh=12.0, buffer_px=25):
    """Mark pixels USABLE (255) vs cloud/shadow-contaminated (0).

    A plain brightness gate does not work here: new construction -- concrete,
    metal roofing, bare ground -- is bright, so a naive gate deletes exactly the
    change type this project cares about most. Cloud is separated from bright
    change by two extra cues: cloud is locally SMOOTH (low texture) and covers a
    LARGE FRACTION of the tile, whereas a new roof is small and edge-rich. The
    area test is a fraction of tile area, not a pixel count, so it does not
    change meaning when the tile size changes. Detected cores are then BUFFERED
    outward: a cloud's feathered edge is not smooth enough to be caught directly
    but is just as unusable, and buffering a cloud mask is standard practice.

    Honest limitations, state these in the paper:
      * RGB-only heuristic. A real system uses the sensor's own cloud-mask
        product (e.g. the Sentinel-2 SCL band), which this stands in for.
      * A very large, perfectly flat bright roof is genuinely indistinguishable
        from cloud by these cues and will be masked. Smoothness plus size is
        the discriminator; nothing here reads spectral signature.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # local texture: std-dev in a small window (clouds are smooth, roofs are not)
    f = gray.astype(np.float32)
    mean = cv2.blur(f, (9, 9))
    texture = np.sqrt(np.maximum(cv2.blur(f * f, (9, 9)) - mean * mean, 0))

    cloud_like = ((gray >= bright) & (texture < texture_thresh)).astype(np.uint8)
    shadow_like = ((gray <= dark) & (texture < texture_thresh)).astype(np.uint8)

    min_area = min_area_frac * gray.size  # fraction of tile, not raw pixels
    bad = np.zeros_like(gray, dtype=np.uint8)
    for candidate in (cloud_like, shadow_like):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
        for i in range(1, n):  # skip background
            if stats[i, cv2.CC_STAT_AREA] >= min_area:  # only LARGE blobs are cloud
                bad[labels == i] = 1

    if buffer_px:  # grow the mask to cover feathered edges and cast shadow
        k = 2 * buffer_px + 1
        bad = cv2.dilate(bad, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    return ((1 - bad) * 255).astype(np.uint8)


def usable_mask(img_t1, img_t2):
    """A pixel is usable only if it is usable in BOTH acquisitions."""
    return cv2.bitwise_and(cloud_shadow_mask(img_t1), cloud_shadow_mask(img_t2))


def confidence(mask, valid_mask):
    """Fraction of flagged pixels that sit in usable (non-cloud) area."""
    flagged = mask > 0
    if flagged.sum() == 0:
        return 0.0
    return float((flagged & (valid_mask > 0)).sum() / flagged.sum())


# ============================================================================
# P4 -- metrics.py : precision-first evaluation
# ============================================================================

def precision_at_k(retrieved_ids, relevant_ids, k=5):
    hits = len(set(retrieved_ids[:k]) & set(relevant_ids))
    return hits / k


def change_metrics(pred_mask, true_mask):
    """Precision is the headline -- PS 2.2.3 prefers precision over recall."""
    p, t = pred_mask > 0, true_mask > 0
    tp, fp, fn = (p & t).sum(), (p & ~t).sum(), (~p & t).sum()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}


def false_positive_area(pred_mask):
    """Gate C metric: on a NO-CHANGE pair, every flagged pixel is a false alarm."""
    return float((pred_mask > 0).sum())


# ============================================================================
# End-to-end draft: query -> retrieve -> compare -> suppress -> report
# ============================================================================

def analyse_pair(img_t1, img_t2, suppress=True):
    """Full change path for one tile pair. Set suppress=False for the ablation."""
    t2 = radiometric_normalise(img_t1, img_t2) if suppress else img_t2
    mask = change_mask(img_t1, t2)
    if suppress:
        # Cloud/shadow is a radiometric property of the RAW acquisition, so the
        # mask is computed on the originals. Running it on the normalised image
        # silently fails: normalisation pulls a bright cloud back under the
        # detection threshold and it is never flagged.
        valid = usable_mask(img_t1, img_t2)   # usable in BOTH dates, not just t2
        mask = cv2.bitwise_and(mask, valid)
        conf = confidence(mask, valid)
    else:
        conf = 1.0
    return mask, conf


def gate_c_check(no_change_pairs, change_pairs, min_recall_retained=0.8):
    """Gate C: suppression must cut false alarms on NO-CHANGE pairs WITHOUT
    destroying sensitivity on real change.

    Both halves are required. Checking only the false-alarm half is gameable:
    a suppressor that outputs an empty mask scores a perfect zero false-alarm
    area and 'passes' while detecting nothing at all. `change_pairs` are
    (img_t1, img_t2, ground_truth_mask) triples and guard against exactly that.
    """
    fp_wins = 0
    for img_t1, img_t2 in no_change_pairs:
        naive, _ = analyse_pair(img_t1, img_t2, suppress=False)
        ours, _ = analyse_pair(img_t1, img_t2, suppress=True)
        if false_positive_area(ours) < false_positive_area(naive):
            fp_wins += 1

    retained = []
    for img_t1, img_t2, truth in change_pairs:
        naive, _ = analyse_pair(img_t1, img_t2, suppress=False)
        ours, _ = analyse_pair(img_t1, img_t2, suppress=True)
        r_naive = change_metrics(naive, truth)["recall"]
        r_ours = change_metrics(ours, truth)["recall"]
        retained.append(1.0 if r_naive == 0 else r_ours / r_naive)
    worst = min(retained) if retained else 0.0

    fp_ok = fp_wins >= 2
    sens_ok = worst >= min_recall_retained
    print(f"Gate C  false-alarm half : suppression won on {fp_wins}/{len(no_change_pairs)}"
          f" no-change pairs -> {'PASS' if fp_ok else 'FAIL'}")
    print(f"Gate C  sensitivity half : worst recall retained {worst:.2f}"
          f" (need >= {min_recall_retained}) -> {'PASS' if sens_ok else 'FAIL'}")
    return fp_ok and sens_ok
