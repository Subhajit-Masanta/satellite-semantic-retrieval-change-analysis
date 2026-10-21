"""Stage 4 -- multi-temporal change analysis.   OWNER: Person C

PS SIH26227 section 2.2.2.

THE NOVELTY LIVES HERE: the change signal is computed in the SAME embedding
space that answers retrieval queries. `embedding_change_score` consumes the
vectors Person A already built for Person B's search -- no second
representation, no second model, no second index.

Pipeline for one tile pair:
    embedding distance  ->  coarse "did anything change here"
    pixel refinement    ->  where exactly, at pixel resolution
    suppression         ->  drop confound-driven false alarms
"""

import cv2
import numpy as np

from .suppress import (confidence, drop_small_regions, radiometric_normalise,
                       usable_mask)


# ---------------------------------------------------------------- shared space
def embedding_change_score(vec_t1, vec_t2):
    """Cosine distance between the two dates' vectors, in [0, 2].

    This is the shared-space change signal -- the same vectors Person B ranks
    queries with. Low = the tile looks semantically unchanged.
    """
    return float(1.0 - np.dot(vec_t1, vec_t2))


def embedding_change_scores(vecs_t1, vecs_t2):
    """Vectorised over a whole tile grid."""
    return 1.0 - np.sum(vecs_t1 * vecs_t2, axis=1)


def screen_by_embedding(vecs_t1, vecs_t2, threshold=0.10):
    """Cheap first pass: which tiles are worth refining at pixel level?

    Skipping unchanged tiles is what makes this scale -- most of an archive is
    unchanged most of the time.
    """
    scores = embedding_change_scores(vecs_t1, vecs_t2)
    return np.flatnonzero(scores >= threshold), scores


# ------------------------------------------------------------ pixel refinement
def change_mask(img_t1, img_t2, threshold=40, blur=3):
    """Pixel-level change mask. Assumes the pair is already co-registered
    (LEVIR-CD and OSCD are; general registration is out of scope)."""
    g1 = cv2.cvtColor(img_t1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img_t2, cv2.COLOR_BGR2GRAY)
    if blur:
        g1 = cv2.GaussianBlur(g1, (blur, blur), 0)
        g2 = cv2.GaussianBlur(g2, (blur, blur), 0)
    diff = cv2.absdiff(g1, g2)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


# ------------------------------------------------------------------- baselines
def baseline_pixel_otsu(img_t1, img_t2):
    """Baseline 1: pixel differencing + Otsu. No features, no suppression."""
    g1 = cv2.cvtColor(img_t1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img_t2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(g1, g2)
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def baseline_cva(img_t1, img_t2, percentile=95):
    """Baseline 2: Change Vector Analysis (Malila 1980) -- per-pixel magnitude
    of the multi-band difference vector."""
    d = img_t1.astype(np.float32) - img_t2.astype(np.float32)
    mag = np.sqrt((d ** 2).sum(axis=2))
    thr = np.percentile(mag, percentile)
    return ((mag >= thr) * 255).astype(np.uint8)


def baseline_ssim(img_t1, img_t2, win=7, threshold=0.6):
    """Baseline 3: structural differencing. Low local SSIM = structure changed."""
    g1 = cv2.cvtColor(img_t1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g2 = cv2.cvtColor(img_t2, cv2.COLOR_BGR2GRAY).astype(np.float32)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k = (win, win)
    mu1, mu2 = cv2.blur(g1, k), cv2.blur(g2, k)
    s11 = cv2.blur(g1 * g1, k) - mu1 * mu1
    s22 = cv2.blur(g2 * g2, k) - mu2 * mu2
    s12 = cv2.blur(g1 * g2, k) - mu1 * mu2
    ssim = (((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) /
            ((mu1 ** 2 + mu2 ** 2 + c1) * (s11 + s22 + c2) + 1e-9))
    return ((ssim < threshold) * 255).astype(np.uint8)


# ---------------------------------------------------------------- the proposal
def analyse_pair(img_t1, img_t2, suppress=True, threshold=40, despeckle=False):
    """Full change path for one tile pair.

    suppress=False reproduces the un-suppressed ablation for Section 6.
    despeckle=False is the default: on OSCD, speck removal dropped recall from
    0.518 to 0.337 because real small changes were removed along with noise.
    Pass despeckle=True only if co-registration errors dominate your data.
    Returns (mask, confidence).
    """
    t2 = radiometric_normalise(img_t1, img_t2) if suppress else img_t2
    mask = change_mask(img_t1, t2, threshold=threshold)

    if not suppress:
        return mask, 1.0

    # Cloud/shadow is a radiometric property of the RAW acquisition, so the
    # mask is computed on the originals. Running it on the normalised image
    # silently fails: normalisation pulls a bright cloud back under the
    # detection threshold and it is never flagged.
    valid = usable_mask(img_t1, img_t2)
    mask = cv2.bitwise_and(mask, valid)
    if despeckle:
        mask = drop_small_regions(mask)
    return mask, confidence(mask, valid)
