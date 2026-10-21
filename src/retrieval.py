"""Stage 3 -- semantic retrieval.   OWNER: Person B

Free-text and image-to-image search over the SAME index Stage 4 uses for
change analysis. PS SIH26227 section 2.2.1.

Person B can develop this without waiting for Person A: build a SharedIndex
over random vectors and every function here still exercises correctly.
"""

import numpy as np

# The five held-out queries used for Gate A and the Section 5 results table.
# Taken from the wording of PS SIH26227 section 2.2.1.
TEST_QUERIES = [
    "newly built structures near a river",
    "large vehicle concentrations on open ground",
    "dense urban rooftops",
    "bare cleared land",
    "a water body",
]


def search_by_text(index, query, embedder, k=5, subset=None):
    """Free-text query -> ranked tiles. Returns [(row, score)]."""
    return index.search(embedder.encode_text(query)[0], k=k, subset=subset)


def search_by_image(index, row, k=5, subset=None, exclude_self=True):
    """Image-to-image: given a tile the analyst picked, find similar tiles."""
    hits = index.search(index.vectors[row], k=k + (1 if exclude_self else 0),
                        subset=subset)
    if exclude_self:
        hits = [h for h in hits if h[0] != row]
    return hits[:k]


def search_by_vector(index, vec, k=5, subset=None):
    return index.search(vec, k=k, subset=subset)


def rank_all(index, query_vec, subset=None):
    """Full ranking (not just top-k) -- needed to compute mAP."""
    return index.search(query_vec, k=index.n, subset=subset)


def explain(index, hits):
    """Attach metadata to hits so results carry provenance, not bare row ids.

    PS 2.2.5 asks that exported results retain source-scene and processing
    provenance. Full audit trail is out of scope, but every returned hit at
    least names where it came from.
    """
    out = []
    for row, score in hits:
        meta = dict(index.metadata[row])
        meta["row"] = row
        meta["score"] = round(float(score), 4)
        out.append(meta)
    return out
