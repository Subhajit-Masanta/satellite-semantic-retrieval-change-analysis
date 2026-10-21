"""Stage 3c -- discovery and clustering.   OWNER: Person B   (STRETCH GOAL)

PS SIH26227 section 2.2.4: an analyst who finds one site of interest should be
able to discover comparable sites without hand-writing a new query for each.

Marked a stretch goal in PROJECT_PLAN.md -- it is cheap because it reuses the
index that already exists. Do not start this before Gates A/B/C pass.
"""

import numpy as np

try:
    from sklearn.cluster import KMeans
    HAVE_SKLEARN = True
except ImportError:                       # pragma: no cover
    HAVE_SKLEARN = False


def cluster_sites(index, n_clusters=8, seed=0):
    """Group the archive into visually/semantically similar site types.

    Returns (labels, centroids). Labels align with index rows.
    """
    if index.n == 0:
        return np.zeros(0, int), np.zeros((0, index.dim), np.float32)
    k = min(n_clusters, index.n)

    if HAVE_SKLEARN:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(index.vectors)
        return km.labels_, km.cluster_centers_.astype(np.float32)
    return _kmeans_numpy(index.vectors, k, seed)


def _kmeans_numpy(x, k, seed=0, iters=50):
    """Tiny spherical k-means so clustering works without scikit-learn."""
    rng = np.random.default_rng(seed)
    centroids = x[rng.choice(len(x), k, replace=False)].copy()
    labels = np.zeros(len(x), int)
    for _ in range(iters):
        new = np.argmax(x @ centroids.T, axis=1)      # cosine, vectors are unit
        if np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            members = x[labels == j]
            if len(members):
                c = members.mean(0)
                centroids[j] = c / max(np.linalg.norm(c), 1e-9)
    return labels, centroids


def similar_sites(index, row, k=10, same_cluster_only=False, labels=None):
    """'Find me more places like this one.' The analyst-facing entry point."""
    subset = None
    if same_cluster_only and labels is not None:
        subset = [i for i, l in enumerate(labels) if l == labels[row] and i != row]
    hits = index.search(index.vectors[row], k=k + 1, subset=subset)
    return [h for h in hits if h[0] != row][:k]


def cluster_summary(index, labels):
    """How many tiles per cluster, with an exemplar for each -- what an analyst
    would actually look at."""
    out = []
    for c in sorted(set(labels.tolist())):
        rows = [i for i, l in enumerate(labels) if l == c]
        centre = index.vectors[rows].mean(0)
        centre = centre / max(np.linalg.norm(centre), 1e-9)
        exemplar = max(rows, key=lambda r: float(index.vectors[r] @ centre))
        out.append({"cluster": int(c), "size": len(rows), "exemplar_row": exemplar,
                    "exemplar": index.metadata[exemplar].get("tile_id", exemplar)})
    return out
