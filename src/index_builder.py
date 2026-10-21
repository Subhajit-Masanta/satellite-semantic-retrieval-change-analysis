"""Stage 2b -- the shared index.   OWNER: Person A

ONE index, read by BOTH Stage 3 (retrieval) and Stage 4 (change analysis).
That sharing is the project's novelty claim, so it lives in one place.

Uses FAISS when available and falls back to exact NumPy search otherwise --
at our tile counts (hundreds to low thousands) brute force is fast enough.

Reports build time, storage footprint and query latency, all of which the PS
requires in the reproducible evaluation report.
"""

import json
import pickle
import time
from pathlib import Path

import numpy as np

try:
    import faiss
    HAVE_FAISS = True
except ImportError:                       # pragma: no cover
    HAVE_FAISS = False


class SharedIndex:
    """Vectors + parallel metadata table. Cosine similarity via inner product
    on L2-normalised vectors."""

    def __init__(self, vectors, metadata, provenance=None):
        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        if len(vectors) != len(metadata):
            raise ValueError(f"{len(vectors)} vectors vs {len(metadata)} metadata rows")
        self.vectors = vectors
        self.metadata = list(metadata)
        self.provenance = provenance or {}
        self.dim = vectors.shape[1] if len(vectors) else 0

        t0 = time.perf_counter()
        if HAVE_FAISS and len(vectors):
            self.backend = "faiss"
            self._index = faiss.IndexFlatIP(self.dim)
            self._index.add(vectors)
        else:
            self.backend = "numpy"
            self._index = None
        self.build_time_s = time.perf_counter() - t0

    # ------------------------------------------------------------------ stats
    @property
    def n(self):
        return len(self.vectors)

    @property
    def storage_mb(self):
        return self.vectors.nbytes / 1e6

    def stats(self):
        """Goes straight into the paper's Experimental Setup table."""
        return {
            "n_tiles": self.n,
            "dim": self.dim,
            "backend": self.backend,
            "build_time_s": round(self.build_time_s, 4),
            "storage_mb": round(self.storage_mb, 3),
            "model": self.provenance.get("name", "unknown"),
            "model_licence": self.provenance.get("licence", "unknown"),
        }

    # ----------------------------------------------------------------- search
    def search(self, query_vec, k=5, subset=None):
        """Return [(row_index, score)] ranked by cosine similarity.

        `subset` restricts the search to given row indices -- this is how
        Person B's metadata filters (AOI / date / sensor) compose with
        semantic ranking instead of fighting it.
        """
        q = np.ascontiguousarray(np.atleast_2d(query_vec).astype(np.float32))

        if subset is not None:
            subset = np.asarray(list(subset), dtype=np.int64)
            if subset.size == 0:
                return []
            scores = (self.vectors[subset] @ q.T).ravel()
            order = np.argsort(-scores)[:k]
            return [(int(subset[i]), float(scores[i])) for i in order]

        if self.backend == "faiss":
            scores, idx = self._index.search(q, min(k, self.n))
            return [(int(i), float(s)) for i, s in zip(idx[0], scores[0]) if i != -1]

        scores = (self.vectors @ q.T).ravel()
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order]

    def timed_search(self, query_vec, k=5, repeats=20):
        """Median query latency in milliseconds -- required by the PS report."""
        t = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            self.search(query_vec, k)
            t.append((time.perf_counter() - t0) * 1000)
        return float(np.median(t))

    # ------------------------------------------------------------ persistence
    def save(self, directory):
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "vectors.npy", self.vectors)
        with open(d / "metadata.pkl", "wb") as fh:
            pickle.dump(self.metadata, fh)
        (d / "provenance.json").write_text(json.dumps(self.provenance, indent=2))
        (d / "stats.json").write_text(json.dumps(self.stats(), indent=2))
        return d

    @classmethod
    def load(cls, directory):
        d = Path(directory)
        vectors = np.load(d / "vectors.npy")
        with open(d / "metadata.pkl", "rb") as fh:
            metadata = pickle.load(fh)
        prov_path = d / "provenance.json"
        prov = json.loads(prov_path.read_text()) if prov_path.exists() else {}
        return cls(vectors, metadata, prov)


def build_index(tiles, embedder=None, vectors=None):
    """Convenience: tiles -> embeddings -> SharedIndex (metadata carried through)."""
    from .embedder import Embedder, embed_tiles
    embedder = embedder or Embedder()
    if vectors is None:
        vectors = embed_tiles(tiles, embedder)
    metadata = [t.metadata() if hasattr(t, "metadata") else {"i": i}
                for i, t in enumerate(tiles)]
    return SharedIndex(vectors, metadata, embedder.provenance())
