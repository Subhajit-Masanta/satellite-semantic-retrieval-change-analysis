"""End-to-end pipeline runner.   OWNER: Person D (integration)

    python main.py --demo                       # synthetic, no dataset needed
    python main.py --data data/LEVIR-CD --limit 200
    python main.py --data data/LEVIR-CD --query "newly built structures near a river"

Chain:  ingest -> embed -> index  ->  {retrieve | change analyse}  -> suppress
        -> classify -> measure -> figures
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from src import change_classify as cc
from src import change_detect as cd
from src import metrics as M
from src import visualizer as V
from src.embedder import Embedder
from src.index_builder import SharedIndex, build_index
from src.ingest import load_pair
from src.retrieval import TEST_QUERIES, explain, search_by_text

OUT = Path("outputs")


def _find_pairs(root, limit=None):
    """Locate LEVIR-CD style A/ B/ label/ triples."""
    root = Path(root)
    a_dir, b_dir, l_dir = root / "A", root / "B", root / "label"
    if not (a_dir.is_dir() and b_dir.is_dir()):
        raise FileNotFoundError(
            f"expected LEVIR-CD layout {root}/A, {root}/B (and optionally /label). "
            "Run  python download_data.py --help  for where to get the data.")
    names = sorted(p.name for p in a_dir.glob("*.png"))
    if limit:
        names = names[:limit]
    return [(a_dir / n, b_dir / n, (l_dir / n) if (l_dir / n).exists() else None)
            for n in names]


def run_demo():
    """Synthetic end-to-end run so the chain is verifiable with no download."""
    from gates import _synthetic_pairs

    print("=" * 62)
    print("DEMO RUN -- synthetic data, no dataset required")
    print("=" * 62)
    base, changed, truth, brighter, cloudy, hard = _synthetic_pairs()

    embedder = Embedder()
    print(f"\n[1] embedder backend : {embedder.backend}")
    if embedder.backend == "hash":
        print("    NOTE: development fallback. Numbers here are not reportable.")

    tiles_t1 = [base, base, base, base]
    tiles_t2 = [changed, brighter, cloudy, hard]
    labels = ["real change", "illumination only", "cloud only", "change + confounds"]

    v1 = embedder.encode_images(tiles_t1)
    v2 = embedder.encode_images(tiles_t2)
    index = SharedIndex(np.concatenate([v1, v2]),
                        [{"tile_id": f"t1_{i}", "date": "2024-01-01"} for i in range(4)]
                        + [{"tile_id": f"t2_{i}", "date": "2026-01-01"} for i in range(4)],
                        embedder.provenance())
    print(f"[2] shared index     : {index.stats()}")

    print("\n[3] retrieval over the shared index")
    for q in TEST_QUERIES[:3]:
        hits = search_by_text(index, q, embedder, k=3)
        print(f"    {q!r:46s} -> rows {[r for r, _ in hits]}")

    print("\n[4] change analysis from the SAME vectors")
    OUT.mkdir(exist_ok=True)
    rows = []
    for i, (a, b, name) in enumerate(zip(tiles_t1, tiles_t2, labels)):
        emb_score = cd.embedding_change_score(v1[i], v2[i])
        mask, conf = cd.analyse_pair(a, b, suppress=True)
        naive, _ = cd.analyse_pair(a, b, suppress=False)
        regions = cc.classify_mask(a, b, mask)
        kinds = cc.summarise(regions) or {"none": 0}
        rows.append([name, f"{emb_score:.3f}", int((naive > 0).sum()),
                     int((mask > 0).sum()), f"{conf:.2f}",
                     ",".join(f"{k}x{v}" for k, v in kinds.items())])
        V.suppression_panel(a, b, naive, mask, OUT / f"demo_{i}_{name.replace(' ', '_')}.png")

    print()
    print(M.format_table(rows, ["pair", "emb Δ", "naive px", "ours px", "conf", "types"]))

    print("\n[5] metrics against ground truth (real-change pair)")
    mask, _ = cd.analyse_pair(base, changed, suppress=True)
    print("   ", {k: round(v, 3) for k, v in M.change_metrics(mask, truth).items()
                  if k in ("precision", "recall", "f1", "iou")})

    V.change_panel(base, changed, mask, truth, OUT / "demo_change_panel.png")
    print(f"\n[6] figures written to {OUT.resolve()}")
    print("\nDemo complete. Run  python gates.py --synthetic  for the validation gates.")
    return 0


def run_dataset(data_root, limit, query, tile_size):
    print("=" * 62)
    print(f"DATASET RUN -- {data_root}")
    print("=" * 62)
    pairs = _find_pairs(data_root, limit)
    print(f"found {len(pairs)} pairs")

    embedder = Embedder()
    print(f"embedder backend: {embedder.backend}")
    if embedder.backend == "hash":
        print("WARNING: hashing fallback -- install torch + open_clip before "
              "reporting any numbers from this run.")

    all_tiles_t1, all_tiles_t2, truths = [], [], []
    t0 = time.perf_counter()
    for a_path, b_path, l_path in pairs:
        t1, t2 = load_pair(a_path, b_path, tile_size=tile_size)
        all_tiles_t1 += t1
        all_tiles_t2 += t2
        gt = cv2.imread(str(l_path), cv2.IMREAD_GRAYSCALE) if l_path else None
        truths.append(gt)
    print(f"tiled into {len(all_tiles_t1)} tile pairs in {time.perf_counter()-t0:.1f}s")

    index = build_index(all_tiles_t1 + all_tiles_t2, embedder)
    print("index stats:", json.dumps(index.stats(), indent=2))

    if query:
        hits = search_by_text(index, query, embedder, k=5)
        print(f"\nquery: {query!r}")
        for h in explain(index, hits):
            print(f"   {h.get('tile_id')}   score={h['score']}")

    per_pair = []
    for (a_path, b_path, l_path), gt in zip(pairs, truths):
        if gt is None:
            continue
        a, b = cv2.imread(str(a_path)), cv2.imread(str(b_path))
        mask, _ = cd.analyse_pair(a, b, suppress=True)
        per_pair.append(M.change_metrics(mask, gt))
    if per_pair:
        print("\nchange metrics:", json.dumps(
            {k: round(v, 4) for k, v in M.aggregate(per_pair).items()}, indent=2))
    else:
        print("\nno label/ masks found -- change metrics skipped "
              "(qualitative figures only, per the dataset validity rule)")

    OUT.mkdir(exist_ok=True)
    (OUT / "run_report.json").write_text(json.dumps({
        "index": index.stats(),
        "provenance": embedder.provenance(),
        "n_pairs": len(pairs),
        "change": M.aggregate(per_pair) if per_pair else None,
        "query_latency_ms": index.timed_search(index.vectors[0]) if index.n else None,
    }, indent=2))
    print(f"\nreport written to {(OUT / 'run_report.json').resolve()}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="synthetic end-to-end run")
    ap.add_argument("--data", help="dataset root (LEVIR-CD layout: A/ B/ label/)")
    ap.add_argument("--limit", type=int, default=50, help="max pairs to process")
    ap.add_argument("--query", help="run one text query against the index")
    ap.add_argument("--tile-size", type=int, default=256)
    args = ap.parse_args()

    if args.demo or not args.data:
        if not args.data:
            print("No --data given; running the synthetic demo.\n")
        return run_demo()
    return run_dataset(args.data, args.limit, args.query, args.tile_size)


if __name__ == "__main__":
    sys.exit(main())
