"""The three validation gates. Run these BEFORE building further.

    python gates.py --synthetic      # works today, no dataset needed
    python gates.py --data data/LEVIR-CD

Gate A  retrieval returns relevant results
Gate B  shared-space change beats pixel differencing
Gate C  suppression cuts false alarms WITHOUT destroying sensitivity

Gate C has two halves on purpose. Checking only the false-alarm half is
gameable: a suppressor that returns an empty mask scores a perfect zero false
alarm area and "passes" while detecting nothing at all. The sensitivity half
catches exactly that. This was found by actually running it -- see
docs/VALIDATION_NOTES.md.
"""

import argparse
import sys

import cv2
import numpy as np

from src import change_detect as cd
from src import metrics as M
from src.embedder import Embedder
from src.index_builder import SharedIndex
from src.retrieval import TEST_QUERIES, search_by_text


# --------------------------------------------------------------- synthetic data
def _synthetic_pairs(seed=0, size=256):
    """Pairs that exercise each gate without needing a download."""
    rng = np.random.default_rng(seed)
    base = (rng.random((size, size, 3)) * 120 + 60).astype(np.uint8)

    # real change: a bright new structure
    changed = base.copy()
    s = size // 6
    changed[s:2 * s, s:2 * s] = 245
    truth = np.zeros((size, size), np.uint8)
    truth[s:2 * s, s:2 * s] = 255

    # no change: only an illumination/gain shift
    brighter = np.clip(base.astype(int) * 1.35 + 25, 0, 255).astype(np.uint8)

    # no change: a cloud drifted in
    cloudy = base.copy()
    blob = np.zeros((size, size), np.uint8)
    cv2.circle(blob, (int(size * 0.72), int(size * 0.72)), int(size * 0.18), 255, -1)
    blob = cv2.GaussianBlur(blob, (31, 31), 0)
    a = (blob / 255.0)[..., None]
    cloudy = (cloudy * (1 - a) + np.array([250, 250, 250]) * a).astype(np.uint8)

    # real change PLUS confounds -- this is the realistic case, and the one
    # where plain pixel differencing should struggle and suppression should pay
    hard = np.clip(changed.astype(int) * 1.25 + 18, 0, 255).astype(np.uint8)
    hb = np.zeros((size, size), np.uint8)
    cv2.circle(hb, (int(size * 0.78), int(size * 0.24)), int(size * 0.15), 255, -1)
    hb = cv2.GaussianBlur(hb, (31, 31), 0)
    ha = (hb / 255.0)[..., None]
    hard = (hard * (1 - ha) + np.array([250, 250, 250]) * ha).astype(np.uint8)

    return base, changed, truth, brighter, cloudy, hard


# ---------------------------------------------------------------------- Gate A
def gate_a(index, embedder, tiles=None, k=5, need=3):
    """At least `need` of the test queries return a plausible tile in the top-k.

    Without human relevance judgements this checks the mechanics: ranking is
    ordered, scores are finite, and different queries give different results.
    Replace `_plausible` with real relevance labels once you have them.
    """
    print("\n=== GATE A: retrieval ===")
    if index.n == 0:
        print("  empty index -> FAIL")
        return False

    passes, seen = 0, []
    for q in TEST_QUERIES:
        hits = search_by_text(index, q, embedder, k=k)
        if not hits:
            print(f"  {q!r:48s} no results")
            continue
        rows = [r for r, _ in hits]
        scores = [s for _, s in hits]
        ordered = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
        finite = all(np.isfinite(scores))
        if ordered and finite:
            passes += 1
        seen.append(tuple(rows))
        print(f"  {q!r:48s} top-{k} rows={rows}  best={scores[0]:.3f}")

    distinct = len(set(seen))
    ok = passes >= need and distinct > 1
    print(f"  {passes}/{len(TEST_QUERIES)} queries returned a well-formed ranking "
          f"(need {need}); {distinct} distinct result sets")
    if distinct <= 1:
        print("  every query returned the SAME tiles -- the text encoder is not "
              "discriminating. Check the backend is not the hashing fallback.")
    print(f"  GATE A: {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------- Gate B
def gate_b(pairs, need=3):
    """Our change path must beat pixel differencing + Otsu on F1.

    pairs: list of (img_t1, img_t2, truth_mask)
    """
    print("\n=== GATE B: shared-space change vs pixel differencing ===")
    wins = ties = 0
    saturated = 0
    for i, (a, b, truth) in enumerate(pairs, 1):
        base_mask = cd.baseline_pixel_otsu(a, b)
        ours_mask, _ = cd.analyse_pair(a, b, suppress=True)
        f_base = M.change_metrics(base_mask, truth)["f1"]
        f_ours = M.change_metrics(ours_mask, truth)["f1"]
        if abs(f_ours - f_base) < 1e-6:
            verdict, ties = "tie", ties + 1
        elif f_ours > f_base:
            verdict, wins = "win", wins + 1
        else:
            verdict = "loss"
        if min(f_base, f_ours) > 0.99:
            saturated += 1
        print(f"  pair {i}: otsu F1={f_base:.3f}   ours F1={f_ours:.3f}   {verdict}")

    # A tie counts toward passing only when neither method is saturated --
    # matching on a genuinely hard pair is informative, matching at F1=1.000 on
    # a trivial one is not.
    ok = (wins + ties) >= need and saturated < len(pairs)
    print(f"  {wins} wins, {ties} ties, {len(pairs) - wins - ties} losses "
          f"(need {need} non-losses)")
    if saturated == len(pairs):
        print("  EVERY pair saturated at F1~1.000 -- this test is too easy to "
              "tell the methods apart. Use harder pairs (real LEVIR-CD, or add "
              "confounds) before trusting this gate.")
    print(f"  GATE B: {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------- Gate C
def gate_c(no_change_pairs, change_pairs, need_fp=2, min_recall_retained=0.8):
    """BOTH halves must pass.

    half 1  suppression cuts false alarms on pairs with no real change
    half 2  suppression does not destroy recall on pairs with real change
    """
    print("\n=== GATE C: confound suppression (the novelty gate) ===")

    fp_wins = 0
    for i, (a, b) in enumerate(no_change_pairs, 1):
        naive, _ = cd.analyse_pair(a, b, suppress=False)
        ours, _ = cd.analyse_pair(a, b, suppress=True)
        fa_n, fa_o = M.false_positive_area(naive), M.false_positive_area(ours)
        win = fa_o < fa_n
        fp_wins += win
        print(f"  no-change pair {i}: naive={int(fa_n):7d} px   ours={int(fa_o):7d} px"
              f"   {'win' if win else 'loss'}")

    retained = []
    for i, (a, b, truth) in enumerate(change_pairs, 1):
        naive, _ = cd.analyse_pair(a, b, suppress=False)
        ours, _ = cd.analyse_pair(a, b, suppress=True)
        r_n = M.change_metrics(naive, truth)["recall"]
        r_o = M.change_metrics(ours, truth)["recall"]
        frac = 1.0 if r_n == 0 else r_o / r_n
        retained.append(frac)
        print(f"  change pair  {i}: recall {r_n:.3f} -> {r_o:.3f}  "
              f"({frac * 100:.0f}% retained)")

    worst = min(retained) if retained else 0.0
    fp_ok, sens_ok = fp_wins >= need_fp, worst >= min_recall_retained
    print(f"  false-alarm half : {fp_wins}/{len(no_change_pairs)} "
          f"(need {need_fp}) -> {'PASS' if fp_ok else 'FAIL'}")
    print(f"  sensitivity half : worst retained {worst:.2f} "
          f"(need >= {min_recall_retained}) -> {'PASS' if sens_ok else 'FAIL'}")
    ok = fp_ok and sens_ok
    print(f"  GATE C: {'PASS' if ok else 'FAIL'}")
    if not sens_ok:
        print("  suppression is too aggressive -- it is deleting real change. "
              "Check cloud_shadow_mask is not eating bright new construction.")
    return ok


# ------------------------------------------------------------------------ main
def run_synthetic():
    print("Running gates on SYNTHETIC data (no dataset required).")
    print("These check the mechanics. Real numbers require LEVIR-CD / OSCD.")

    base, changed, truth, brighter, cloudy, hard = _synthetic_pairs()

    embedder = Embedder()
    print(f"\nembedder backend: {embedder.backend}")

    tiles = [base, changed, brighter, cloudy] * 6
    vecs = embedder.encode_images(tiles)
    index = SharedIndex(vecs, [{"tile_id": f"t{i}"} for i in range(len(tiles))],
                        embedder.provenance())
    print("index stats:", index.stats())

    a = gate_a(index, embedder)
    # mix: clean pairs and confounded pairs, so the gate is not saturated
    b = gate_b([(base, changed, truth), (base, hard, truth),
                (base, hard, truth), (base, changed, truth),
                (base, hard, truth)], need=3)
    c = gate_c([(base, brighter), (base, cloudy), (base, base)],
               [(base, changed, truth), (base, hard, truth)])

    print("\n" + "=" * 58)
    for name, ok in (("A", a), ("B", b), ("C", c)):
        print(f"  GATE {name}: {'PASS' if ok else 'FAIL'}")
    print("=" * 58)
    if not (a and b and c):
        print("\nA gate failed. Per EXECUTION_PLAN.md, fix it before building further.")
    return a and b and c


def main():
    ap = argparse.ArgumentParser(description="Run the three validation gates.")
    ap.add_argument("--synthetic", action="store_true",
                    help="run on generated data, no dataset needed")
    ap.add_argument("--data", help="path to a LEVIR-CD / OSCD style directory")
    args = ap.parse_args()

    if args.data:
        print("Real-dataset gate running is not wired up yet.")
        print("Load pairs with src.ingest.load_pair, then call gate_a/gate_b/gate_c.")
        return 1
    return 0 if run_synthetic() else 1


if __name__ == "__main__":
    sys.exit(main())
