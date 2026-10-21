"""Stage 6a -- evaluation.   OWNER: Person D

Precision-first, as PS SIH26227 section 2.2.3 requires.

Person D can verify all of this by hand before any real data exists: build a
10x10 ground-truth array with a known overlap and check the arithmetic.
"""

import numpy as np


# ----------------------------------------------------------------- retrieval
def precision_at_k(retrieved, relevant, k=5):
    """Share of the top-k that are relevant. `retrieved` is ranked row ids."""
    if k <= 0:
        return 0.0
    top = list(retrieved)[:k]
    return len(set(top) & set(relevant)) / k


def recall_at_k(retrieved, relevant, k=5):
    relevant = set(relevant)
    if not relevant:
        return 0.0
    return len(set(list(retrieved)[:k]) & relevant) / len(relevant)


def average_precision(retrieved, relevant):
    """AP for one query: mean of precision@i at every rank where a hit lands."""
    relevant = set(relevant)
    if not relevant:
        return 0.0
    hits, total = 0, 0.0
    for i, row in enumerate(retrieved, start=1):
        if row in relevant:
            hits += 1
            total += hits / i
    return total / len(relevant)


def mean_average_precision(rankings, relevances):
    """mAP over several queries. rankings[i] pairs with relevances[i]."""
    if not rankings:
        return 0.0
    return float(np.mean([average_precision(r, rel)
                          for r, rel in zip(rankings, relevances)]))


# ------------------------------------------------------------------- change
def change_metrics(pred_mask, true_mask):
    """Precision is the headline; recall is reported for completeness."""
    p, t = np.asarray(pred_mask) > 0, np.asarray(true_mask) > 0
    tp = float((p & t).sum())
    fp = float((p & ~t).sum())
    fn = float((~p & t).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou,
            "tp": tp, "fp": fp, "fn": fn}


def false_positive_area(pred_mask):
    """On a NO-CHANGE pair every flagged pixel is a false alarm. Gate C metric."""
    return float((np.asarray(pred_mask) > 0).sum())


def false_positive_rate(pred_mask, true_mask):
    """FP as a share of all truly-unchanged pixels."""
    p, t = np.asarray(pred_mask) > 0, np.asarray(true_mask) > 0
    neg = float((~t).sum())
    return float((p & ~t).sum()) / neg if neg else 0.0


def aggregate(per_pair):
    """Mean of each metric across pairs, plus micro-averaged P/R/F1/IoU.

    Micro-averaging pools tp/fp/fn before dividing, so large changed regions
    are not given the same weight as tiny ones. Report micro as the headline
    and macro alongside it -- they answer different questions.
    """
    if not per_pair:
        return {}
    keys = ("precision", "recall", "f1", "iou")
    macro = {f"macro_{k}": float(np.mean([m[k] for m in per_pair])) for k in keys}
    tp = sum(m["tp"] for m in per_pair)
    fp = sum(m["fp"] for m in per_pair)
    fn = sum(m["fn"] for m in per_pair)
    mp = tp / (tp + fp) if tp + fp else 0.0
    mr = tp / (tp + fn) if tp + fn else 0.0
    return {
        **macro,
        "micro_precision": mp,
        "micro_recall": mr,
        "micro_f1": 2 * mp * mr / (mp + mr) if mp + mr else 0.0,
        "micro_iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
        "n_pairs": len(per_pair),
    }


def format_table(rows, headers):
    """Plain-text table for dropping into the report or the terminal."""
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
              for i, h in enumerate(headers)]
    line = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    out = [line, "  ".join("-" * w for w in widths)]
    for r in rows:
        out.append("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))
    return "\n".join(out)
