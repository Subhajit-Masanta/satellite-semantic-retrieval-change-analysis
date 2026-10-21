"""Stage 6b -- figures.   OWNER: Person D

Everything that ends up in the paper or on a slide. Saves to disk rather than
calling plt.show(), so the same code works in Colab, in a script, and headless.
"""

from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

PALETTE = {"ours": "#208492", "baseline": "#9FB3C8",
           "accent": "#C89132", "ink": "#1F2D3D"}


def _bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _save(fig, out_path):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(out)


def change_panel(img_t1, img_t2, pred_mask, true_mask=None, out_path="figs/change.png",
                 title=None):
    """Before | after | predicted | (ground truth) -- the core results figure."""
    n = 4 if true_mask is not None else 3
    fig, ax = plt.subplots(1, n, figsize=(4 * n, 4.2))
    ax[0].imshow(_bgr2rgb(img_t1)); ax[0].set_title("t1 (before)")
    ax[1].imshow(_bgr2rgb(img_t2)); ax[1].set_title("t2 (after)")

    overlay = _bgr2rgb(img_t2).copy()
    overlay[pred_mask > 0] = (0.45 * overlay[pred_mask > 0]
                              + 0.55 * np.array([230, 60, 60])).astype(np.uint8)
    ax[2].imshow(overlay); ax[2].set_title("detected change")

    if true_mask is not None:
        ax[3].imshow(true_mask, cmap="gray"); ax[3].set_title("ground truth")

    for a in ax:
        a.axis("off")
    if title:
        fig.suptitle(title, fontsize=13, color=PALETTE["ink"])
    return _save(fig, out_path)


def suppression_panel(img_t1, img_t2, mask_naive, mask_ours,
                      out_path="figs/suppression.png"):
    """The Gate C figure: same pair, with and without suppression.

    On a no-change pair the third panel should be visibly emptier than the
    second. This is the single most persuasive image in the report.
    """
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))
    ax[0].imshow(_bgr2rgb(img_t1)); ax[0].set_title("t1")
    ax[1].imshow(_bgr2rgb(img_t2)); ax[1].set_title("t2")
    ax[2].imshow(mask_naive, cmap="gray")
    ax[2].set_title(f"no suppression\n{int((mask_naive > 0).sum())} px flagged")
    ax[3].imshow(mask_ours, cmap="gray")
    ax[3].set_title(f"with suppression\n{int((mask_ours > 0).sum())} px flagged")
    for a in ax:
        a.axis("off")
    return _save(fig, out_path)


def retrieval_grid(query, tiles, scores, out_path="figs/retrieval.png", cols=5):
    """Top-k tiles returned for one text query, with similarity scores."""
    k = len(tiles)
    rows = int(np.ceil(k / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(3 * cols, 3.2 * rows))
    ax = np.atleast_1d(ax).ravel()
    for i in range(len(ax)):
        ax[i].axis("off")
        if i < k:
            ax[i].imshow(_bgr2rgb(tiles[i]))
            ax[i].set_title(f"#{i + 1}   {scores[i]:.3f}", fontsize=10)
    fig.suptitle(f'query: "{query}"', fontsize=13, color=PALETTE["ink"])
    return _save(fig, out_path)


def baseline_bars(labels, values, ylabel, out_path="figs/baselines.png",
                  ours_label="Ours", title=None):
    """Baseline comparison bar chart. Our bar is highlighted."""
    colors = [PALETTE["ours"] if ours_label.lower() in str(l).lower()
              else PALETTE["baseline"] for l in labels]
    fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 2, 4.2))
    bars = ax.bar(range(len(labels)), values, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9, color=PALETTE["ink"])
    if title:
        ax.set_title(title, color=PALETTE["ink"])
    return _save(fig, out_path)


def precision_recall_tradeoff(thresholds, precisions, recalls,
                              out_path="figs/tradeoff.png"):
    """Shows the precision-first choice explicitly -- honest about what recall
    is being given up to get there."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(thresholds, precisions, "o-", color=PALETTE["ours"], label="precision")
    ax.plot(thresholds, recalls, "s--", color=PALETTE["accent"], label="recall")
    ax.set_xlabel("change threshold")
    ax.set_ylabel("score")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)
    return _save(fig, out_path)
