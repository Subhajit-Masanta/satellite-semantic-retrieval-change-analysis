"""Text search and image search on EuroSAT (Sentinel-2 patches, human labels).

    python experiments/run_eurosat.py

EuroSAT has 27,000 patches of 64x64 px in 10 land-cover classes. We take a
fixed random sample of PER_CLASS patches per class so the run fits on a laptop
CPU. The text prompts below were written once, before running, and not tuned.

Saves results/eurosat_results.json, results/eurosat_log.txt and figures.
"""

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("RUN_LOG", "eurosat_log.txt")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import metrics as M                    # noqa: E402
from src.embedder import Embedder               # noqa: E402

DATA = ROOT / "data" / "EuroSAT"
OUT = ROOT / "results"
FIG = OUT / "figs"
FIG.mkdir(parents=True, exist_ok=True)
LOG = open(OUT / os.environ["RUN_LOG"], "w", encoding="utf-8")
PER_CLASS = int(os.environ.get("PER_CLASS", 100000))   # default: every patch
SEED = 0

PROMPTS = {
    "AnnualCrop": "a satellite image of annual crop land",
    "Forest": "a satellite image of a forest",
    "HerbaceousVegetation": "a satellite image of herbaceous vegetation",
    "Highway": "a satellite image of a highway or road",
    "Industrial": "a satellite image of industrial buildings",
    "Pasture": "a satellite image of pasture land",
    "PermanentCrop": "a satellite image of permanent crop land",
    "Residential": "a satellite image of residential buildings",
    "River": "a satellite image of a river",
    "SeaLake": "a satellite image of a sea or lake",
}


# Second, declared-in-advance variant: average the text embedding over five
# standard caption templates (the usual CLIP "prompt ensemble"). Both variants
# are always reported; we do not pick whichever looks better.
PHRASES = {c: p.replace("a satellite image of ", "") for c, p in PROMPTS.items()}
TEMPLATES = ["a satellite image of {}", "a satellite photo of {}",
             "an aerial image of {}", "a remote sensing image of {}",
             "a centered satellite photo of {}"]


def say(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    LOG.write(line + "\n")
    LOG.flush()


def main():
    t_all = time.perf_counter()
    say("=" * 70)
    say("EUROSAT EXPERIMENT RUN")
    say("=" * 70)
    root = next(p.parent for p in DATA.rglob("Forest") if p.is_dir())
    classes = sorted(PROMPTS)
    rng = np.random.default_rng(SEED)

    say("\n[1] Loading dataset")
    paths, labels, total = [], [], 0
    for ci, c in enumerate(classes):
        files = sorted((root / c).glob("*.jpg"))
        total += len(files)
        pick = rng.choice(len(files), min(PER_CLASS, len(files)), replace=False)
        for i in sorted(pick):
            paths.append(files[i]); labels.append(ci)
        say(f"   {c:22s} {len(files):5d} patches in dataset, {len(pick)} sampled")
    labels = np.array(labels)
    imgs = [cv2.imread(str(p)) for p in paths]
    say(f"   dataset has {total} patches; we use {len(imgs)} "
        f"(seed {SEED}); patch size {imgs[0].shape[1]}x{imgs[0].shape[0]} px")

    R = {"n_total": total, "n_used": len(imgs), "per_class": PER_CLASS,
         "classes": classes, "prompts": PROMPTS, "models": {}}
    keep = {}
    for backend in ("openclip", "remoteclip"):
        say(f"\n[2] {backend}: embedding")
        e = Embedder(backend=backend)
        t0 = time.perf_counter()
        V = np.concatenate([e.encode_images(imgs[i:i + 64])
                            for i in range(0, len(imgs), 64)])
        T = e.encode_text([PROMPTS[c] for c in classes])
        TE = np.stack([e.encode_text([t.format(PHRASES[c]) for t in TEMPLATES]).mean(0)
                       for c in classes])
        TE = TE / np.linalg.norm(TE, axis=1, keepdims=True)
        SE = V @ TE.T
        ens_ap = {c: round(M.average_precision(
            np.argsort(-SE[:, ci]).tolist(), set(np.flatnonzero(labels == ci).tolist())), 4)
            for ci, c in enumerate(classes)}
        ens_acc = float((SE.argmax(1) == labels).mean())
        say(f"   {len(imgs)} patches in {time.perf_counter()-t0:.0f}s, dim {V.shape[1]}")

        S = V @ T.T                                   # (n_images, n_classes)
        say(f"[3] {backend}: text search (one query per class)")
        say(f"   {'query class':22s} {'P@10':>5s} {'P@20':>5s} {'P@100':>6s} {'AP':>6s}")
        per = {}
        for ci, c in enumerate(classes):
            order = np.argsort(-S[:, ci])
            rel = set(np.flatnonzero(labels == ci).tolist())
            per[c] = {"p10": M.precision_at_k(order, rel, 10),
                      "p20": M.precision_at_k(order, rel, 20),
                      "p100": M.precision_at_k(order, rel, 100),
                      "ap": round(M.average_precision(order.tolist(), rel), 4),
                      "top10": [int(i) for i in order[:10]]}
            say(f"   {c:22s} {per[c]['p10']:5.2f} {per[c]['p20']:5.2f} "
                f"{per[c]['p100']:6.2f} {per[c]['ap']:6.3f}")
        mAP = float(np.mean([per[c]["ap"] for c in classes]))
        mp10 = float(np.mean([per[c]["p10"] for c in classes]))
        say(f"   mean P@10 = {mp10:.3f}   mAP = {mAP:.3f}   (random: 0.100)")

        pred = S.argmax(1)
        acc = float((pred == labels).mean())
        conf = np.zeros((len(classes), len(classes)), int)
        for t, p in zip(labels, pred):
            conf[t, p] += 1
        say(f"[4] {backend}: zero-shot classification accuracy = {100*acc:.1f}%  (random: 10.0%)")
        say("   per class: " + ", ".join(
            f"{c}:{100*conf[i, i]/conf[i].sum():.0f}%" for i, c in enumerate(classes)))

        nn = np.zeros((len(V), 10), dtype=np.int64)      # chunked: 27k x 27k is too big
        for a in range(0, len(V), 2000):
            sims = V[a:a + 2000] @ V.T
            sims[np.arange(sims.shape[0]), np.arange(a, a + sims.shape[0])] = -np.inf
            nn[a:a + 2000] = np.argsort(-sims, axis=1)[:, :10]
        same = (labels[nn] == labels[:, None])
        i2i = float(same.mean())
        i2i_top1 = float(same[:, 0].mean())
        say(f"   with 5-template prompt ensemble: mAP = {np.mean(list(ens_ap.values())):.3f}, "
            f"zero-shot accuracy = {100*ens_acc:.1f}%")
        say(f"[5] {backend}: image-to-image search, same class in top 10 = {100*i2i:.1f}%, "
            f"nearest neighbour same class = {100*i2i_top1:.1f}%  (random: 10.0%)")

        R["models"][backend] = {
            "per_class": per, "mAP": round(mAP, 4), "mean_p10": round(mp10, 4),
            "zero_shot_acc": round(acc, 4), "confusion": conf.tolist(),
            "per_class_acc": {c: round(conf[i, i] / conf[i].sum(), 4)
                              for i, c in enumerate(classes)},
            "ensemble": {"templates": TEMPLATES, "per_class_ap": ens_ap,
                         "mAP": round(float(np.mean(list(ens_ap.values()))), 4),
                         "zero_shot_acc": round(ens_acc, 4)},
            "image_search_p10": round(i2i, 4), "image_search_top1": round(i2i_top1, 4),
            "provenance": e.provenance()}
        keep[backend] = per

    R["runtime_s"] = round(time.perf_counter() - t_all, 1)
    (OUT / "eurosat_results.json").write_text(json.dumps(R, indent=2))
    say(f"\nTotal runtime: {R['runtime_s']} s")
    say(f"Saved {OUT / 'eurosat_results.json'}")

    # ------------------------------------------------------------ figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    TEAL, NAVY, GREEN, RED = "#208492", "#1C5C91", "#2E7D5A", "#B53A3A"

    fig, ax = plt.subplots(figsize=(9.0, 3.8), facecolor="white")
    x = np.arange(len(classes)); w = 0.38
    for k, (b, col) in enumerate((("openclip", NAVY), ("remoteclip", TEAL))):
        vals = [R["models"][b]["per_class"][c]["ap"] for c in classes]
        bars = ax.bar(x + (k - 0.5) * w, vals, w, color=col,
                      label=f"{b} (mAP {R['models'][b]['mAP']:.2f})")
        for bb, v in zip(bars, vals):
            ax.text(bb.get_x() + bb.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=6.5)
    ax.axhline(0.1, ls=":", color="black", lw=1, label="random (0.10)")
    ax.set_xticks(x); ax.set_xticklabels(classes, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("average precision of the text query"); ax.set_ylim(0, 1.08)
    ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); fig.savefig(FIG / "eurosat_ap.png", dpi=170, facecolor="white"); plt.close(fig)

    best = max(R["models"], key=lambda b: R["models"][b]["mAP"])
    aps = {c: R["models"][best]["per_class"][c]["ap"] for c in classes}
    show = [max(aps, key=aps.get), sorted(aps, key=aps.get)[len(aps) // 2], min(aps, key=aps.get)]
    fig, ax = plt.subplots(3, 10, figsize=(10.5, 3.9), facecolor="white")
    for r, c in enumerate(show):
        ci = classes.index(c)
        for k, i in enumerate(R["models"][best]["per_class"][c]["top10"]):
            a = ax[r, k]; a.imshow(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB), interpolation="nearest")
            ok = labels[i] == ci
            for s_ in a.spines.values():
                s_.set_edgecolor(GREEN if ok else RED); s_.set_linewidth(2.5)
            a.set_xticks([]); a.set_yticks([])
            a.set_title(classes[labels[i]][:11], fontsize=6.5, color=GREEN if ok else RED)
        ax[r, 0].set_ylabel(f"{c}\nAP {aps[c]:.2f}", fontsize=7.5)
    fig.suptitle(f"{best}: top 10 patches for three text queries "
                 "(best, middle and worst class; green = correct label)", fontsize=9)
    plt.tight_layout(); fig.savefig(FIG / "eurosat_grid.png", dpi=170, facecolor="white"); plt.close(fig)

    conf = np.array(R["models"][best]["confusion"], float)
    conf = conf / conf.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.6, 4.9), facecolor="white")
    ax.imshow(conf, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if conf[i, j] >= 0.05:
                ax.text(j, i, f"{conf[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if conf[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=7); ax.set_yticklabels(classes, fontsize=7)
    ax.set_xlabel("predicted from text prompts"); ax.set_ylabel("true class")
    ax.set_title(f"{best}: zero-shot confusion matrix "
                 f"(accuracy {100*R['models'][best]['zero_shot_acc']:.0f}%)", fontsize=9)
    plt.tight_layout(); fig.savefig(FIG / "eurosat_confusion.png", dpi=170, facecolor="white"); plt.close(fig)
    say("Figures: eurosat_ap.png, eurosat_grid.png, eurosat_confusion.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
