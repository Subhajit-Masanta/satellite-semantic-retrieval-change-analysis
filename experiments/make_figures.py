"""Build every report figure from the saved real results. Run after
run_oscd.py and extra_analysis.py.

    python experiments/make_figures.py
"""

import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt                  # noqa: E402

os.environ["RUN_LOG"] = "figs_log.txt"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import run_oscd as X                             # noqa: E402
from src import change_detect as cd              # noqa: E402
from src.suppress import radiometric_normalise, cloud_shadow_mask  # noqa: E402

RES = ROOT / "results"
FIG = RES / "figs"
FIG.mkdir(exist_ok=True)
R = json.loads((RES / "results.json").read_text())
E = json.loads((RES / "extra_results.json").read_text())

TEAL, NAVY, GOLD, GREEN, RED, GREY = "#208492", "#1C5C91", "#C89132", "#2E7D5A", "#B53A3A", "#9FB3C8"
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save(fig, name):
    fig.savefig(FIG / name, dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("  ", name)


# ------------------------------------------------------ terminal captures
def terminal(lines, name, cmd, width=112):
    lines = [f"$ {cmd}"] + [l.rstrip("\n")[:width] for l in lines]
    h = 0.172 * len(lines) + 0.5
    fig = plt.figure(figsize=(10.2 * max(1.0, width / 112), h), facecolor="#111820")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_facecolor("#111820")
    for i, l in enumerate(lines):
        col = "#7FD6A0" if i == 0 else ("#F2C572" if l.strip().startswith("[") else "#D7DEE6")
        ax.text(0.012, 1 - (i + 0.9) / (len(lines) + 0.8), l, family="DejaVu Sans Mono",
                fontsize=7.3, color=col, va="center", transform=ax.transAxes)
    save(fig, name)


def section(log, start, end=None):
    out, on = [], False
    for l in log:
        if l.startswith(start):
            on = True
        elif on and end and l.startswith(end):
            break
        if on:
            out.append(l)
    while out and not out[-1].strip():
        out.pop()
    return out


log = (RES / "run_log.txt").read_text(encoding="utf-8").splitlines()
xlog = (RES / "extra_log.txt").read_text(encoding="utf-8").splitlines()
CMD = "python experiments/run_oscd.py"
terminal(section(log, "[1]", "[2]"), "term_1_dataset.png", CMD)
terminal(section(log, "[2]", "[4]"), "term_2_tiles_index.png", CMD)
terminal(section(log, "[4]", "[7]"), "term_3_retrieval.png", CMD)
terminal(section(log, "[7]", "Total"), "term_4_change.png", CMD, width=128)
terminal(section(xlog, "[A]", "[B]"), "term_5_change_size.png",
         "python experiments/extra_analysis.py", width=132)
terminal(section(xlog, "[B]", "[C]") + [""] + section(xlog, "[D]"), "term_6_gatec.png",
         "python experiments/extra_analysis.py", width=118)

# ----------------------------------------------------------- load imagery
cities = {c: X.load_city(c) for c in X.TRAIN + X.TEST}
tiles, meta = [], []
for c, d in cities.items():
    h, w = d["label"].shape
    for (y, x) in X.tile_grid(h, w):
        for date in (1, 2):
            tiles.append(d[f"img{date}"][y:y + X.TILE, x:x + X.TILE])
            meta.append({"city": c, "date": d["dates"].get(f"date_{date}"), "y": y, "x": x,
                         **X.spectral_tags(d["bands"][date], y, x)})
assert len(tiles) == R["n_tiles"]

# ------------------------------------------------- dataset sample pairs
fig, ax = plt.subplots(3, 3, figsize=(8.6, 8.4), facecolor="white")
for r, c in enumerate(("lasvegas", "montpellier", "beirut")):
    d = cities[c]
    for k, (im, t) in enumerate(((rgb(d["img1"]), f"{c}  {d['dates']['date_1']}"),
                                 (rgb(d["img2"]), f"{c}  {d['dates']['date_2']}"),
                                 (d["label"], "ground-truth change mask"))):
        ax[r, k].imshow(im, cmap="gray"); ax[r, k].set_title(t, fontsize=9); ax[r, k].axis("off")
plt.tight_layout(); save(fig, "fig_dataset_samples.png")

# ------------------------------------------------------- tiling picture
d = cities["montpellier"]; im = rgb(d["img1"]).copy(); h, w = d["label"].shape
fig, ax = plt.subplots(1, 2, figsize=(8.6, 4.3), facecolor="white",
                       gridspec_kw={"width_ratios": [1.5, 1]})
ax[0].imshow(im)
for (y, x) in X.tile_grid(h, w):
    ax[0].add_patch(plt.Rectangle((x, y), X.TILE, X.TILE, fill=False, ec="yellow", lw=0.7))
ax[0].add_patch(plt.Rectangle((128, 128), X.TILE, X.TILE, fill=False, ec="red", lw=2))
ax[0].set_title(f"montpellier cut into {len(X.tile_grid(h, w))} tiles of 64x64 px", fontsize=9)
ax[0].axis("off")
ax[1].imshow(im[128:192, 128:192], interpolation="nearest")
ax[1].set_title("one tile (64 px = 640 m on the ground)", fontsize=9); ax[1].axis("off")
plt.tight_layout(); save(fig, "fig_tiling.png")

# ------------------------------------------------------- retrieval grids
def grid(backend, key, name):
    top = R["retrieval"][backend][key]["top10"]
    fig, ax = plt.subplots(2, 5, figsize=(8.8, 4.4), facecolor="white")
    for a, i in zip(ax.ravel(), top):
        a.imshow(rgb(tiles[i]), interpolation="nearest")
        ok = meta[i][key]
        for s in a.spines.values():
            s.set_edgecolor(GREEN if ok else RED); s.set_linewidth(3)
        a.set_xticks([]); a.set_yticks([])
        a.set_title(f"{meta[i]['city']}\n{'relevant' if ok else 'not relevant'}", fontsize=7.5,
                    color=GREEN if ok else RED)
    fig.suptitle(f'{backend}: top 10 for "{X.QUERIES[key]}"', fontsize=9.5)
    plt.tight_layout(h_pad=2.2); save(fig, name)


grid("openclip", "builtup", "fig_retrieval_builtup_openclip.png")
grid("remoteclip", "water", "fig_retrieval_water_remoteclip.png")

# -------------------------------------------- same place across dates
V = np.load(RES / "embeddings_remoteclip.npy")
i1 = np.arange(0, len(tiles), 2)
sims = V[i1] @ V[i1 + 1].T
rank = (sims > np.diag(sims)[:, None]).sum(1)
rng = np.random.default_rng(3)
hit = int(rng.choice(np.flatnonzero(rank == 0)))
miss = int(rng.choice(np.flatnonzero(rank > 20)))
fig, ax = plt.subplots(2, 6, figsize=(9.4, 3.6), facecolor="white")
for r, q in enumerate((hit, miss)):
    ax[r, 0].imshow(rgb(tiles[i1[q]]), interpolation="nearest")
    ax[r, 0].set_title(f"query (date 1)\n{meta[i1[q]]['city']}", fontsize=7.5)
    for k, j in enumerate(np.argsort(-sims[q])[:5]):
        a = ax[r, k + 1]; a.imshow(rgb(tiles[i1[j] + 1]), interpolation="nearest")
        ok = j == q
        for s in a.spines.values():
            s.set_edgecolor(GREEN if ok else GREY); s.set_linewidth(3 if ok else 1)
        a.set_title(f"#{k+1} {meta[i1[j]]['city']}\n{'SAME PLACE' if ok else ''}", fontsize=7.5,
                    color=GREEN if ok else "black")
    for a in ax[r]:
        a.set_xticks([]); a.set_yticks([])
ax[0, 0].set_ylabel("found", fontsize=8); ax[1, 0].set_ylabel("missed", fontsize=8)
plt.tight_layout(); save(fig, "fig_cross_date.png")

# --------------------------------------------- tile change: hist + ROC
from sklearn.metrics import roc_curve            # noqa: E402
S = np.load(RES / "tile_scores.npy"); y = S[0] > 0.5
names = R["tile_change"]["score_order"]
fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.7), facecolor="white")
dist = S[1 + names.index("embedding distance (openclip)")]
bins = np.linspace(0, np.quantile(dist, 0.995), 40)
ax[0].hist(dist[~y], bins, color=GREY, alpha=0.9, density=True, label=f"unchanged tiles ({int((~y).sum())})")
ax[0].hist(dist[y], bins, color=RED, alpha=0.55, density=True, label=f"changed tiles ({int(y.sum())})")
ax[0].set_xlabel("embedding distance between date 1 and date 2 (CLIP)"); ax[0].set_ylabel("density")
ax[0].legend(fontsize=8, frameon=False); ax[0].set_title("The two groups overlap heavily", fontsize=9.5)
for n, c in zip(names, (GREY, NAVY, GOLD, RED)):
    f, t, _ = roc_curve(y, S[1 + names.index(n)])
    ax[1].plot(f, t, color=c, lw=1.8, label=f"{n}  AUC={R['tile_change']['methods'][n]['auc']:.2f}")
ax[1].plot([0, 1], [0, 1], ":", color="black", lw=0.8)
ax[1].set_xlabel("false positive rate"); ax[1].set_ylabel("true positive rate")
ax[1].legend(fontsize=7, frameon=False, loc="lower right"); ax[1].set_title("ROC for tile-level change", fontsize=9.5)
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); save(fig, "fig_tile_change.png")

# ---------------------------------------------------- AUC vs change size
fig, ax = plt.subplots(figsize=(6.6, 3.6), facecolor="white")
xs = [float(k) * 100 for k in E["by_change_size"]]
for n, c in zip(names, (GREY, NAVY, GOLD, RED)):
    ax.plot(range(len(xs)), [E["by_change_size"][k]["auc"][n] for k in E["by_change_size"]],
            "o-", color=c, lw=1.8, label=n)
ax.axhline(0.5, ls=":", color="black", lw=0.8)
ax.set_xticks(range(len(xs)))
ax.set_xticklabels([f">{x:.0f}%\n(n={E['by_change_size'][k]['n_changed']})"
                    for x, k in zip(xs, E["by_change_size"])], fontsize=8)
ax.set_xlabel("how much of the tile changed"); ax.set_ylabel("AUC")
ax.legend(fontsize=7.5, frameon=False); ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); save(fig, "fig_auc_vs_size.png")

# ------------------------------------------------------- pixel bar chart
P = dict(R["pixel"]["test"])
P["Ours, brightness matching only"] = E["brightness_only"]["test"]
order = ["Pixel difference + Otsu", "SSIM differencing", "Change Vector Analysis",
         "Ours, no suppression", "Ours, with suppression", "Ours, brightness matching only"]
fig, ax = plt.subplots(figsize=(9.0, 3.9), facecolor="white")
xx = np.arange(len(order)); wd = 0.27
for k, (m, c) in enumerate((("precision", NAVY), ("recall", GREY), ("f1", TEAL))):
    vals = [P[o][m] for o in order]
    b = ax.bar(xx + (k - 1) * wd, vals, wd, color=c, label=m.capitalize() if m != "f1" else "F1")
    for bb, v in zip(b, vals):
        ax.text(bb.get_x() + bb.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
ax.set_xticks(xx); ax.set_xticklabels([o.replace(", ", ",\n").replace(" + ", " +\n") for o in order], fontsize=8)
ax.set_ylabel("score on the 10 test cities"); ax.legend(frameon=False, fontsize=8, ncol=3)
ax.spines[["top", "right"]].set_visible(False); ax.set_ylim(0, 1.0)
plt.tight_layout(); save(fig, "fig_pixel_bars.png")

# ------------------------------------------- Gate C at matched thresholds
G = E["gate_c_matched"]; th = [int(k) for k in G]
fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.5), facecolor="white")
ax[0].plot(th, [G[str(t)]["off"]["fpr_pct"] for t in th], "o-", color=RED, label="without suppression")
ax[0].plot(th, [G[str(t)]["on"]["fpr_pct"] for t in th], "o-", color=TEAL, label="with suppression")
ax[0].set_ylabel("false alarms (% of unchanged pixels)"); ax[0].set_title("False alarms go down", fontsize=9.5)
ax[1].plot(th, [G[str(t)]["off"]["recall"] for t in th], "o-", color=RED, label="without suppression")
ax[1].plot(th, [G[str(t)]["on"]["recall"] for t in th], "o-", color=TEAL, label="with suppression")
ax[1].plot(th, [0.8 * G[str(t)]["off"]["recall"] for t in th], ":", color="black", lw=1, label="80% floor (Gate C)")
ax[1].set_ylabel("recall (real change found)"); ax[1].set_title("...but too much real change is lost", fontsize=9.5)
for a in ax:
    a.set_xlabel("pixel threshold"); a.legend(fontsize=7.5, frameon=False)
    a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); save(fig, "fig_gatec.png")

# ----------------------------------------------- qualitative change maps
MB = np.load(RES / "masks_brightness_only.npz")
MT = np.load(RES / "masks_test.npz")


def panel(c, name):
    d = cities[c]
    cva = MT[f"Change Vector Analysis||{c}"]
    ours = MB[c]
    f1o = E["brightness_only"]["test"]["per_city"][c]["f1"]
    f1c = R["pixel"]["test"]["Change Vector Analysis"]["per_city"][c]["f1"]
    fig, ax = plt.subplots(1, 5, figsize=(11.5, 2.9), facecolor="white")
    for a, (im, t) in zip(ax, ((rgb(d["img1"]), f"{c} {d['dates']['date_1']}"),
                               (rgb(d["img2"]), d["dates"]["date_2"]),
                               (cva, f"CVA baseline (F1 {f1c:.2f})"),
                               (ours, f"ours, brightness matching (F1 {f1o:.2f})"),
                               (d["label"], "ground truth"))):
        a.imshow(im, cmap="gray"); a.set_title(t, fontsize=8.5); a.axis("off")
    plt.tight_layout(); save(fig, name)


panel("lasvegas", "fig_change_good.png")
panel("valencia", "fig_change_bad.png")

# -------------------------------- what brightness matching does (visual)
c = "brasilia"; d = cities[c]
raw = cd.change_mask(d["img1"], d["img2"], 40)
nrm = cd.change_mask(d["img1"], radiometric_normalise(d["img1"], d["img2"]), 40)
fig, ax = plt.subplots(1, 5, figsize=(11.5, 2.9), facecolor="white")
for a, (im, t) in zip(ax, ((rgb(d["img1"]), f"{c} {d['dates']['date_1']}"),
                           (rgb(d["img2"]), d["dates"]["date_2"]),
                           (raw, f"no brightness matching\n{100*(raw>0).mean():.0f}% of image flagged"),
                           (nrm, f"with brightness matching\n{100*(nrm>0).mean():.0f}% of image flagged"),
                           (d["label"], f"ground truth\n{100*(d['label']>0).mean():.1f}% really changed"))):
    a.imshow(im, cmap="gray"); a.set_title(t, fontsize=8.5); a.axis("off")
plt.tight_layout(); save(fig, "fig_brightness_matching.png")

# ------------------------------------------- cloud mask firing wrongly
c = "rio"; d = cities[c]
um = cv2.bitwise_and(cloud_shadow_mask(d["img1"]), cloud_shadow_mask(d["img2"]))
ov = rgb(d["img2"]).copy(); ov[um == 0] = (0.4 * ov[um == 0] + 0.6 * np.array([230, 60, 60])).astype(np.uint8)
fig, ax = plt.subplots(1, 3, figsize=(8.8, 3.4), facecolor="white")
for a, (im, t) in zip(ax, ((rgb(d["img1"]), f"{c} {d['dates']['date_1']}"),
                           (rgb(d["img2"]), d["dates"]["date_2"]),
                           (ov, f"red = removed by our cloud/shadow mask\n({100*(um==0).mean():.0f}% of the image: dark water and dark vegetation, not cloud)"))):
    a.imshow(im); a.set_title(t, fontsize=8.5); a.axis("off")
plt.tight_layout(); save(fig, "fig_cloudmask_wrong.png")

print("done ->", FIG)
