"""Generate the four figures the Project-I report template requires."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path

OUT = Path(r"C:\Users\subha\Desktop\projects\Computer Vision\report_figs")
OUT.mkdir(exist_ok=True)

TEAL = "#208492"; NAVY = "#1C5C91"; GOLD = "#C89132"; GREEN = "#2E7D5A"
CARD = "#EEF3F7"; INK = "#1F2D3D"; GREY = "#8FA3B5"; RED = "#B53A3A"

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]


def box(ax, x, y, w, h, text, fc=CARD, ec=NAVY, tc=INK, fs=9, bold=False, r=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.004,rounding_size={r}",
                                fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, weight="bold" if bold else "normal", wrap=True)


def arrow(ax, x1, y1, x2, y2, color=NAVY, style="-|>", lw=1.2, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=12, lw=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))


# ============================================================ Fig 1: Gantt
def gantt():
    tasks = [
        ("Study the problem & choose topic",          0, 2, NAVY),
        ("Download OSCD / EuroSAT data",        1, 2, NAVY),
        ("Build tiling + embedding + index (A)", 2, 3, TEAL),
        ("Build text & image search (B)",        2, 3, TEAL),
        ("Build change detection (C)",           2, 3, TEAL),
        ("Build metrics & plots (D)",            2, 3, TEAL),
        ("Test each part separately",            4, 2, GOLD),
        ("Join all modules together",            5, 2, GOLD),
        ("Run experiments & collect results",    6, 3, GREEN),
        ("Write report & prepare review",        7, 3, GREEN),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    for i, (name, start, dur, c) in enumerate(tasks):
        y = len(tasks) - i - 1
        ax.barh(y, dur, left=start, height=0.55, color=c, edgecolor="white")
        ax.text(start + dur + 0.12, y, f"W{start+1}-W{start+dur}", va="center",
                fontsize=7.5, color=GREY)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([t[0] for t in reversed(tasks)], fontsize=8.5)
    ax.set_xticks(range(11))
    ax.set_xticklabels([f"W{i+1}" for i in range(11)], fontsize=8)
    ax.set_xlabel("Project weeks", fontsize=9)
    ax.set_xlim(0, 11.6)
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    fig.savefig(OUT / "fig1_gantt.png", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ================================================== Fig 2: System architecture
def architecture():
    fig, ax = plt.subplots(figsize=(9.4, 6.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10.6); ax.axis("off")

    box(ax, 2.6, 9.6, 4.8, 0.7, "Satellite Images\n(two dates of the same place)",
        fc=NAVY, ec=NAVY, tc="white", fs=9.5, bold=True)

    box(ax, 2.6, 8.4, 4.8, 0.75,
        "STEP 1  Cut image into small tiles\n(keep location + date of each tile)",
        fc=CARD, ec=TEAL, fs=9)
    arrow(ax, 5, 9.6, 5, 9.15)

    box(ax, 2.6, 7.2, 4.8, 0.75,
        "STEP 2  Frozen CLIP model turns each tile\ninto one list of numbers (embedding)",
        fc=CARD, ec=TEAL, fs=9)
    arrow(ax, 5, 8.4, 5, 7.95)

    box(ax, 2.9, 6.1, 4.2, 0.7, "SHARED INDEX\n(all embeddings stored together)",
        fc=GOLD, ec=GOLD, tc="white", fs=9.5, bold=True)
    arrow(ax, 5, 7.2, 5, 6.8)

    # two branches
    box(ax, 0.35, 4.3, 4.1, 1.3,
        "STEP 3  SEARCH\n\n• type words, e.g. \"new buildings\"\n"
        "• or give an example tile\n• sort tiles by similarity",
        fc=CARD, ec=NAVY, fs=8.5)
    box(ax, 5.55, 4.3, 4.1, 1.3,
        "STEP 4  FIND CHANGE\n\n• compare the SAME tile's numbers\n  at date 1 and date 2\n"
        "• big difference = something changed",
        fc=CARD, ec=NAVY, fs=8.5)
    arrow(ax, 4.2, 6.1, 2.4, 5.6, rad=0.18)
    arrow(ax, 5.8, 6.1, 7.6, 5.6, rad=-0.18)
    ax.text(5.0, 5.82, "same numbers used by both sides", ha="center",
            fontsize=8, style="italic", color=GOLD)

    box(ax, 5.55, 2.9, 4.1, 1.0,
        "STEP 5  REMOVE FALSE ALARMS\ncloud, shadow, season,\nbrightness differences",
        fc=CARD, ec=RED, fs=8.5)
    arrow(ax, 7.6, 4.3, 7.6, 3.9)

    box(ax, 0.35, 1.5, 9.3, 0.9,
        "STEP 6  MEASURE AND SHOW RESULTS\n"
        "Precision@k · mAP · Precision · Recall · F1 · IoU · speed · storage",
        fc=CARD, ec=GREEN, fs=9)
    arrow(ax, 2.4, 4.3, 2.4, 2.4)
    arrow(ax, 7.6, 2.9, 7.6, 2.4)

    box(ax, 2.6, 0.35, 4.8, 0.7, "Search Results + Change Maps",
        fc=GREEN, ec=GREEN, tc="white", fs=9.5, bold=True)
    arrow(ax, 5, 1.5, 5, 1.05)

    plt.tight_layout()
    fig.savefig(OUT / "fig2_architecture.png", dpi=190, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


# ========================================================== Fig 3: DFD level 1
def dfd():
    fig, ax = plt.subplots(figsize=(9.6, 6.0))
    ax.set_xlim(0, 10.4); ax.set_ylim(0, 7.8); ax.axis("off")

    def lab(x, y, t, c=GREY):
        ax.text(x, y, t, fontsize=7.4, color=c, ha="center", style="italic")

    # external entities = squares
    ax.add_patch(Rectangle((0.15, 5.05), 1.5, 0.95, fc=NAVY, ec=NAVY))
    ax.text(0.9, 5.52, "User\n(Analyst)", ha="center", va="center", color="white",
            fontsize=8.5, weight="bold")
    ax.add_patch(Rectangle((0.15, 0.55), 1.5, 0.95, fc=NAVY, ec=NAVY))
    ax.text(0.9, 1.02, "Satellite\nImage Files", ha="center", va="center", color="white",
            fontsize=8.5, weight="bold")

    # processes (rounded) -- bottom row = build, top row = use
    box(ax, 2.35, 0.5, 1.7, 1.05, "1.0\nTile the\nimages", fc=CARD, ec=TEAL, fs=8.5, r=0.3)
    box(ax, 4.75, 0.5, 1.7, 1.05, "2.0\nMake\nembeddings", fc=CARD, ec=TEAL, fs=8.5, r=0.3)
    box(ax, 2.35, 5.0, 1.7, 1.05, "3.0\nSearch\ntiles", fc=CARD, ec=TEAL, fs=8.5, r=0.3)
    box(ax, 5.05, 5.0, 1.7, 1.05, "4.0\nDetect\nchange", fc=CARD, ec=TEAL, fs=8.5, r=0.3)
    box(ax, 7.75, 5.0, 1.9, 1.05, "5.0\nRemove false\nalarms", fc=CARD, ec=RED, fs=8.5, r=0.3)

    # data store D1
    ax.add_patch(Rectangle((3.45, 2.75), 3.0, 0.85, fc="#FFF6E5", ec=GOLD, lw=1.3))
    ax.plot([3.45, 6.45], [3.33, 3.33], color=GOLD, lw=1.0)
    ax.text(3.72, 3.44, "D1", fontsize=8, color=GOLD, weight="bold")
    ax.text(4.95, 3.02, "Shared Embedding Index", ha="center", fontsize=8.5, color=INK)

    # ---- build path (bottom) ----
    arrow(ax, 1.65, 1.02, 2.35, 1.02, GREY);  lab(2.0, 1.17, "raw image")
    arrow(ax, 4.05, 1.02, 4.75, 1.02, GREY);  lab(4.4, 1.17, "tiles")
    arrow(ax, 5.60, 1.55, 4.95, 2.75, GREY, rad=-0.15); lab(4.72, 2.15, "vectors")

    # ---- index is read by BOTH sides ----
    arrow(ax, 4.10, 3.60, 3.20, 5.00, GREY, rad=0.15); lab(3.18, 4.35, "read")
    arrow(ax, 5.80, 3.60, 5.90, 5.00, GREY, rad=-0.12); lab(6.25, 4.35, "read")

    # ---- search path ----  (user asks 3.0, 3.0 answers user)
    arrow(ax, 1.65, 5.78, 2.35, 5.78, GREY); lab(2.0, 5.93, "query")
    arrow(ax, 2.35, 5.27, 1.65, 5.27, GREY); lab(2.0, 4.98, "ranked tiles")

    # ---- change path ----  (5.0 answers the user along the top)
    arrow(ax, 6.75, 5.52, 7.75, 5.52, GREY); lab(7.25, 5.68, "raw change")
    arrow(ax, 8.70, 6.05, 0.90, 6.00, GREY, rad=0.08)
    lab(4.80, 7.00, "change map + confidence")

    ax.text(0.05, 7.60, "Level-1 Data Flow Diagram", fontsize=10, weight="bold", color=NAVY)
    ax.text(0.05, 0.08,
            "Square = outside entity     Rounded = process     Open box = data store",
            fontsize=7.5, color=GREY)
    plt.tight_layout()
    fig.savefig(OUT / "fig3_dfd.png", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ========================================================== Fig 4: Use case
def usecase():
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")

    def actor(x, y, name):
        ax.add_patch(Circle((x, y + 0.72), 0.19, fc="white", ec=INK, lw=1.3))
        ax.plot([x, x], [y + 0.53, y + 0.12], color=INK, lw=1.3)
        ax.plot([x - 0.3, x + 0.3], [y + 0.40, y + 0.40], color=INK, lw=1.3)
        ax.plot([x, x - 0.24], [y + 0.12, y - 0.26], color=INK, lw=1.3)
        ax.plot([x, x + 0.24], [y + 0.12, y - 0.26], color=INK, lw=1.3)
        ax.text(x, y - 0.56, name, ha="center", fontsize=8.5, weight="bold", color=INK)

    actor(0.85, 5.3, "Analyst")
    actor(0.85, 1.9, "Data\nAdmin")

    ax.add_patch(FancyBboxPatch((2.6, 0.5), 5.6, 7.0,
                                boxstyle="round,pad=0.02,rounding_size=0.05",
                                fc="#FAFCFD", ec=GREY, lw=1.1, ls="--"))
    ax.text(5.4, 7.18, "Satellite Search & Change Analysis System",
            ha="center", fontsize=9.5, weight="bold", color=NAVY)

    cases = [
        (5.4, 6.45, "Search tiles by typing words"),
        (5.4, 5.55, "Search by example image"),
        (5.4, 4.65, "Filter by date / area / sensor"),
        (5.4, 3.75, "Compare two dates for change"),
        (5.4, 2.85, "View change type and confidence"),
        (5.4, 1.95, "Load images and build index"),
        (5.4, 1.05, "Export results with source details"),
    ]
    from matplotlib.patches import Ellipse
    for x, y, t in cases:
        ax.add_patch(Ellipse((x, y), 4.6, 0.68, fc=CARD, ec=TEAL, lw=1.1))
        ax.text(x, y, t, ha="center", va="center", fontsize=8.5, color=INK)

    for y in (6.45, 5.55, 4.65, 3.75, 2.85, 1.05):
        ax.plot([1.25, 3.1], [5.3, y], color=GREY, lw=0.9)
    ax.plot([1.25, 3.1], [1.9, 1.95], color=GREY, lw=0.9)
    ax.plot([1.25, 3.1], [1.9, 1.05], color=GREY, lw=0.9)

    plt.tight_layout()
    fig.savefig(OUT / "fig4_usecase.png", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


gantt(); architecture()
print("figures written to", OUT)
for f in sorted(OUT.glob("*.png")):
    print("  ", f.name, f"{f.stat().st_size/1024:.0f} KB")
