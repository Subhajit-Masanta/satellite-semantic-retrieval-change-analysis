"""Demo app: search satellite tiles by meaning and see what changed.

    run_demo.bat            (Windows, from the project folder)
or  python -m streamlit run demo/app.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import change_types as CT  # noqa: E402
from src import time_series as TS  # noqa: E402
from src import clustering as CL  # noqa: E402

st.set_page_config(page_title="Satellite Search & Change", page_icon="🛰️", layout="wide")


# ------------------------------------------------------------------ loading
@st.cache_resource(show_spinner="Loading the CLIP model...")
def get_model():
    return core.Clip()


@st.cache_data(show_spinner="Loading the index...")
def get_index():
    return core.load_index()


@st.cache_data(show_spinner=False)
def get_sites():
    return {s["key"]: s for s in core.list_sites()}


@st.cache_data(show_spinner="Reading the images...", max_entries=6)
def get_site_data(key):
    return core.load_site(get_sites()[key])


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def tile_img(m, size=160):
    d = get_site_data(m["site"])
    img = d["img1"] if m["slot"] == 1 else d["img2"]
    T = m.get("tile", core.TILE)
    t = img[m["y"]:m["y"] + T, m["x"]:m["x"] + T]
    interp = cv2.INTER_NEAREST if T < size else cv2.INTER_AREA
    return cv2.resize(rgb(t), (size, size), interpolation=interp)


def caption(m, score=None):
    c = f"{m['site_name'].split(',')[0]} | {m['date']}"
    if m.get("lonlat"):
        c += f" | {m['lonlat'][1]:.3f}N {m['lonlat'][0]:.3f}E"
    if score is not None:
        c += f" | score {score:.3f}"
    return c


def show_hits(hits, meta, cols=5):
    if not hits:
        st.info("No tiles match these filters.")
        return
    rows = [hits[i:i + cols] for i in range(0, len(hits), cols)]
    for row in rows:
        cs = st.columns(cols)
        for c, (i, s) in zip(cs, row):
            c.image(tile_img(meta[i]), caption=caption(meta[i], s), width="stretch")
    with st.expander("Result details (source scene of every tile)"):
        st.dataframe([{"rank": r + 1, "site": meta[i]["site_name"], "date": meta[i]["date"],
                       "scene": meta[i]["scene"], "tile x,y": f"{meta[i]['x']},{meta[i]['y']}",
                       "lon,lat": str(meta[i].get("lonlat") or "-"), "sensor": meta[i]["sensor"],
                       "score": round(s, 4)} for r, (i, s) in enumerate(hits)],
                     width="stretch", hide_index=True)


# -------------------------------------------------------------------- data
if not (core.INDEX_DIR / "vectors.npy").exists():
    st.error("The index is not built yet. Run:  python demo/build_index.py")
    st.stop()
V, HEAT, META, LOCS, INFO = get_index()
SITES = get_sites()
collections = sorted({m["collection"] for m in META})

# ----------------------------------------------------------------- sidebar
st.sidebar.title("🛰️ Satellite Search & Change")
st.sidebar.caption("BCSE407L project demo")
pick_coll = st.sidebar.multiselect("Image collection", collections, default=collections[:1])
site_keys = [k for k, s in SITES.items() if s["collection"] in pick_coll]
pick_sites = st.sidebar.multiselect("Sites (empty = all)", site_keys,
                                    format_func=lambda k: SITES[k]["name"])
which = st.sidebar.radio("Which date", ["Both dates", "Earlier date", "Later date"], index=0)
topk = st.sidebar.slider("Number of results", 5, 30, 10, step=5)
st.sidebar.divider()
st.sidebar.caption(f"Index: {INFO['n_vectors']:,} tiles from {INFO['n_sites']} sites, "
                   f"Sentinel-2 tiles are 640 m wide, LEVIR-CD tiles 128 m. "
                   f"Model: {INFO['model']}, used as it is, "
                   "nothing trained.")

keep = np.array([(m["collection"] in pick_coll)
                 and (not pick_sites or m["site"] in pick_sites)
                 and (which == "Both dates" or (which == "Earlier date") == (m["slot"] == 1))
                 for m in META])

st.title("Search satellite images by meaning and see what changed")
tab_text, tab_img, tab_change, tab_years, tab_cluster, tab_about = st.tabs(
    ["🔎 Text search", "🖼️ Image search", "🏗️ Change between two dates",
     "📅 Change over years", "🗺️ Discover clusters", "ℹ️ About"])

# ------------------------------------------------------------- text search
with tab_text:
    st.write("Type what you are looking for. The tiles are ranked by how well they match.")
    examples = ["an airport runway", "a river", "dense city buildings", "farm fields",
                "bare land under construction", "a lake or sea"]
    ex = st.pills("Examples", examples, selection_mode="single")
    q = st.text_input("Search text", value=ex or "an airport runway")
    if q.strip():
        vec = get_model().text("a satellite image of " + q.strip())
        hits = core.search(V, vec, keep, topk)
        st.caption(f"{int(keep.sum()):,} tiles searched")
        show_hits(hits, META)

# ------------------------------------------------------------ image search
with tab_img:
    st.write("Pick one tile. The app finds other tiles that look like it.")
    all_keys = site_keys or list(SITES)
    c1, c2, c3 = st.columns([2, 1, 1])
    skey = c1.selectbox("Site", all_keys, format_func=lambda k: SITES[k]["name"], key="img_site")
    slot = c2.radio("Date", [1, 2], horizontal=True,
                    format_func=lambda s: SITES[skey]["date1"] if s == 1 else SITES[skey]["date2"])
    rows = [i for i, m in enumerate(META) if m["site"] == skey and m["slot"] == slot]
    n = c3.number_input("Tile number", 1, len(rows), min(len(rows), 40))
    qi = rows[int(n) - 1]
    d = get_site_data(skey)
    img = rgb(d["img1"] if slot == 1 else d["img2"]).copy()
    m = META[qi]
    T = m.get("tile", core.TILE)
    cv2.rectangle(img, (m["x"], m["y"]), (m["x"] + T, m["y"] + T), (255, 40, 40),
                  3 if T == core.TILE else 6)
    a, b = st.columns([2, 1])
    a.image(img, caption="The picked tile is marked in red", width="stretch")
    b.image(tile_img(m, 256), caption="Query tile: " + caption(m))
    keep_q = keep.copy(); keep_q[qi] = False
    st.subheader("Most similar tiles")
    show_hits(core.search(V, V[qi], keep_q, topk), META)

# ------------------------------------------------------------------ change
with tab_change:
    all_keys = site_keys or list(SITES)
    ckey = st.selectbox("Site", all_keys, format_func=lambda k: SITES[k]["name"], key="chg_site")
    site = SITES[ckey]
    d = get_site_data(ckey)
    h, w = d["img1"].shape[:2]
    c1, c2 = st.columns(2)
    c1.image(rgb(d["img1"]), caption=f"Earlier: {site['date1']}  ({site['scene1']})",
             width="stretch")
    c2.image(rgb(d["img2"]), caption=f"Later: {site['date2']}  ({site['scene2']})",
             width="stretch")

    st.subheader("Where did it change?")
    pct = st.slider("Pixel change: keep the strongest % of pixels", 1.0, 10.0, 3.0, 0.5)
    mask, _ = core.pixel_change(d["b1"], d["b2"], pct)
    hm = core.site_heat(HEAT, LOCS, ckey, (h, w))
    lo, hi = np.percentile(hm[hm > 0], [5, 99]) if (hm > 0).any() else (0, 1)
    hn = np.clip((hm - lo) / max(hi - lo, 1e-6), 0, 1)
    heat_rgb = cv2.cvtColor(cv2.applyColorMap((hn * 255).astype(np.uint8), cv2.COLORMAP_INFERNO),
                            cv2.COLOR_BGR2RGB)
    over = (0.55 * rgb(d["img2"]) + 0.45 * heat_rgb).astype(np.uint8)
    pix = rgb(d["img2"]).copy()
    pix[mask > 0] = (255, 40, 40)
    c1, c2 = st.columns(2)
    c1.image(over, caption="Patch feature heat map (bright = model sees a different place)",
             width="stretch")
    c2.image(pix, caption=f"Pixel change after brightness matching, red = top {pct:.1f}% of pixels",
             width="stretch")
    if d["label"] is not None:
        st.image(d["label"], caption="Ground truth change mask from the dataset", width=420)

    st.subheader("Review list: the most changed tiles")
    T = site["tile"]
    st.caption(f"Ranked by the patch feature score. Each row shows the same "
               f"{T * site['gsd']:.0f} m tile at both dates, so a person can confirm or reject it.")
    mine = [(i, l) for i, l in enumerate(LOCS) if l["site"] == ckey]
    mine.sort(key=lambda t: -t[1]["patch_score"])
    scores = np.array([l["patch_score"] for _, l in mine])
    for rank, (i, l) in enumerate(mine[:8], 1):
        y, x = l["y"], l["x"]
        interp = cv2.INTER_NEAREST if T < 180 else cv2.INTER_AREA
        t1 = cv2.resize(rgb(d["img1"][y:y + T, x:x + T]), (180, 180), interpolation=interp)
        t2 = cv2.resize(rgb(d["img2"][y:y + T, x:x + T]), (180, 180), interpolation=interp)
        share = float((mask[y:y + T, x:x + T] > 0).mean())
        truth = (f"\n- ground truth: {100 * float((d['label'][y:y + T, x:x + T] > 0).mean()):.1f}% "
                 f"of this tile really changed" if d["label"] is not None else "")
        conf = float((scores <= l["patch_score"]).mean())
        a, b, c = st.columns([1, 1, 3])
        a.image(t1, caption=site["date1"])
        b.image(t2, caption=site["date2"])
        where = (f"{l['lonlat'][1]:.4f} N, {l['lonlat'][0]:.4f} E" if l.get("lonlat")
                 else f"tile x={x}, y={y}")
        c.markdown(f"**#{rank}**  &nbsp; {where}\n\n"
                   f"- patch feature score: **{l['patch_score']:.3f}** "
                   f"(higher than {100 * conf:.0f}% of tiles in this site)\n"
                   f"- whole tile embedding distance: {l['embedding_distance']:.3f}\n"
                   f"- pixels flagged in this tile: {100 * share:.1f}%\n"
                   f"- source: {site['scene1']} → {site['scene2']}" + truth)

# -------------------------------------------------------- change over years
@st.cache_data(show_spinner="Working through the yearly images...", max_entries=12)
def get_years(site_dir):
    years, cube, valid = TS.load_series(site_dir)
    z, _ = TS.normalise(cube, valid)
    det = TS.detect(z)
    k3 = np.ones((3, 3), np.uint8)
    two = cv2.morphologyEx(det["two_mask"].astype(np.uint8), cv2.MORPH_OPEN, k3) > 0
    last = cv2.morphologyEx(det["lasting"].astype(np.uint8), cv2.MORPH_OPEN, k3) > 0
    mb, ma = TS.before_after(cube, valid, det["first"], det["lasting"])
    types = CT.classify(CT.indices(mb[1], mb[2], mb[3], mb[4]),
                        CT.indices(ma[1], ma[2], ma[3], ma[4]))
    types[~last] = 0
    pics = [(np.transpose(np.clip(c[[2, 1, 0]] / 3000.0, 0, 1) ** 0.8, (1, 2, 0)) * 255
             ).astype(np.uint8) for c in cube]
    first_year = np.where(last, np.array(years)[det["first"]], 0)
    return years, pics, two, last, first_year, types, TS.series_info(site_dir)


with tab_years:
    ysites = [k for k, v in SITES.items()
              if v["collection"].startswith("India") and (Path(v["dir"]) / "series").is_dir()]
    if not ysites:
        st.info("No yearly series on disk. Run:  run.bat demo\\fetch_series.py")
    else:
        st.write("One cloud-free image per year (2019 to 2025), each made from three clear "
                 "Sentinel-2 scenes of the dry season. Move the slider to go through the years.")
        ykey = st.selectbox("Site", ysites, format_func=lambda k: SITES[k]["name"], key="yr_site",
                            index=ysites.index("jewar_airport_up") if "jewar_airport_up" in ysites else 0)
        years, pics, two, last, first_year, types, info = get_years(SITES[ykey]["dir"])
        yr = st.select_slider("Year", options=years, value=years[0])
        a, b = st.columns(2)
        a.image(pics[years.index(yr)], caption=f"{yr} composite", width="stretch")
        b.image(pics[-1], caption=f"{years[-1]} composite (latest)", width="stretch")
        used = info.get(str(yr), [])
        if used:
            st.caption(f"{yr} is the median of: " + ", ".join(
                f"{u['scene']} ({u['date']})" for u in used))

        st.subheader("Two dates against many dates")
        base = pics[-1]
        p1 = base.copy(); p1[two] = (255, 40, 40)
        p2 = base.copy(); p2[last] = (255, 40, 40)
        a, b = st.columns(2)
        a.image(p1, caption=f"First year against last year: {100 * two.mean():.1f}% of the area "
                            "flagged", width="stretch")
        b.image(p2, caption=f"Change that lasts over the years: {100 * last.mean():.1f}% of the "
                            "area flagged", width="stretch")
        st.caption("The right map keeps a pixel only if it changes in one year and then stays "
                   "changed. This removes most farm field and weather differences. It can also "
                   "miss real building work that keeps changing every year, like an airport "
                   "under construction, so compare both maps with the yearly images.")

        st.subheader("When and what")
        palette = {2019: (68, 1, 84), 2020: (68, 1, 84), 2021: (59, 82, 139), 2022: (33, 145, 140),
                   2023: (94, 201, 98), 2024: (253, 231, 37), 2025: (253, 231, 37)}
        when = base.copy()
        shown = sorted({int(y) for y in np.unique(first_year) if y > 0})
        for y in shown:
            when[first_year == y] = palette.get(y, (255, 255, 255))
        tcol = base.copy()
        tc = CT.colour(types)
        tcol[types > 0] = tc[types > 0]
        a, b = st.columns(2)
        a.image(when, caption="First year the lasting change is seen: "
                + ", ".join(f"{y}" for y in shown) + " (dark purple = early, yellow = late)",
                width="stretch")
        b.image(tcol, caption="Change type from spectral indices: red built-up or bare, orange "
                              "vegetation lost, green vegetation gained, blue water gained, "
                              "cyan water lost", width="stretch")
        area = CT.area_table(types)
        st.dataframe([{"change type": k, "area (hectares)": v} for k, v in area.items()],
                     hide_index=True, width="content")
        st.caption("The change types are a rough guide only. In our OSCD test the index rules "
                   "did not separate built-up change well, so do not read these labels as exact.")


# --------------------------------------------------------- discover clusters
@st.cache_data(show_spinner="Running K-means per collection...", max_entries=4)
def get_clusters(collection, n_clusters):
    """Cluster tiles from one collection to avoid mixing resolutions."""
    rows = [i for i, m in enumerate(META) if m["collection"] == collection]
    if len(rows) < n_clusters:
        return {}
    vecs = V[rows]                                   # already L2-normalised

    class _TinyIndex:                                 # minimal shim for CL
        def __init__(self, v):
            self.vectors = v
            self.n = len(v)
            self.dim = v.shape[1]
            self.metadata = [{} for _ in range(len(v))]
        def search(self, q, k=10, subset=None):
            pool = subset if subset is not None else list(range(self.n))
            scores = self.vectors[pool] @ q
            top = sorted(pool, key=lambda i: -float(scores[i - (0 if subset else 0)]))
            return [(i, float(self.vectors[i] @ q)) for i in top[:k]]

    idx = _TinyIndex(vecs)
    labels, _ = CL.cluster_sites(idx, n_clusters=n_clusters)
    summary = CL.cluster_summary(idx, labels)
    # map local indices back to global META indices
    return {"rows": rows, "labels": labels, "summary": summary}


with tab_cluster:
    st.write(
        "Tiles are grouped by visual similarity using K-means on the same "
        "embedding vectors used for search. "
        "**Clustered per collection** so that resolution differences "
        "(Sentinel-2 at 10 m vs LEVIR-CD at 0.5 m) do not dominate the grouping. "
        "No ground-truth metric exists for this — it is a qualitative demo only."
    )
    c1, c2 = st.columns([2, 1])
    coll_choice = c1.selectbox(
        "Collection to cluster",
        [c for c in collections if c in {m["collection"] for m in META}],
        key="cl_coll"
    )
    n_cl = c2.slider("Number of clusters", 4, 12, 8, key="cl_k")

    cl_data = get_clusters(coll_choice, n_cl)
    if not cl_data:
        st.warning("Not enough tiles in this collection to form that many clusters.")
    else:
        rows_g = cl_data["rows"]          # global META indices for this collection
        labels = cl_data["labels"]
        summary = cl_data["summary"]

        for entry in summary:
            c_id = entry["cluster"]
            size = entry["size"]
            ex_local = entry["exemplar_row"]   # index within the collection subset
            ex_global = rows_g[ex_local]
            m = META[ex_global]

            members_global = [rows_g[i] for i, l in enumerate(labels) if l == c_id]
            sample_global = members_global[:8]    # show up to 8 example tiles

            with st.expander(
                f"Cluster {c_id + 1}  —  {size} tiles  "
                f"(exemplar: {m['site_name'].split(',')[0]}, {m['date']})",
                expanded=(c_id < 2),
            ):
                st.caption(
                    "The exemplar tile (most central in the cluster) is shown first "
                    "in red, followed by up to 7 other members. Land-type labels are "
                    "not available — a person must judge whether the cluster is "
                    "meaningful."
                )
                thumb_indices = [ex_global] + [g for g in sample_global if g != ex_global][:7]
                cols = st.columns(len(thumb_indices))
                for col, gi in zip(cols, thumb_indices):
                    mi = META[gi]
                    img = tile_img(mi, size=120)
                    border = "exemplar ★" if gi == ex_global else mi["date"]
                    col.image(img, caption=f"{mi['site_name'].split(',')[0]}\n{border}",
                              width="stretch")

        st.caption(
            f"Clustered {len(rows_g):,} tiles from **{coll_choice}** into "
            f"{n_cl} groups. K-means is run on the L2-normalised CLIP embeddings "
            "(cosine distance). No metric is computed — no labelled cluster ground "
            "truth exists for this data."
        )


# ------------------------------------------------------------------- about
with tab_about:
    st.markdown(f"""
### What this demo does
- Every image is cut into tiles: 64 px for Sentinel-2 (640 m on the ground) and 256 px for
  LEVIR-CD (128 m on the ground).
- One pre-trained model, **{INFO['model']}**, turns each tile into numbers. We did not train it.
- The same numbers are stored once and used for **search** and for **change**.

### What works and what does not (from our tests)
| Test | Result |
|---|---|
| Image search, same place at another date (OSCD) | correct tile first 46% of the time (random 0.05%) |
| Text search on EuroSAT (27,000 labelled patches) | 69% of the first 10 results correct |
| Finding changed tiles on LEVIR-CD (0.5 m images) | AUC 0.92 |
| Finding changed tiles on OSCD (10 m images) | AUC 0.70, weak |
| Exact shape of the change | weak on both datasets |

The images in this demo are 10 m Sentinel-2, which is the hard case for change. So the
heat map is rough, and the pixel change map also reacts to crops, water level and haze.
A person should check the review list.

### Data
- **India sites**: Copernicus Sentinel-2 L2A, early 2019 and early 2025, read from the public
  AWS open data mirror. Bands used: blue, green, red, near infrared.
- **OSCD**: Onera Satellite Change Detection dataset (24 city pairs with change masks).
- **LEVIR-CD**: 14 test pairs at 0.5 m per pixel with building change masks. These are the
  sharp images where the change idea works best.
""")
    st.json(INFO)
