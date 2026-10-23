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
import importlib  # noqa: E402
import core  # noqa: E402
core = importlib.reload(core)   # a hot update on Streamlit Cloud otherwise keeps the old core.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import change_types as CT  # noqa: E402
from src import time_series as TS  # noqa: E402
from src import clustering as CL  # noqa: E402

st.set_page_config(page_title="Satellite Search & Change", layout="wide")


# ------------------------------------------------------------------ loading
@st.cache_resource(show_spinner="Loading the CLIP model...")
def get_model():
    return core.Clip()


@st.cache_data(show_spinner="Loading the index...")
def get_index():
    return core.load_index()


# Cached results depend on core.py, so its modified time is part of the cache
# key: after an update the old cached site list is not reused.
CORE_V = Path(core.__file__).stat().st_mtime


@st.cache_data(show_spinner=False)
def _sites(v):
    return {s["key"]: s for s in core.list_sites()}


@st.cache_data(show_spinner="Reading the images...", max_entries=6)
def _site_data(key, v):
    return core.load_site(get_sites()[key])


def get_sites():
    return _sites(CORE_V)


def get_site_data(key):
    return _site_data(key, CORE_V)


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def fit(img, longest=1400):
    """Shrink only for display, so big scenes (ISRO 5 m) do not slow the browser."""
    h, w = img.shape[:2]
    s = longest / max(h, w)
    return img if s >= 1 else cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


@st.cache_data(show_spinner=False, max_entries=5000)
def _tile(site, slot, y, x, T, size, v):
    d = get_site_data(site)
    img = d["img1"] if slot == 1 else d["img2"]
    interp = cv2.INTER_NEAREST if T < size else cv2.INTER_AREA
    return cv2.resize(rgb(img[y:y + T, x:x + T]), (size, size), interpolation=interp)


def tile_img(m, size=160):
    return _tile(m["site"], m["slot"], m["y"], m["x"], m.get("tile", core.TILE), size, CORE_V)


def site_label(k):
    s = SITES[k]
    return f"{s['name']} · {s['country']} · {core.res_label(s['gsd'])}"


def caption(m, score=None):
    c = f"{m['site_name'].split(',')[0]} | {core.res_label(m['gsd_m'])} | {m['date']}"
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
        st.dataframe([{"rank": r + 1, "site": meta[i]["site_name"],
                       "country": meta[i].get("country", "-"),
                       "resolution": core.res_label(meta[i]["gsd_m"]), "date": meta[i]["date"],
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
st.sidebar.title("Satellite Search & Change")
st.sidebar.caption("BCSE407L project demo")
res_opts = sorted({s["gsd"] for s in SITES.values()}, reverse=True)
pick_res = st.sidebar.multiselect("Resolution (ground size of one pixel)", res_opts,
                                  default=res_opts, format_func=core.res_label)
coll_opts = sorted({s["collection"] for s in SITES.values() if s["gsd"] in pick_res})
pick_coll = st.sidebar.multiselect("Image collection", coll_opts, default=coll_opts)
country_opts = sorted({s["country"] for s in SITES.values() if s["collection"] in pick_coll})
pick_country = st.sidebar.multiselect("Country (empty = all)", country_opts)
site_keys = [k for k, s in SITES.items() if s["collection"] in pick_coll
             and (not pick_country or s["country"] in pick_country)]
pick_sites = st.sidebar.multiselect("City / location (empty = all)", site_keys,
                                    format_func=site_label)
which = st.sidebar.radio("Which date", ["Both dates", "Earlier date", "Later date"], index=0)
topk = st.sidebar.slider("Number of results", 5, 30, 10, step=5)
st.sidebar.divider()
st.sidebar.caption(f"Index: {INFO['n_vectors']:,} tiles from {INFO['n_sites']} sites, "
                   f"Sentinel-2 tiles are 640 m wide, LEVIR-CD and SYSU-CD tiles 128 m. "
                   f"Model: {INFO['model']}, used as it is, "
                   "nothing trained.")

keep = np.array([m["site"] in site_keys                 # only sites whose images are on disk
                 and (not pick_sites or m["site"] in pick_sites)
                 and (which == "Both dates" or (which == "Earlier date") == (m["slot"] == 1))
                 for m in META])

st.title("Search satellite images by meaning and see what changed")
# Only the chosen section runs (st.tabs would compute every tab on every click,
# which is slow with the large ISRO scene).
SECTIONS = ["Map of all data", "Text search", "Check the search (EuroSAT)", "Image search",
            "Change between two dates", "Search by GPS", "Change over years", "Discover clusters", "About"]
PAGE = st.radio("Section", SECTIONS, horizontal=True, label_visibility="collapsed", key="section")
st.divider()

# --------------------------------------------------------------------- map
RES_COLOURS = {10.0: [31, 119, 180], 5.0: [44, 160, 44], 0.5: [230, 110, 20]}


def all_places():
    """One row per place in the index (India, OSCD) plus one approximate row
    per collection without coordinates (LEVIR-CD, SYSU-CD)."""
    cov = core.coverage(LOCS)
    sites, regions = {}, {}
    for m in META:
        s = sites.setdefault(m["site"], {"name": m["site_name"], "collection": m["collection"],
                                         "gsd": m["gsd_m"], "dates": {}, "tiles": 0,
                                         "country": m.get("country")})
        s["dates"][m["slot"]] = m["date"]
        s["tiles"] += m["slot"] == 1
    rows = []
    for key, s in sites.items():
        country = s["country"] or SITES.get(key, {}).get("country", "?")
        row = {"dataset": s["collection"].split(" (")[0], "resolution": core.res_label(s["gsd"]),
               "gsd": s["gsd"], "tiles": s["tiles"], "here": "yes" if key in SITES else "no",
               "dates": f"{s['dates'].get(1, '?')} to {s['dates'].get(2, '?')}"}
        if key in cov:
            c = cov[key]
            rows.append(dict(row, name=s["name"], country=country, lat=c["lat"], lon=c["lon"],
                             gps="yes", box=[[c["lon_min"], c["lat_min"]], [c["lon_max"], c["lat_min"]],
                                             [c["lon_max"], c["lat_max"]], [c["lon_min"], c["lat_max"]]]))
        elif s["collection"] in core.REGION_CENTRES:
            r = regions.setdefault(s["collection"], dict(row, pairs=0, tiles=0, here="no"))
            r["pairs"] += 1
            r["tiles"] += s["tiles"]
            r["here"] = "yes" if (key in SITES or r["here"] == "yes") else "no"
    for coll, r in regions.items():
        place, country, lat, lon = core.REGION_CENTRES[coll]
        rows.append(dict(r, name=f"{place} ({r.pop('pairs')} image pairs, exact spots not published)",
                         country=country, lat=lat, lon=lon, gps="no, region only", box=None))
    if INFO.get("n_eurosat_patches"):
        rows.append({"name": f"Europe, 34 countries ({INFO['n_eurosat_patches']:,} labelled patches, "
                             "exact spots not published)", "country": "Europe", "dataset": "EuroSAT",
                     "resolution": "10 m", "gsd": 10.0, "tiles": INFO["n_eurosat_patches"],
                     "here": "yes" if core.eurosat_files()[0] else "no", "dates": "2015 to 2017",
                     "lat": 50.0, "lon": 10.0, "gps": "no, region only", "box": None})
    for r in rows:
        r["color"] = RES_COLOURS.get(r["gsd"], [120, 120, 120])
    return rows


if PAGE == "Map of all data":
    import pydeck as pdk
    PLACES = all_places()
    st.write("Every place in this project. Hover over a marker for its details; zoom in to see "
             "the exact area covered by each GPS site.")
    c1, c2 = st.columns(2)
    m_res = c1.multiselect("Resolution", sorted({r["gsd"] for r in PLACES}, reverse=True),
                           default=sorted({r["gsd"] for r in PLACES}, reverse=True),
                           format_func=core.res_label, key="map_res")
    m_ds = c2.multiselect("Dataset", sorted({r["dataset"] for r in PLACES}),
                          default=sorted({r["dataset"] for r in PLACES}), key="map_ds")
    shown = [r for r in PLACES if r["gsd"] in m_res and r["dataset"] in m_ds]
    a, b, c, d = st.columns(4)
    a.metric("Places", len(shown))
    b.metric("Countries", len({r["country"] for r in shown}))
    c.metric("Sharpest pixel", core.res_label(min(r["gsd"] for r in shown)) if shown else "-")
    d.metric("Image tiles and patches", f"{sum(r['tiles'] for r in shown):,}")
    tip = {"html": "<b>{name}</b><br/>{country} &middot; {dataset}<br/>Resolution: {resolution} "
                   "per pixel<br/>Dates: {dates}<br/>Tiles: {tiles}<br/>GPS search: {gps}<br/>"
                   "Images available here: {here}",
           "style": {"fontSize": "13px"}}
    boxes = [dict(r, fill=r["color"] + [50]) for r in shown if r["box"]]
    layers = [
        pdk.Layer("PolygonLayer", boxes, id="boxes", get_polygon="box", get_fill_color="fill",
                  get_line_color="color", line_width_min_pixels=1, pickable=True),
        pdk.Layer("ScatterplotLayer", shown, id="places", get_position="[lon, lat]",
                  get_fill_color="color", get_line_color=[255, 255, 255], stroked=True,
                  line_width_min_pixels=1, get_radius=4000, radius_min_pixels=6,
                  radius_max_pixels=14, pickable=True),
    ]
    event = st.pydeck_chart(pdk.Deck(layers=layers, tooltip=tip, map_style=None,
                                     initial_view_state=pdk.ViewState(latitude=20, longitude=15, zoom=0.9)),
                            height=520, on_select="rerun", selection_mode="single-object", key="map")
    picked = [o for objs in (event.selection.objects or {}).values() for o in objs] if event else []
    if picked:
        r = picked[0]
        st.info(f"**{r['name']}** · {r['country']} · {r['dataset']} · {r['resolution']} per pixel · "
                f"dates {r['dates']} · {r['tiles']:,} tiles · GPS search: {r['gps']} · "
                f"images available here: {r['here']}")
    else:
        st.caption("Click a marker (or hover over it) to see its details.")
    st.caption("Colour = resolution: " + ", ".join(
        f"{['blue', 'green', 'orange'][i]} = {core.res_label(g)}"
        for i, g in enumerate(RES_COLOURS) if any(r['gsd'] == g for r in PLACES))
        + ". LEVIR-CD and SYSU-CD do not publish exact locations, so they have one marker at "
          "their region. The map background needs internet.")
    st.dataframe([{"place": r["name"], "country": r["country"], "dataset": r["dataset"],
                   "resolution": r["resolution"], "dates": r["dates"], "tiles": r["tiles"],
                   "GPS search": r["gps"], "images available here": r["here"]}
                  for r in sorted(shown, key=lambda r: (-r["gsd"], r["dataset"], r["name"]))],
                 hide_index=True, width="stretch")

# ------------------------------------------------------------- text search
if PAGE == "Text search":
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

# ------------------------------------------- check the search on EuroSAT
@st.cache_data(show_spinner="Loading the EuroSAT check set...")
def get_eurosat(v):
    return core.load_eurosat()


def framed(img, ok, size=110):
    """Patch with a green (correct) or red (wrong) frame; RGB in, RGB out."""
    t = cv2.resize(rgb(img), (size, size), interpolation=cv2.INTER_NEAREST)
    return cv2.copyMakeBorder(t, 6, 6, 6, 6, cv2.BORDER_CONSTANT,
                              value=(40, 170, 60) if ok else (220, 40, 40))


if PAGE == "Check the search (EuroSAT)":
    ES = get_eurosat(CORE_V)
    if ES is None:
        st.info("No EuroSAT patches or EuroSAT index found. Run:  python demo/build_index.py")
    else:
        EV, EF, EBASE = ES
        classes = sorted({c for _, c in EF})
        st.write("Here every patch has a label made by people, so each search result can be "
                 "marked right or wrong. Pick a land-cover class: the app searches with the text "
                 "on the right and frames each result green (correct class) or red (wrong).")
        c1, c2, c3 = st.columns([1, 2, 1])
        cls = c1.selectbox("Class to find", classes, index=classes.index("Forest") if "Forest" in classes else 0)
        q = c2.text_input("Search text (you can change it)", value=core.EUROSAT_PROMPTS.get(cls, cls),
                          key=f"es_q_{cls}")
        k = c3.slider("Results shown", 10, 50, 20, 10, key="es_k")
        scores = EV @ get_model().text("a satellite image of " + q.strip())
        order = np.argsort(-scores)
        truth = np.array([c == cls for _, c in EF])
        hit = truth[order]
        ap = float((np.cumsum(hit) / np.arange(1, len(hit) + 1))[hit].mean()) if hit.any() else 0.0
        a, b, c, d = st.columns(4)
        a.metric(f"Correct in the top {k}", f"{int(hit[:k].sum())} / {k}")
        b.metric("Correct in the top 10", f"{int(hit[:10].sum())} / 10")
        c.metric("Average precision (all patches)", f"{ap:.2f}")
        d.metric("Random guess would get", f"{truth.mean():.2f}")
        cols = st.columns(10)
        for r, i in enumerate(order[:k]):
            p, true_cls = EF[i]
            ok = true_cls == cls
            cols[r % 10].image(framed(cv2.imread(str(EBASE / p)), ok),
                               caption=f"{r + 1}. " + ("correct" if ok else f"wrong: {true_cls}"))
        st.caption(f"Searched {len(EF):,} labelled EuroSAT patches (10 m Sentinel-2, 64 x 64 px, "
                   f"{len(classes)} classes). Average precision rewards putting all {int(truth.sum()):,} "
                   f"{cls} patches near the top. Across all 10 classes our measured mean is 0.44 "
                   "(random 0.10).")


# ------------------------------------------------------------ image search
if PAGE == "Image search":
    st.write("Pick one tile. The app finds other tiles that look like it.")
    all_keys = site_keys or list(SITES)
    c1, c2, c3 = st.columns([2, 1, 1])
    skey = c1.selectbox("Site", all_keys, format_func=site_label, key="img_site")
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
    a.image(fit(img), caption="The picked tile is marked in red", width="stretch")
    b.image(tile_img(m, 256), caption="Query tile: " + caption(m))
    keep_q = keep.copy(); keep_q[qi] = False
    st.subheader("Most similar tiles")
    show_hits(core.search(V, V[qi], keep_q, topk), META)

# ------------------------------------------------------------------ change
if PAGE == "Change between two dates":
    all_keys = site_keys or list(SITES)
    ckey = st.selectbox("Site", all_keys, format_func=site_label, key="chg_site")
    site = SITES[ckey]
    d = get_site_data(ckey)
    h, w = d["img1"].shape[:2]
    st.caption(f"Location: {site['place']}, {site['country']} · {core.res_label(site['gsd'])} per pixel · "
               f"{site['sensor']} · image {w * site['gsd'] / 1000:.2f} × {h * site['gsd'] / 1000:.2f} km")
    c1, c2 = st.columns(2)
    c1.image(fit(rgb(d["img1"])), caption=f"Earlier: {site['date1']}  ({site['scene1']})",
             width="stretch")
    c2.image(fit(rgb(d["img2"])), caption=f"Later: {site['date2']}  ({site['scene2']})",
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
    c1.image(fit(over), caption="Patch feature heat map (bright = model sees a different place)",
             width="stretch")
    c2.image(fit(pix), caption=f"Pixel change after brightness matching, red = top {pct:.1f}% of pixels",
             width="stretch")
    if d["label"] is not None:
        st.image(fit(d["label"]), caption="Ground truth change mask from the dataset", width=420)

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

# -------------------------------------------------------------- GPS search
if PAGE == "Search by GPS":
    COV = {k: v for k, v in core.coverage(LOCS).items() if k in SITES}
    if not COV:
        st.info("No site in this index has GPS coordinates. Rebuild it:  python demo/build_index.py")
    else:
        st.write("Type a latitude and longitude. It works only inside the areas we have images "
                 "for (table and map below). LEVIR-CD and SYSU-CD publish no coordinates, so "
                 "they cannot be searched by GPS.")
        c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
        lat = c1.number_input("Latitude (°N)", -90.0, 90.0, 12.9692, 0.001, format="%.4f")
        lon = c2.number_input("Longitude (°E)", -180.0, 180.0, 79.1559, 0.001, format="%.4f")
        rad = c3.slider("Radius (km)", 0.5, 5.0, 1.0, 0.5)
        jump = c4.selectbox("...or go to the centre of a site", [None] + sorted(COV, key=site_label),
                            format_func=lambda k: "(use the numbers I typed)" if k is None
                            else site_label(k))
        if jump:
            lat, lon = COV[jump]["lat"], COV[jump]["lon"]
        locs_ok = [l if l["site"] in SITES else {} for l in LOCS]
        near, nearest = core.gps_lookup(locs_ok, lat, lon, rad)
        # every image set that has a tile over the point (centre to tile corner), sharpest first
        cover = []
        for i, km in near:
            k = LOCS[i]["site"]
            if km <= SITES[k]["tile"] * SITES[k]["gsd"] / 1000 * 0.71 and k not in cover:
                cover.append(k)
        cover.sort(key=lambda k: SITES[k]["gsd"])
        if not cover:
            ni, nkm = nearest
            st.error(f"No images at {lat:.4f} °N, {lon:.4f} °E. The nearest images are "
                     f"**{site_label(LOCS[ni]['site'])}**, {nkm:.1f} km away. Pick a point inside "
                     "one of the areas listed below.")
        else:
            ns_key = cover[0]
            if len(cover) > 1:
                ns_key = st.radio(f"{len(cover)} image sets cover this point", cover, horizontal=True,
                                  format_func=lambda k: f"{site_label(k)} ({SITES[k]['date1']} → "
                                                        f"{SITES[k]['date2']})")
            ns = SITES[ns_key]
            st.success(f"{lat:.4f} °N, {lon:.4f} °E is inside **{ns['name']}**, "
                       f"{ns['country']} · {core.res_label(ns['gsd'])} per pixel · {ns['sensor']} · "
                       f"{ns['date1']} → {ns['date2']}")
            d = get_site_data(ns_key)
            T = ns["tile"]
            mine = [(i, km) for i, km in near if LOCS[i]["site"] == ns_key]
            pic = rgb(d["img2"]).copy()
            for i, _ in mine:
                l = LOCS[i]
                cv2.rectangle(pic, (l["x"], l["y"]), (l["x"] + T, l["y"] + T), (255, 210, 0), 1)
            if d["geo"]:
                px = core.lonlat_to_px(d["geo"], lon, lat)
                cv2.circle(pic, px, max(2, int(rad * 1000 / ns["gsd"])), (255, 40, 40), 2)
                cv2.drawMarker(pic, px, (255, 40, 40), cv2.MARKER_CROSS, 18, 2)
            st.image(fit(pic), caption=f"Later image ({ns['date2']}): red = your point and {rad:g} km "
                                  f"circle, yellow = the {len(mine)} tiles inside it",
                     width="stretch")
            st.subheader("Tiles at this point, both dates (most changed first)")
            mine.sort(key=lambda t: -LOCS[t[0]]["patch_score"])
            for i, km in mine[:8]:
                l = LOCS[i]
                y, x = l["y"], l["x"]
                t1 = cv2.resize(rgb(d["img1"][y:y + T, x:x + T]), (160, 160), interpolation=cv2.INTER_NEAREST)
                t2 = cv2.resize(rgb(d["img2"][y:y + T, x:x + T]), (160, 160), interpolation=cv2.INTER_NEAREST)
                a, b, c = st.columns([1, 1, 3])
                a.image(t1, caption=ns["date1"])
                b.image(t2, caption=ns["date2"])
                c.markdown(f"**{l['lonlat'][1]:.4f} °N, {l['lonlat'][0]:.4f} °E** "
                           f"({km:.2f} km from your point)\n\n"
                           f"- patch feature change score: **{l['patch_score']:.3f}**\n"
                           f"- whole tile embedding distance: {l['embedding_distance']:.3f}\n"
                           f"- tile size: {T * ns['gsd']:.0f} m × {T * ns['gsd']:.0f} m")

        st.subheader("Where we have images (GPS searchable)")
        import pandas as pd
        rows = [{"site": SITES[k]["name"], "country": SITES[k]["country"],
                 "resolution": core.res_label(SITES[k]["gsd"]),
                 "dates": f"{SITES[k]['date1']} → {SITES[k]['date2']}",
                 "latitude": f"{v['lat_min']:.3f} to {v['lat_max']:.3f}",
                 "longitude": f"{v['lon_min']:.3f} to {v['lon_max']:.3f}"}
                for k, v in sorted(COV.items(), key=lambda kv: site_label(kv[0]))]
        st.dataframe(rows, hide_index=True, width="stretch")
        pts = pd.DataFrame([{"lat": v["lat"], "lon": v["lon"], "color": "#1f77b4", "size": 4000}
                            for v in COV.values()]
                           + [{"lat": lat, "lon": lon, "color": "#e8282899", "size": 9000}])
        st.map(pts, latitude="lat", longitude="lon", color="color", size="size")
        st.caption("Blue = site centres, red = your point. The map background needs internet; "
                   "the search itself works offline.")


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


if PAGE == "Change over years":
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


if PAGE == "Discover clusters":
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
                    border = "exemplar" if gi == ex_global else mi["date"]
                    col.image(img, caption=f"{mi['site_name'].split(',')[0]}\n{border}",
                              width="stretch")

        st.caption(
            f"Clustered {len(rows_g):,} tiles from **{coll_choice}** into "
            f"{n_cl} groups. K-means is run on the L2-normalised CLIP embeddings "
            "(cosine distance). No metric is computed — no labelled cluster ground "
            "truth exists for this data."
        )


# ------------------------------------------------------------------- about
if PAGE == "About":
    st.markdown(f"""
### What this demo does
- Every image is cut into tiles: 64 px for Sentinel-2 (640 m on the ground) and 256 px for
  LEVIR-CD and SYSU-CD (128 m on the ground).
- One pre-trained model, **{INFO['model']}**, turns each tile into numbers. We did not train it.
- The same numbers are stored once and used for **search** and for **change**.

### What works and what does not (from our tests)
| Test | Result |
|---|---|
| Image search, same place at another date (OSCD) | correct tile first 46% of the time (random 0.05%) |
| Text search on EuroSAT (27,000 labelled patches) | 69% of the first 10 results correct |
| Finding changed tiles on LEVIR-CD (0.5 m, new buildings) | AUC 0.92; typing "residential area with roofs" 0.95 |
| Finding changed tiles on SYSU-CD (0.5 m, mixed change) | AUC 0.76 (pixel difference 0.70); text query only 0.59 |
| Finding changed tiles on OSCD (10 m images) | AUC 0.70, weak |
| Exact shape of the change | weak on OSCD (F1 0.35) and LEVIR-CD (0.37); best method on SYSU-CD (F1 0.58) |

The images in this demo are 10 m Sentinel-2, which is the hard case for change. So the
heat map is rough, and the pixel change map also reacts to crops, water level and haze.
A person should check the review list.

### Data
| Collection | Resolution | Places | Ground truth |
|---|---|---|---|
| India sites: Copernicus Sentinel-2 L2A, early 2019 and early 2025, plus one image per year | 10 m | 10 sites in India | none (shown, not scored) |
| ISRO LISS-IV (ResourceSat-2/2A, Bhoonidhi open data), 21 Mar 2023 and 4 Jan 2026, false colour | 5 m | VIT, Katpadi and Vellore town | none (shown, not scored) |
| OSCD: Onera Satellite Change Detection | 10 m | 24 cities in 13 countries | change masks |
| EuroSAT: 27,000 labelled patches | 10 m | Europe | land-cover classes |
| LEVIR-CD: building change pairs | 0.5 m | Austin area, Texas | change masks |
| SYSU-CD: mixed change pairs, 2007 and 2014 | 0.5 m | Hong Kong | change masks |

The website version carries a smaller copy (6 OSCD cities, 200 EuroSAT patches per class);
the local version uses every image on disk.
""")
    st.json(INFO)
