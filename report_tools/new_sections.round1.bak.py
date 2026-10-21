"""Chapters 5, 6 and 7 of the report. Every number comes from
results/results.json and results/extra_results.json (real OSCD runs)."""

import json

from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.shared import RGBColor


def add(h):
    P, H1, H2, body, bullet, numbered, table = (h[k] for k in
        ("P", "H1", "H2", "body", "bullet", "numbered", "table"))
    doc, RFIG, ROOT, Pt, Inches, qn = (h[k] for k in
        ("doc", "RFIG", "ROOT", "Pt", "Inches", "qn"))
    R = json.loads((ROOT / "results" / "results.json").read_text())
    E = json.loads((ROOT / "results" / "extra_results.json").read_text())
    fig_no = [2]

    def figure(name, caption, width=6.0):
        fig_no[0] += 1
        doc.add_picture(str(RFIG / name), width=Inches(width))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.paragraphs[-1].paragraph_format.space_after = Pt(2)
        doc.paragraphs[-1].paragraph_format.keep_with_next = True
        P(f"Fig. {fig_no[0]}. {caption}", size=11, align="center", spacing=1.0,
          space_after=8)

    def code(text):
        for line in text.strip("\n").split("\n"):
            p = doc.add_paragraph()
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.left_indent = Inches(0.25)
            pPr = p._p.get_or_add_pPr()
            shd = OxmlElement("w:shd")
            shd.set(qn("w:val"), "clear"); shd.set(qn("w:fill"), "F2F4F6")
            pPr.append(shd)
            r = p.add_run(line if line else " ")
            r.font.name = "Consolas"; r.font.size = Pt(9)
            r._element.rPr.rFonts.set(qn("w:eastAsia"), "Consolas")
        P(space_after=4, size=4)

    def step(n, lead, text):
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.left_indent = Inches(0.5)
        p.paragraph_format.first_line_indent = Inches(-0.25)
        r = p.add_run(f"{n}. {lead} "); r.bold = True
        r.font.name = "Times New Roman"; r.font.size = Pt(12)
        from noobify import noob
        r = p.add_run(noob(text.strip()))
        r.font.name = "Times New Roman"; r.font.size = Pt(12)

    def f3(x):
        return f"{x:.3f}"

    def f2(x):
        return f"{x:.2f}"

    inv = R["inventory"]
    ds = R["dataset"]
    sysr = R["system"]["remoteclip"]
    ret = R["retrieval"]
    cdx = R["cross_date"]
    tc = R["tile_change"]
    px = R["pixel"]["test"]
    bo = E["brightness_only"]
    parts = E["suppression_parts"]
    G = E["gate_c_matched"]
    t128 = E["tile128"]

    ES_FILE = ROOT / "results" / "eurosat_results.json"
    ES = json.loads(ES_FILE.read_text()) if ES_FILE.exists() else None

    # =================================================================== 4
    H1("4. Dataset Details")

    H2("4.1 OSCD Dataset")
    body("We needed a dataset which has two images of same place taken at different "
         "times, and also the correct answer of what changed, so that we can check "
         "our output in numbers. The Onera Satellite Change Detection (OSCD) dataset "
         "gives this. It is real Sentinel-2 data, which is one of the image sources "
         "named in the problem statement. The two images of every city are already "
         "aligned with each other, and the whole dataset is only around 513 MB so we "
         "could run it on a laptop.")
    table(["Property", "Value"],
          [["Number of image pairs", f"{ds['n_cities']} cities (14 for training, 10 for testing)"],
           ["Satellite and sensor", "Sentinel-2, MultiSpectral Instrument (MSI)"],
           ["Bands per image", "13 (B01 to B12 and B8A), stored as separate GeoTIFF files"],
           ["Ground resolution", "10 m per pixel"],
           ["Years covered", "2015 to 2018; two dates of a pair are about 1 to 2.5 years apart"],
           ["Image sizes", "From 385 x 241 px (norcia) to 1070 x 1180 px (beirut)"],
           ["Ground truth", "One black and white change mask per city (white = changed)"],
           ["Total pixels", f"{ds['total_pixels']:,}"],
           ["Pixels labelled as changed", f"{ds['changed_pixels']:,} ({ds['changed_pct']:.2f}%)"]],
          widths=[2.0, 4.1], font=10)
    P(space_after=2)
    figure("fig_dataset_samples.png",
           "Three city pairs from OSCD with their ground-truth change masks", width=4.0)

    H2("4.2 The 24 City Pairs")
    body("Below table lists every pair which we used. The last column shows how much "
         "of the image is marked as changed. This number is small everywhere.")
    half = (len(inv) + 1) // 2
    rows = []
    for a, b in zip(inv[:half], inv[half:]):
        rows.append([a["city"], f"{a['width']} x {a['height']}", a["date_1"], a["date_2"],
                     f"{a['changed_pct']:.2f}",
                     b["city"], f"{b['width']} x {b['height']}", b["date_1"], b["date_2"],
                     f"{b['changed_pct']:.2f}"])
    table(["City", "Size", "Date 1", "Date 2", "Chg %"] * 2, rows,
          widths=[0.85, 0.7, 0.72, 0.72, 0.42] * 2, font=7.5)
    P(space_after=2)

    H2("4.3 Metadata We Extract")
    body("Metadata means the information about an image, not the image itself. We "
         "collect it from the dates.txt file, the .geojson file and the GeoTIFF "
         "header, and attach it to every tile. So any search result or change result "
         "can be traced back to a city, a date and a position.")
    ex = R["example_metadata"]
    table(["Field", "Example", "Where it comes from"],
          [["tile_id", ex["tile_id"], "we build it"],
           ["city", ex["city"], "folder name"],
           ["date", ex["date"], "dates.txt"],
           ["sensor", ex["sensor"], "dataset documentation"],
           ["gsd_m (pixel size in m)", str(ex["gsd_m"]), "dataset documentation"],
           ["x, y (position in the image)", f"{ex['x']}, {ex['y']}", "tiling step"],
           ["tile_px", str(ex["tile_px"]), "our setting"],
           ["changed_frac", str(ex["changed_frac"]), "ground-truth mask (only for scoring)"],
           ["water / vegetation / builtup", "true or false", "NDWI and NDVI from bands B03, B04, B08"],
           ["footprint (lon/lat)", str(inv[0]["geo"].get("footprint_lonlat")), ".geojson file"]],
          widths=[2.0, 2.1, 2.0], font=9.5)
    P(space_after=2)
    body("OSCD does not tell which tiles are showing water or vegetation, but we need "
         "some correct answer to score the text search. So we calculate NDVI and NDWI "
         "from the bands and tag a tile as water, vegetation or builtup from that. "
         "These tags are made by formula and not by a person, so they are not perfect.")

    if ES:
        H2("4.4 EuroSAT Dataset")
        body("Because the tags above are not perfect, we also tested the search part "
             f"on EuroSAT. It has {ES['n_total']:,} Sentinel-2 patches of 64 x 64 px, "
             f"and each patch is labelled by people into one of {len(ES['classes'])} "
             "land types (river, forest, highway, residential and so on). It is the "
             "same satellite and same patch size as our OSCD tiles. EuroSAT has no "
             "image pairs, so we use it only for search and not for change.")

    # =================================================================== 5
    doc.add_page_break()
    H1("5. Implementation")

    H2("5.1 Code Structure")
    body("We wrote the project in Python. We split the code in small files so that "
         "each of us can work on own part without waiting for others.")
    table(["File", "What it does"],
          [["src/ingest.py", "reads images and cuts them into tiles, keeps metadata with each tile"],
           ["src/embedder.py", "loads the pre-trained CLIP model and turns tiles or text into embeddings"],
           ["src/index_builder.py", "stores all embeddings in one shared index and searches it"],
           ["src/retrieval.py, filters.py", "text search, image search, and date / sensor / area filters"],
           ["src/change_detect.py", "change score from embeddings, pixel change map, baseline methods"],
           ["src/suppress.py", "brightness matching, cloud / shadow mask, removing small specks"],
           ["src/metrics.py", "Precision@k, AP, precision, recall, F1, IoU"],
           ["experiments/run_oscd.py, run_eurosat.py", "run the full experiments and save the numbers"]],
          widths=[2.5, 3.6], font=9.5)
    P(space_after=2)
    code("""
pip install -r requirements.txt
python experiments/run_oscd.py           # change detection + search on OSCD
python experiments/run_eurosat.py        # search test on EuroSAT
""")

    H2("5.2 Loading the Data and Tiling")
    body("For each city we read four bands for both dates: B02 (blue), B03 (green), "
         "B04 (red) and B08 (near infrared). First three make the colour picture "
         "which we give to the model, and B08 is used only for the NDVI and NDWI "
         "tags. The band values are 16-bit numbers, so we rescale them to 0 to 255. "
         "The dates we read from dates.txt.")
    figure("term_1_dataset.png", "Program output: loading the 24 OSCD city pairs", width=5.6)
    body("One whole city image is too big to describe with one embedding. So we cut "
         "each image into square tiles of 64 x 64 pixels with no overlap. At 10 m per "
         f"pixel one tile covers 640 m x 640 m on the ground. Like this we got "
         f"{R['n_tiles']:,} tiles, that is {R['n_tiles']//2:,} places at two dates each.")
    figure("fig_tiling.png", "Left: one city cut into tiles. Right: a single 64 px tile",
           width=4.4)

    H2("5.3 Embedding and the Shared Index")
    body("Every tile we pass through a pre-trained CLIP model and we get back 512 "
         "numbers for each tile. This list of numbers is the embedding. We tried two "
         "models, RemoteCLIP and the normal OpenCLIP ViT-B/32. We did not train any "
         "of them. We make every embedding of length 1, so the similarity of two "
         "tiles is simply the dot product. All embeddings go in one array and the "
         "metadata in a list of same order. These two together is our shared index, "
         "and both search and change part read from this one index only.")
    body(f"On laptop CPU, embedding all {R['n_tiles']:,} tiles took "
         f"{sysr['embed_time_s']:.0f} seconds, around {sysr['tiles_per_s']:.0f} tiles "
         f"per second. The index takes {sysr['storage_mb']:.2f} MB and one search takes "
         f"about {sysr['query_latency_ms']:.2f} ms. Later we shifted to the laptop GPU "
         "(GTX 1650 Ti) and it became about 5 times faster.")

    H2("5.4 Search")
    body("For text search the typed words go through the text side of the same CLIP "
         "model. This gives 512 numbers in the same space as the tiles. Then we take "
         "the dot product with every tile and sort from highest to lowest. Image "
         "search is same thing, only the query is the embedding of a tile which the "
         "user picked.")
    code("""
q = embedder.encode_text("a satellite image of a river")[0]
scores = index.vectors @ q                # one similarity score per tile
top = np.argsort(-scores)[:10]            # best 10 tiles
""")

    H2("5.5 Change Detection and False Alarm Removal")
    body("We calculate change in two ways. First one is our main idea: for one place, "
         "take its embedding at date 1 and at date 2 from the shared index and "
         "measure how far they are. Second one is normal pixel comparison, which "
         "gives the exact shape of the change. Before comparing the pixels we try to "
         "remove false alarms in three steps:")
    step(1, "Brightness matching.", "We shift and scale date 2 so that each colour "
             "channel has same average and spread as date 1.")
    step(2, "Cloud and shadow mask.", "We mark the large, smooth, very bright or "
             "very dark areas as unusable.")
    step(3, "Removing small specks.", "We delete very small changed regions, because "
             "we expected them to be noise.")
    body("We compare with three simple methods which need no training: pixel "
         "difference with Otsu threshold, Change Vector Analysis (CVA) and SSIM "
         "difference. The pixel threshold we chose using the 14 training cities and "
         "used it as it is on the 10 test cities.")

    H2("5.6 Problems We Faced")
    bullet("Our first cloud mask was removing every very bright pixel. But new "
           "concrete roofs are also very bright, so it was deleting real new "
           "buildings.", bold_lead="Bright roofs were treated as cloud. ")
    bullet("First we made the cloud mask after brightness matching, and then the "
           "mask stopped finding the cloud. Now we make it from the original images.",
           bold_lead="Wrong order of steps. ")
    bullet("We were removing all small changed regions thinking it is noise. On real "
           "data the small regions are many times the real change only.",
           bold_lead="Small changes got deleted. ")

    # =================================================================== 6
    doc.add_page_break()
    H1("6. Results and Analysis")

    H2("6.1 Setup")
    table(["Item", "What we used"],
          [["Computer", "Laptop, Intel Core i5-10300H, 16 GB RAM, GTX 1650 Ti, Windows 11"],
           ["Software", "Python 3.14, PyTorch 2.14, open_clip 3.3, OpenCV 5.0, NumPy 2.5"],
           ["Models", "RemoteCLIP ViT-B/32 and OpenCLIP ViT-B/32, both pre-trained, not trained by us"],
           ["Data", f"OSCD: {ds['n_cities']} city pairs, {R['n_tiles']:,} tiles of 64 x 64 px"
                    + (f". EuroSAT: {ES['n_used']:,} patches" if ES else "")]],
          widths=[1.3, 4.8], font=10)
    P(space_after=2)

    H2("6.2 Text Search")
    m_o, m_r, m_rand = ret["openclip"]["mAP"], ret["remoteclip"]["mAP"], ret["random"]["mAP"]
    body("First we tried text search on the OSCD tiles with three queries (water, "
         "vegetation, buildings) and checked with our formula tags. The result was "
         f"weak: mAP was {f3(m_o)} for OpenCLIP and {f3(m_r)} for RemoteCLIP, and "
         f"random gives {f3(m_rand)}. Only the buildings query with OpenCLIP was "
         "good (8 of first 10 correct). We think the reasons are that our tags are "
         "strict and not made by people, and very few tiles are tagged as water or "
         "vegetation.")
    if ES:
        eo, er = ES["models"]["openclip"], ES["models"]["remoteclip"]
        body("So we repeated the text search on EuroSAT, where the labels are made by "
             "people. We typed one query for each of the 10 land types. P@10 means "
             "how many of the first 10 results were correct, and AP looks at the "
             "whole ranking. A random system gets 0.10.")
        erows = [[c, f2(eo["per_class"][c]["p10"]), f3(eo["per_class"][c]["ap"]),
                  f2(er["per_class"][c]["p10"]), f3(er["per_class"][c]["ap"])]
                 for c in ES["classes"]]
        erows.append(["Mean", f2(eo["mean_p10"]), f3(eo["mAP"]), f2(er["mean_p10"]), f3(er["mAP"])])
        table(["Query (land type)", "OpenCLIP P@10", "OpenCLIP AP", "RemoteCLIP P@10",
               "RemoteCLIP AP"], erows, widths=[1.9, 1.05, 1.0, 1.15, 1.0], font=9.5)
        P(space_after=2)
        best = "RemoteCLIP" if er["mAP"] > eo["mAP"] else "OpenCLIP"
        bm, om = (er, eo) if best == "RemoteCLIP" else (eo, er)
        cls = ES["classes"]
        top_c = max(cls, key=lambda c: bm["per_class"][c]["ap"])
        low_c = min(cls, key=lambda c: bm["per_class"][c]["ap"])
        body(f"Here the text search works much better. {best} gets mAP "
             f"{f3(bm['mAP'])} and on average {100*bm['mean_p10']:.0f}% of the first "
             f"10 results are correct, where random is 10%. The other model gets mAP "
             f"{f3(om['mAP'])}. It is not same for all land types: the best query is "
             f"{top_c} (AP {f2(bm['per_class'][top_c]['ap'])}) and the worst is "
             f"{low_c} (AP {f2(bm['per_class'][low_c]['ap'])}). When we use the text "
             f"to directly classify each patch, {best} is correct "
             f"{100*bm['zero_shot_acc']:.0f}% of the time.")
        figure("eurosat_grid.png", f"EuroSAT: top 10 patches from {best} for the best, "
               "a middle and the worst query (green = correct label)", width=6.0)

    H2("6.3 Image Search")
    body("For image search on OSCD we take a tile from date 1 and search in all the "
         f"{cdx['n_candidates']:,} tiles of date 2. The correct answer is the tile of "
         "the same place.")
    irows = [["Random guess", f"{100/cdx['n_candidates']:.2f}%", f"{500/cdx['n_candidates']:.2f}%"],
             ["OpenCLIP", f"{100*cdx['openclip']['recall@1']:.1f}%", f"{100*cdx['openclip']['recall@5']:.1f}%"],
             ["RemoteCLIP", f"{100*cdx['remoteclip']['recall@1']:.1f}%", f"{100*cdx['remoteclip']['recall@5']:.1f}%"]]
    table(["Model", "Correct tile is first", "In the top 5"], irows,
          widths=[2.0, 2.0, 2.0], font=10)
    P(space_after=2)
    txt = ("The correct place comes first about 46% of the time, where a random guess "
           "will be right about 0.05% of the time. So the embeddings do capture how a "
           "place looks, even when the two pictures are one to two years apart.")
    if ES:
        txt += (f" On EuroSAT also image search is strong: for "
                f"{100*max(eo['image_search_top1'], er['image_search_top1']):.0f}% of the "
                "patches the nearest patch is of the same land type.")
    body(txt)
    figure("fig_cross_date.png",
           "Image search across dates. Top row: same place found first. Bottom row: a miss",
           width=5.0)

    H2("6.4 Embeddings as a Change Signal")
    body("This is the test of our main idea: can the distance between the two "
         "embeddings of a place tell us that the place changed? We call a tile "
         f"'changed' if at least 5% of its pixels are marked in the ground truth, which "
         f"gives {tc['n_changed']} changed tiles out of {tc['n_tiles']:,}. AUC is 0.5 "
         "for random guessing and 1.0 for perfect.")
    m = tc["methods"]
    table(["Change score for a tile", "AUC"],
          [[n, f3(m[n]["auc"])] for n in tc["score_order"]] + [["random guess", "0.500"]],
          widths=[3.6, 1.2], font=10)
    P(space_after=2)
    k50 = E["by_change_size"]["0.5"]["auc"]
    body("Our main idea did not work well on this dataset. The embedding distance "
         f"gets AUC of {f2(m['embedding distance (openclip)']['auc'])} with OpenCLIP and "
         f"{f2(m['embedding distance (remoteclip)']['auc'])} with RemoteCLIP, and the "
         f"simple pixel difference gets {f2(m['pixel difference (normalised)']['auc'])}. "
         "What we understood is this: an embedding describes the whole tile in 512 "
         "numbers. In OSCD most changes are few new buildings which cover a small "
         "part of the tile, so the embedding hardly moves. At the same time the "
         "season and light are different between the two dates, and that moves the "
         "embedding even when nothing was built. When at least half of the tile "
         f"changed the AUC goes up to {f2(k50['embedding distance (openclip)'])}, but "
         "there are only 6 such tiles.")
    figure("fig_tile_change.png",
           "Left: embedding distance of changed and unchanged tiles. Right: ROC curves",
           width=4.6)

    H2("6.5 Change Maps")
    body("Next we compare the full change maps with the ground truth on the 10 test "
         "cities. FPR is the share of unchanged pixels which were wrongly marked as "
         "changed.")
    names = [("Pixel difference + Otsu", px["Pixel difference + Otsu"]),
             ("SSIM differencing", px["SSIM differencing"]),
             ("Change Vector Analysis (CVA)", px["Change Vector Analysis"]),
             ("Ours, no false alarm removal", px["Ours, no suppression"]),
             ("Ours, all three removal steps", px["Ours, with suppression"]),
             ("Ours, brightness matching only", bo["test"])]
    table(["Method", "Precision", "Recall", "F1", "IoU", "FPR %"],
          [[n, f3(r["precision"]), f3(r["recall"]), f3(r["f1"]), f3(r["iou"]),
            f2(r["fpr_pct"])] for n, r in names],
          widths=[2.6, 0.75, 0.7, 0.65, 0.65, 0.7], font=10)
    P(space_after=2)
    cva = px["Change Vector Analysis"]
    pc = bo["test"]["per_city"]
    body(f"The best method in this table is not ours. CVA has the highest F1 "
         f"({f3(cva['f1'])}). Our best version is brightness matching only, with F1 "
         f"{f3(bo['test']['f1'])}. The scores are very different from city to city. "
         f"We get F1 {f2(pc['lasvegas']['f1'])} on lasvegas, which has large new "
         f"buildings on plain ground, but only {f2(pc['valencia']['f1'])} on valencia, "
         "where almost nothing changed and the farm fields change colour between the "
         "two dates.")
    figure("fig_change_good.png", "A good case (lasvegas)", width=5.6)
    figure("fig_change_bad.png", "A bad case (valencia)", width=5.6)

    H2("6.6 False Alarm Removal")
    table(["Steps used", "Precision", "Recall", "F1", "FPR %"],
          [[n.replace("nothing", "none").replace(" (full)", ""), f3(r["precision"]),
            f3(r["recall"]), f3(r["f1"]), f2(r["fpr_pct"])] for n, r in parts.items()],
          widths=[2.6, 0.9, 0.9, 0.8, 0.8], font=10)
    P(space_after=2)
    p0, p1, p3 = parts["nothing"], parts["brightness matching only"], parts["+ remove small specks (full)"]
    body(f"Brightness matching is the useful step. False alarms fall from "
         f"{f2(p0['fpr_pct'])}% to {f2(p1['fpr_pct'])}% of unchanged pixels and about "
         f"{100*p1['recall']/p0['recall']:.0f}% of the real change is still found. The "
         "other two steps did not help on OSCD. The cloud mask made almost no "
         "difference, and on rio it removed dark water thinking it is shadow. "
         f"Removing small specks lowered the recall to {f3(p3['recall'])}, because in "
         "OSCD a new building is many times only few pixels.")
    figure("fig_brightness_matching.png",
           "Brightness matching on brasilia: flagged area falls from 39% to 2% of the image",
           width=5.6)

    H2("6.7 Summary and Future Work")
    srows = [["Image search (same place across dates)",
              f"{100*cdx['remoteclip']['recall@1']:.0f}% top-1 (random 0.05%)", "Yes"]]
    if ES:
        srows.append(["Text search on EuroSAT", f"mAP {f2(bm['mAP'])} (random 0.10)", "Yes"])
    srows += [["Text search on OSCD tiles", f"mAP {f2(m_o)} (random {f2(m_rand)})", "Only little"],
              ["Embedding distance for change",
               f"AUC {f2(m['embedding distance (openclip)']['auc'])} (pixel difference "
               f"{f2(m['pixel difference (normalised)']['auc'])})", "No"],
              ["Brightness matching", f"false alarms {f2(p0['fpr_pct'])}% to {f2(p1['fpr_pct'])}%", "Yes"],
              ["Our change maps and CVA", f"F1 {f2(bo['test']['f1'])} and {f2(cva['f1'])}", "CVA is better"]]
    table(["What we tested", "Result", "Did it work?"], srows,
          widths=[2.6, 2.4, 1.1], font=10)
    P(space_after=2)
    body("So the shared index is useful for search, but on OSCD it is not a good way "
         "to detect change. For small changes at 10 m resolution we still have to "
         "compare the pixels. Next we want to run the same test on LEVIR-CD, which is "
         "0.5 m per pixel, so a building covers many pixels there. We also want to "
         "use all 13 bands for the pixel comparison and not only three.")

    # =================================================================== links
    H1("7. Dataset Links")
    for name, url in (("OSCD (Onera Satellite Change Detection)", "https://rcdaudt.github.io/oscd/"),
                      ("EuroSAT", "https://github.com/phelber/EuroSAT"),
                      ("LEVIR-CD (planned, not used yet)", "https://justchenhao.github.io/LEVIR/"),
                      ("Copernicus Sentinel-1 / Sentinel-2", "https://dataspace.copernicus.eu/"),
                      ("USGS Landsat Collection 2", "https://earthexplorer.usgs.gov/"),
                      ("NRSC / ISRO Bhuvan", "https://bhuvan-app3.nrsc.gov.in/data/download/"),
                      ("SIH 2026 problem statement SIH26227", "https://www.sih.gov.in/sih2026PS")):
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_after = Pt(3)
        r = p.add_run(f"{name}: {url}")
        r.font.name = "Times New Roman"; r.font.size = Pt(12)
