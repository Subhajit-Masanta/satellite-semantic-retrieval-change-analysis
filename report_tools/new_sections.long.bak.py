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

    # =================================================================== 5
    H1("4. Dataset Details")

    H2("4.1 Why We Chose OSCD")
    body("We needed a dataset which has two images of same place taken at different "
         "times, and also the correct answer of what changed, so that we can check "
         "our output in numbers. The Onera Satellite Change Detection (OSCD) dataset "
         "gives exactly this. We took it for three reasons.")
    bullet("It is real Sentinel-2 data, which is one of the image sources named in "
           "the problem statement. So our test is on same kind of data which the real "
           "system will see.", bold_lead="It is real satellite data. ")
    bullet("The two images of every city are already kept one on top of other "
           "properly. Aligning two satellite images is itself a hard problem, and "
           "this lets us skip it.", bold_lead="The pairs are already aligned. ")
    bullet("Whole dataset is around 513 MB, so we could download it and run on a "
           "laptop.", bold_lead="It is small enough for us. ")
    body("We had planned to use LEVIR-CD dataset also (637 pairs, about 3.8 GB). But "
         "we have not run it yet. So every result in this report is from OSCD only, "
         "and LEVIR-CD is kept as next step.")

    H2("4.2 What the Dataset Contains")
    table(["Property", "Value"],
          [["Number of image pairs", f"{ds['n_cities']} cities (14 for training, 10 for testing, as fixed by the dataset authors)"],
           ["Satellite and sensor", "Sentinel-2, MultiSpectral Instrument (MSI)"],
           ["Bands per image", "13 (B01 to B12 and B8A), stored as separate GeoTIFF files"],
           ["Pixel type", "16-bit unsigned integer (uint16)"],
           ["Ground resolution", "10 m per pixel after resampling (one pixel = 10 m x 10 m)"],
           ["Years covered", "2015 to 2018; the two dates of a pair are about 1 to 2.5 years apart"],
           ["Image sizes", "From 385 x 241 px (norcia) to 1070 x 1180 px (beirut)"],
           ["Ground truth", "One black and white change mask per city (white = changed)"],
           ["Type of change labelled", "Mostly urban change: new buildings and new roads"],
           ["Total pixels", f"{ds['total_pixels']:,}"],
           ["Pixels labelled as changed", f"{ds['changed_pixels']:,} ({ds['changed_pct']:.2f}%)"]],
          widths=[2.0, 4.1])
    P()
    body("Each city folder is having the files shown below. We use the 'rect' "
         "folders, because in those the two dates are already aligned with each "
         "other.")
    code("""
paris/
  imgs_1/        13 band files, date 1 (original crop)
  imgs_2/        13 band files, date 2 (original crop)
  imgs_1_rect/   13 band files, date 1 (aligned)   <- we use this
  imgs_2_rect/   13 band files, date 2 (aligned)   <- we use this
  pair/          ready-made colour pictures of both dates
  dates.txt      date_1: 20161130   date_2: 20171107
  paris.geojson  outline of the area in longitude / latitude
""")
    figure("fig_dataset_samples.png",
           "Three city pairs from OSCD with their ground-truth change masks", width=5.0)

    H2("4.3 The 24 City Pairs")
    body("Below table lists every pair which we used. The last column shows how much "
         "of the image is marked as changed. This number is small everywhere, and "
         "that becomes important later in our results.")
    table(["City", "Split", "Size (px)", "Date 1", "Date 2", "Changed %"],
          [[r["city"], r["split"], f"{r['width']} x {r['height']}", r["date_1"],
            r["date_2"], f"{r['changed_pct']:.2f}"] for r in inv],
          widths=[1.3, 0.7, 1.2, 1.1, 1.1, 0.9], font=9.5)
    P()

    H2("4.4 Metadata We Extract")
    body("Metadata means the information about an image, not the image itself. We "
         "collect it from three places: the dates.txt file, the .geojson file and the "
         "GeoTIFF file header. Then we attach it to every tile, so that any search "
         "result or change result can be traced back to a city, a date and a "
         "position.")
    ex = R["example_metadata"]
    table(["Field", "Example", "Where it comes from", "Why we keep it"],
          [["tile_id", ex["tile_id"], "we build it", "unique name for each tile"],
           ["city", ex["city"], "folder name", "tells us which scene the tile is from"],
           ["date", ex["date"], "dates.txt", "needed to compare date 1 with date 2 and to filter by date"],
           ["date_slot", str(ex["date_slot"]), "which folder (imgs_1 or imgs_2)", "pairs the two dates of one place"],
           ["sensor", ex["sensor"], "dataset documentation", "lets a user filter by satellite"],
           ["gsd_m", str(ex["gsd_m"]), "dataset documentation", "size of one pixel on the ground in metres"],
           ["x, y", f"{ex['x']}, {ex['y']}", "tiling step", "position of the tile inside the city image"],
           ["tile_px", str(ex["tile_px"]), "our setting", "tile size, so results can be repeated"],
           ["split", ex["split"], "dataset authors", "keeps train and test cities separate"],
           ["changed_frac", str(ex["changed_frac"]), "ground-truth mask", "share of the tile that really changed (only used for scoring)"],
           ["water / vegetation / builtup", "true or false", "NDWI and NDVI from bands B03, B04, B08", "automatic labels to score text search"],
           ["footprint (lon/lat)", str(inv[0]["geo"].get("footprint_lonlat")), ".geojson file", "where the scene is on the Earth"]],
          widths=[1.25, 1.55, 1.5, 1.8], font=9)
    P()
    body("One thing we noticed while reading the files. The aligned GeoTIFF files do "
         "not have any coordinate system inside them (the reader gives 'crs: None'). "
         "So we cannot get longitude and latitude from the image file itself. We are "
         "taking the location from the .geojson file. This means our location is "
         "correct for the whole scene, but we do not have exact coordinates for each "
         "single tile.")
    body("The second last row needs small explanation. OSCD does not tell which tiles "
         "are showing water or vegetation, but we need some correct answer to score "
         "the text search. So we calculate two standard indices from the bands, NDVI "
         "(vegetation) and NDWI (water). A tile is tagged 'water' if at least 30% of "
         "it has NDWI above 0.1, 'vegetation' if at least 70% has NDVI above 0.5, and "
         "'builtup' if at least 70% has low NDVI and low NDWI. These tags are made by "
         "formula and not by a person, so they are not perfect. We use them only for "
         "scoring and never as input to our system.")

    H2("4.5 Limitations of the Dataset")
    bullet(f"Only {ds['changed_pct']:.1f}% of all pixels are changed. A method which "
           "says 'nothing changed' everywhere will be right about 97% of the time, so "
           "accuracy is a useless measure here. That is why we report precision, "
           "recall, F1 and IoU.", bold_lead="Very few changed pixels. ")
    bullet("At 10 m per pixel a house is only one or two pixels wide. Small changes "
           "are hard to see even for a person.", bold_lead="Low resolution. ")
    bullet("The two dates of a pair are many times in different months, for example "
           "August and October. So trees, fields and sunlight look different even "
           "where nothing was built.", bold_lead="Different seasons. ")
    bullet("The masks mark only urban change. If a field was ploughed or a lake got "
           "smaller it is not marked, so our method can be 'wrong' on something which "
           "really did change.", bold_lead="Only urban change is labelled. ")
    bullet("24 pairs is a small number, so our scores can move a lot because of one "
           "or two cities.", bold_lead="Small dataset. ")

    # =================================================================== 6
    doc.add_page_break()
    H1("5. Implementation")

    H2("5.1 Setup and Code Structure")
    body("We wrote the project in Python. We split the code in small files so that "
         "each of us can work on own part without waiting for others. The table shows "
         "what each file is doing.")
    table(["File", "What it does"],
          [["src/ingest.py", "reads images and cuts them into tiles, keeps metadata with each tile"],
           ["src/embedder.py", "loads the pre-trained CLIP model and turns tiles or text into embeddings"],
           ["src/index_builder.py", "stores all embeddings in one shared index and searches it"],
           ["src/retrieval.py, filters.py", "text search, image search, and date / sensor / area filters"],
           ["src/change_detect.py", "change score from embeddings, pixel change map, and the three baseline methods"],
           ["src/suppress.py", "brightness matching, cloud / shadow mask, removing small specks"],
           ["src/metrics.py, visualizer.py", "Precision@k, AP, precision, recall, F1, IoU, and the figures"],
           ["experiments/run_oscd.py", "runs the full experiment on OSCD and saves every number"],
           ["experiments/extra_analysis.py", "the follow-up checks (tile size, change size, false alarms)"]],
          widths=[2.2, 3.9], font=10)
    P()
    body("To run the project we install the libraries, download OSCD in the data "
         "folder, and then run two scripts. First run also downloads the model "
         "weights (about 600 MB each for the two CLIP models).")
    code("""
pip install -r requirements.txt
python download_data.py --check          # checks the dataset is in place
python experiments/run_oscd.py           # main experiment (about 6 minutes)
python experiments/extra_analysis.py     # follow-up checks
python experiments/make_figures.py       # draws all result figures
""")
    body("The output of these scripts gets saved in the results folder "
         "(run_log.txt and results.json).")

    H2("5.2 Loading the Data and Extracting Metadata")
    body("For each city we read four bands for both dates: B02 (blue), B03 (green), "
         "B04 (red) and B08 (near infrared). First three make the colour picture "
         "which we give to the model. B08 is used only for the NDVI and NDWI tags. "
         "The band values are 16-bit numbers, so we rescale them to 0 to 255.")
    body("Here one detail is important. We rescale both dates of a city with the same "
         "minimum and maximum. If we rescale each date separately, we will hide the "
         "brightness difference between the two dates. But that brightness difference "
         "is one of the false alarm sources which we want to test, so we keep it.")
    code("""
for i, b in enumerate(("B02", "B03", "B04")):
    both = np.concatenate([bands[1][b].ravel(), bands[2][b].ravel()])
    lo, hi = np.percentile(both, [2, 98])       # same stretch for both dates
    for d in (1, 2):
        ch = np.clip((bands[d][b] - lo) / (hi - lo) * 255, 0, 255)
""")
    body("The dates we read from dates.txt and change into normal year-month-day "
         "form. The scene outline we read from the .geojson file. The screenshot "
         "below is the first part of our program output, it shows all 24 cities "
         "getting loaded with their size, dates and changed percentage.")
    figure("term_1_dataset.png", "Program output: loading the 24 OSCD city pairs", width=6.1)

    H2("5.3 Tiling")
    body("One whole city image is too big to describe with one embedding, and we want "
         "results which point to a small area. So we cut each image into square tiles "
         "of 64 x 64 pixels with no overlap. At 10 m per pixel, one tile covers "
         "640 m x 640 m on the ground. The parts at right and bottom edge which do "
         f"not fill a whole tile are left out. Like this we got {R['n_tiles']:,} "
         f"tiles, that is {R['n_tiles']//2:,} places at two dates each.")
    figure("fig_tiling.png", "Left: one city cut into tiles. Right: a single 64 px tile "
           "as the model sees it", width=5.4)
    body("The right side of the figure is showing a real problem. A 64 px tile is "
         "very blurry. The CLIP model expects 224 x 224 px images, so every tile gets "
         "enlarged about 3.5 times before going in. We come back to this in results, "
         "because we think it is one reason why our text search is weak.")

    H2("5.4 Embedding and the Shared Index")
    body("Every tile we pass through a pre-trained CLIP model and we get back 512 "
         "numbers for each tile. This list of numbers is the embedding. We tried two "
         "models: RemoteCLIP, which was trained further on satellite and aerial "
         "pictures, and the normal OpenCLIP ViT-B/32. We did not train any of them. "
         "We put the model in evaluation mode and switch off the gradients.")
    code("""
self.model = model.to(self.device).eval()        # frozen, prediction only
for p in self.model.parameters():
    p.requires_grad = False
...
with torch.no_grad():
    feats = self.model.encode_image(batch)
return feats / np.linalg.norm(feats, axis=1, keepdims=True)   # length 1
""")
    body("We make every embedding of length 1. After that the similarity of two "
         "tiles is simply the dot product of their embeddings (cosine similarity). "
         "All embeddings go in one array, and the metadata goes in a list in the same "
         "order. These two together is our shared index. Both search part and change "
         "part read from this one index only.")
    figure("term_2_tiles_index.png",
           "Program output: tiling, the metadata of one tile, embedding time and index size",
           width=6.1)
    body(f"On our laptop CPU, embedding all {R['n_tiles']:,} tiles took "
         f"{sysr['embed_time_s']:.0f} seconds with RemoteCLIP, which is around "
         f"{sysr['tiles_per_s']:.0f} tiles per second. The index takes "
         f"{sysr['storage_mb']:.2f} MB, and one search takes about "
         f"{sysr['query_latency_ms']:.2f} ms. In this run the index used plain NumPy "
         "search, because FAISS was not installed on the machine. Our code shifts to "
         "NumPy automatically. With only few thousand tiles this is fast enough, but "
         "for millions of tiles FAISS will be needed.")

    H2("5.5 Search")
    body("For text search, the typed words go through the text side of the same CLIP "
         "model. This gives 512 numbers in the same space as the tiles. Then we take "
         "the dot product with every tile and sort from highest to lowest. Image "
         "search is same thing, only the query is the embedding of a tile which the "
         "user picked.")
    code("""
q = embedder.encode_text("a satellite image of a river, sea or water body")[0]
scores = index.vectors @ q                # one similarity score per tile
top = np.argsort(-scores)[:10]            # best 10 tiles
""")
    body("We wrote the queries like 'a satellite image of ...' because CLIP was "
         "trained on captions which look like sentences, not single words.")

    H2("5.6 Change Detection and False Alarm Removal")
    body("We calculate change in two ways. First one is our main idea: for one place, "
         "take its embedding at date 1 and at date 2 from the shared index, and "
         "measure how far they are. Second one is normal pixel comparison, which "
         "gives the exact shape of the change.")
    code("""
def embedding_change_score(vec_t1, vec_t2):
    return 1.0 - np.dot(vec_t1, vec_t2)      # 0 = same, bigger = more different
""")
    body("Before comparing the pixels we try to remove false alarms in three steps:")
    step(1, "Brightness matching.", "We shift and scale date 2 so that each colour "
             "channel has same average and spread as date 1. This removes the "
             "differences which come only from sunlight or haze.")
    step(2, "Cloud and shadow mask.", "We mark the large, smooth, very bright or "
             "very dark areas as unusable and ignore the changes inside them.")
    step(3, "Removing small specks.", "We delete very small changed regions, because "
             "we expected them to be noise.")
    code("""
t2    = radiometric_normalise(img_t1, img_t2)     # step 1
mask  = change_mask(img_t1, t2, threshold)        # pixel comparison
valid = usable_mask(img_t1, img_t2)               # step 2 (on the raw images)
mask  = cv2.bitwise_and(mask, valid)
mask  = drop_small_regions(mask)                  # step 3
""")
    body("We compare with three simple baseline methods which need no training: "
         "pixel difference with Otsu threshold, Change Vector Analysis (CVA), and "
         "SSIM difference. The pixel threshold of our method we chose using only the "
         "14 training cities, and then used it as it is on the 10 test cities.")

    H2("5.7 Problems We Faced")
    body("Some things did not work in the first time. We are listing them because "
         "they changed how we built the system.")
    bullet("Our first cloud mask was simply removing every very bright pixel. But "
           "new concrete roofs are also very bright, so it was deleting real new "
           "buildings. We changed it to remove only the areas which are bright, "
           "smooth and large.", bold_lead="Bright roofs were treated as cloud. ")
    bullet("First we made the cloud mask after brightness matching. Brightness "
           "matching makes a cloud darker, so the mask stopped finding it. Now we "
           "make the mask from the original images.",
           bold_lead="Wrong order of steps. ")
    bullet("In our first try we were removing all the small changed regions "
           "thinking it is noise. Later on real data we saw that small regions are "
           "many times the real change only.", bold_lead="Small changes got deleted. ")

    # =================================================================== 7
    doc.add_page_break()
    H1("6. Results and Analysis")

    H2("6.1 Experimental Setup")
    table(["Item", "What we used"],
          [["Computer", "Laptop, Intel Core i5-10300H (4 cores), 16 GB RAM, Windows 11"],
           ["GPU", "Not used. We installed the CPU version of PyTorch, so everything ran on the CPU"],
           ["Software", "Python 3.14, PyTorch 2.14, open_clip 3.3, OpenCV 5.0, NumPy 2.5"],
           ["Models", "RemoteCLIP ViT-B/32 and OpenCLIP ViT-B/32 (LAION-2B), both frozen"],
           ["Data", f"OSCD, {ds['n_cities']} city pairs, {R['n_tiles']:,} tiles of 64 x 64 px"],
           ["Train / test", "14 training cities used only to pick thresholds; 10 test cities used for the reported scores"],
           ["Total run time", f"{R['total_runtime_s']:.0f} seconds for the main experiment"]],
          widths=[1.4, 4.7], font=10)
    P()
    body("No model was trained. The pixel threshold we picked using the training "
         "cities only, and the scores below are on the test cities.")

    H2("6.2 Text Search")
    body("We ran three text queries and checked the top results with the automatic "
         "tags from Section 4.4. P@10 means how many of the first 10 results were "
         "correct. AP (average precision) looks at the whole ranking. The 'random' "
         "row is what we get if the tiles are shuffled, so it is the score of a "
         "system which knows nothing.")
    rows = []
    for key, label in (("water", "water"), ("vegetation", "vegetation"), ("builtup", "buildings")):
        rows.append([label, str(ret["openclip"][key]["n_relevant"]),
                     f2(ret["random"][key]["p10"]), f3(ret["random"][key]["ap"]),
                     f2(ret["openclip"][key]["p10"]), f3(ret["openclip"][key]["ap"]),
                     f2(ret["remoteclip"][key]["p10"]), f3(ret["remoteclip"][key]["ap"])])
    rows.append(["Mean AP (mAP)", "", "", f3(ret["random"]["mAP"]), "", f3(ret["openclip"]["mAP"]),
                 "", f3(ret["remoteclip"]["mAP"])])
    table(["Query", "Correct tiles", "Random P@10", "Random AP", "OpenCLIP P@10",
           "OpenCLIP AP", "RemoteCLIP P@10", "RemoteCLIP AP"], rows,
          widths=[1.05, 0.7, 0.7, 0.7, 0.75, 0.75, 0.8, 0.8], font=9)
    P()
    body(f"The result is weak. The mAP is {f3(ret['openclip']['mAP'])} for OpenCLIP and "
         f"{f3(ret['remoteclip']['mAP'])} for RemoteCLIP, while random gives "
         f"{f3(ret['random']['mAP'])}. So text search is only little better than "
         "guessing. Only the 'buildings' query with OpenCLIP looks good (8 of the "
         "first 10 correct), and RemoteCLIP got 0 of 10 on that same query. We were "
         "expecting RemoteCLIP to be better because it was trained on overhead "
         "images, but on this data it was not.")
    figure("fig_retrieval_builtup_openclip.png",
           "Top 10 tiles for the 'buildings' query with OpenCLIP (green = correct by our tag)",
           width=4.5)
    body("Seeing the pictures helped us to understand the numbers. We think there are "
         "three reasons for the weak result.")
    bullet("A 64 px tile at 10 m resolution has very less detail. The models were "
           "trained on much sharper pictures.", bold_lead="The tiles are too blurry. ")
    bullet("When we saw the results of the water query, some returned tiles did show "
           "water along one edge, but they got marked wrong because our tag needs "
           "30% of the tile to be water. So some part of the low score is coming from "
           "our own scoring rule.", bold_lead="Our tags are strict. ")
    bullet("Only 47 tiles are tagged as vegetation and 336 as water, out of "
           f"{R['n_tiles']:,}. With so few correct tiles, a few mistakes at the top "
           "change the score a lot.", bold_lead="Few correct tiles. ")
    body("We also tried bigger tiles of 128 px. The mAP did not improve "
         f"({f3(t128['openclip']['mAP'])} for OpenCLIP and "
         f"{f3(t128['remoteclip']['mAP'])} for RemoteCLIP), so tile size alone is not "
         "fixing it.")

    H2("6.3 Finding the Same Place Across Dates")
    body("This is the image search test. We take a tile from date 1 and search in all "
         f"the {cdx['n_candidates']:,} tiles of date 2. The correct answer is the tile "
         "of the same place. Here we do not need any tags, because we know exactly "
         "which tile is the right one.")
    table(["Model", "Correct tile is first", "In the top 5", "In the top 10"],
          [["Random guess", f"{100/cdx['n_candidates']:.2f}%", f"{500/cdx['n_candidates']:.2f}%",
            f"{1000/cdx['n_candidates']:.2f}%"],
           ["OpenCLIP", f"{100*cdx['openclip']['recall@1']:.1f}%",
            f"{100*cdx['openclip']['recall@5']:.1f}%", f"{100*cdx['openclip']['recall@10']:.1f}%"],
           ["RemoteCLIP", f"{100*cdx['remoteclip']['recall@1']:.1f}%",
            f"{100*cdx['remoteclip']['recall@5']:.1f}%", f"{100*cdx['remoteclip']['recall@10']:.1f}%"]],
          widths=[1.6, 1.6, 1.4, 1.4], font=10.5)
    P()
    body("This is our best result. The correct place comes first about 46% of the "
         "time, where a random guess will be right about 0.05% of the time. So the "
         "embeddings do capture how a place looks, even when the two pictures are one "
         "to two years apart and taken in different seasons. With 128 px tiles the "
         f"top-1 score went up to {100*t128['openclip']['recall@1']:.0f}% for OpenCLIP, "
         f"but there were also less tiles to choose from ({t128['n_tiles']//2}), so "
         "that number cannot be compared directly.")
    figure("fig_cross_date.png",
           "Image search across dates. Top row: the same place is found first. "
           "Bottom row: a miss, where the results are similar-looking towns", width=5.3)
    body("The bottom row shows how it fails. For a tile of red-roof houses, the "
         "system returns other red-roof towns. They look alike, so this mistake is "
         "understandable.")

    H2("6.4 Embeddings as a Change Signal")
    body("This is the test of our main idea: can the distance between the two "
         "embeddings of a place tell us that the place changed? We call a tile "
         "'changed' if at least 5% of its pixels are marked in the ground truth. "
         f"That gives {tc['n_changed']} changed tiles out of {tc['n_tiles']:,}. We use "
         "AUC to score this. AUC is 0.5 for random guessing and 1.0 for perfect.")
    m = tc["methods"]
    table(["Change score for a tile", "AUC", "AP"],
          [[n, f3(m[n]["auc"]), f3(m[n]["ap"])] for n in tc["score_order"]]
          + [["random guess", "0.500", f3(tc["changed_share"])]],
          widths=[3.4, 1.0, 1.0], font=10.5)
    P()
    body("Our main idea did not work well on this dataset. The embedding distance "
         f"gets AUC of {f2(m['embedding distance (openclip)']['auc'])} with OpenCLIP and "
         f"{f2(m['embedding distance (remoteclip)']['auc'])} with RemoteCLIP. The simple "
         f"pixel difference gets {f2(m['pixel difference (normalised)']['auc'])}. So the "
         "embedding is a weaker change signal than just subtracting the pixels, and "
         "RemoteCLIP is almost same as guessing.")
    figure("fig_tile_change.png",
           "Left: embedding distance for changed and unchanged tiles. Right: ROC curves "
           "of the four change scores", width=5.0)
    k50 = E["by_change_size"]["0.5"]["auc"]
    body("The left plot shows why. The embedding distance of changed tiles and "
         "unchanged tiles are overlapping almost completely. Then we checked if the "
         "size of the change matters.")
    body("It does. When at least half of the tile changed, the OpenCLIP embedding "
         f"distance reaches AUC of {f2(k50['embedding distance (openclip)'])}. But "
         "there are only 6 such tiles, so this number is not reliable, and pixel "
         "difference is still higher at every size. What we understood is this: an "
         "embedding describes the whole tile in 512 numbers. In OSCD most changes are "
         "few new buildings which cover a small part of the tile, so the description "
         "hardly moves. At the same time the season and light are different between "
         "the two dates, and that moves the embedding even when nothing was built. So "
         "the small real signal gets hidden by the large unwanted one.")

    H2("6.5 Pixel-Level Change Maps")
    body("Next we compare the full change maps with the ground truth on the 10 test "
         "cities. All pixels of the 10 cities are counted together. FPR is the share "
         "of unchanged pixels which were wrongly marked as changed.")
    names = [("Pixel difference + Otsu", px["Pixel difference + Otsu"]),
             ("SSIM differencing", px["SSIM differencing"]),
             ("Change Vector Analysis (CVA)", px["Change Vector Analysis"]),
             ("Ours, no false alarm removal", px["Ours, no suppression"]),
             ("Ours, all three removal steps", px["Ours, with suppression"]),
             ("Ours, all steps + embedding check", px["Ours, suppression + embedding gate"]),
             ("Ours, brightness matching only", bo["test"])]
    table(["Method", "Precision", "Recall", "F1", "IoU", "FPR %"],
          [[n, f3(r["precision"]), f3(r["recall"]), f3(r["f1"]), f3(r["iou"]),
            f2(r["fpr_pct"])] for n, r in names],
          widths=[2.6, 0.75, 0.7, 0.65, 0.65, 0.7], font=10)
    P()
    cva = px["Change Vector Analysis"]
    body(f"The best method in this table is not ours. CVA has the highest F1 "
         f"({f3(cva['f1'])}) and the highest precision ({f3(cva['precision'])}). Our "
         f"best version is brightness matching only, with F1 {f3(bo['test']['f1'])}. "
         f"Our full method with all three removal steps is lower "
         f"({f3(px['Ours, with suppression']['f1'])}). Using the embedding to throw "
         "away the tiles which it thinks are unchanged also did not help. All scores "
         "are low, which is expected for methods which learn nothing from labelled "
         "examples.")
    figure("fig_change_good.png", "A good case (lasvegas): large new buildings on plain ground",
           width=5.7)
    figure("fig_change_bad.png", "A bad case (valencia): almost nothing changed, and both "
           "methods mark the wrong places", width=5.7)
    pc = bo["test"]["per_city"]
    body("The scores are very different from city to city. Our brightness-only "
         f"method gets F1 {f2(pc['lasvegas']['f1'])} on lasvegas and "
         f"{f2(pc['montpellier']['f1'])} on montpellier, but only "
         f"{f2(pc['valencia']['f1'])} on valencia and {f2(pc['saclay_w']['f1'])} on "
         "saclay_w. The good cities have large new buildings on dry plain ground. The "
         "bad cities have very little real change (under 1.2% of pixels) and lot of "
         "farmland, where fields change colour between the two dates. Those field "
         "changes are real differences in the picture, but they are not marked in the "
         "ground truth, so they get counted as false alarms.")

    H2("6.6 False Alarm Removal")
    body("Here we look at each removal step separately, all at the same pixel "
         "threshold of 40, on the test cities.")
    table(["Steps used", "Precision", "Recall", "F1", "FPR %"],
          [[n.replace("nothing", "none").replace(" (full)", ""), f3(r["precision"]),
            f3(r["recall"]), f3(r["f1"]), f2(r["fpr_pct"])] for n, r in parts.items()],
          widths=[2.6, 0.9, 0.9, 0.8, 0.8], font=10.5)
    P()
    p0, p1, p3 = parts["nothing"], parts["brightness matching only"], parts["+ remove small specks (full)"]
    body(f"Brightness matching is clearly the useful step. False alarms fall from "
         f"{f2(p0['fpr_pct'])}% to {f2(p1['fpr_pct'])}% of unchanged pixels, and "
         f"recall drops only from {f3(p0['recall'])} to {f3(p1['recall'])}, so about "
         f"{100*p1['recall']/p0['recall']:.0f}% of the real change is kept. F1 goes "
         f"from {f3(p0['f1'])} to {f3(p1['f1'])}.")
    figure("fig_brightness_matching.png",
           "Effect of brightness matching on brasilia: the flagged area falls from 39% "
           "to 2% of the image", width=5.7)
    body("The other two steps did not help on OSCD.")
    bullet(f"It made almost no difference in the scores, and on some cities it "
           f"removed a large area which had no cloud at all "
           f"({E['cloud_mask_pct_per_test_city']['rio']:.0f}% of rio and "
           f"{E['cloud_mask_pct_per_test_city']['brasilia']:.0f}% of brasilia). It "
           "took dark water and dark vegetation as shadow. OSCD images have almost no "
           "cloud, so here this step could only do harm.",
           bold_lead="Cloud and shadow mask. ")
    bullet(f"This step lowered the recall from {f3(parts['+ cloud/shadow mask']['recall'])} "
           f"to {f3(p3['recall'])}. In OSCD a new building is many times only few "
           "pixels, so what we treated as noise was actually real change.",
           bold_lead="Removing small specks. ")
    figure("fig_cloudmask_wrong.png",
           "Our cloud/shadow mask removing dark water and vegetation in rio by mistake",
           width=4.4)
    body("Lastly we checked the false alarm removal on the test cities, with and "
         "without the removal steps at the same threshold. For us it should do two "
         "things: false alarms must go down, and at least 80% of the real change "
         "must still be found.")
    figure("fig_gatec.png", "False alarms and recall on the test cities at each pixel threshold", width=4.8)
    g20, g40 = G["20"], G["40"]
    body("False alarms go down at every threshold, so the first condition always "
         "passes. The second condition passes only at threshold 20 "
         f"({100*g20['recall_kept']:.0f}% of recall kept). At threshold 40 only "
         f"{100*g40['recall_kept']:.0f}% is kept, so the check fails. The reason is the "
         "speck removal step explained above. With brightness matching alone, about "
         f"{100*p1['recall']/p0['recall']:.0f}% of recall is kept at threshold 40, "
         "which will pass. So our full false alarm removal does not pass our own "
         "check on real data, but the brightness matching part of it does.")

    H2("6.7 Summary and Future Work")
    body("This table puts our main findings in one place.")
    table(["What we tested", "Result", "Did it work?"],
          [["Search for the same place across dates",
            f"{100*cdx['remoteclip']['recall@1']:.0f}% top-1 (random: 0.05%)", "Yes"],
           ["Text search", f"mAP {f2(ret['openclip']['mAP'])} (random: {f2(ret['random']['mAP'])})",
            "Only slightly"],
           ["Embedding distance as change signal",
            f"AUC {f2(m['embedding distance (openclip)']['auc'])} (pixel difference: "
            f"{f2(m['pixel difference (normalised)']['auc'])})", "No"],
           ["Brightness matching", f"false alarms {f2(p0['fpr_pct'])}% to {f2(p1['fpr_pct'])}%", "Yes"],
           ["Cloud mask and speck removal", "no gain, or lost real change", "No"],
           ["Our change maps against CVA", f"F1 {f2(bo['test']['f1'])} against {f2(cva['f1'])}", "No, CVA is better"],
           ["RemoteCLIP against normal CLIP", "no clear advantage on this data", "No"]],
          widths=[2.5, 2.4, 1.2], font=10)
    P()
    body("So the shared index is useful for searching by example image, but on OSCD "
         "it is not a good way to detect change. One set of embeddings can do the "
         "search work, but for small changes at 10 m resolution we still have to "
         "compare the pixels.")
    body("These results have some limits also. They are from one dataset with 24 "
         "pairs, our search tags are made by formula, and we tried only two models "
         "and two tile sizes. So we cannot say the idea fails everywhere, only that "
         "it did not work in this setting.")
    body("What we are planning to do next:")
    bullet("Run the same experiment on LEVIR-CD, which is 0.5 m per pixel. There a "
           "building covers many pixels, so the embedding may react more to a new "
           "building. This is the most important next step.")
    bullet("Use patch-level features from the model instead of one embedding for the "
           "whole tile, so that a small change is not averaged away.")
    bullet("Drop the speck removal step for low-resolution data, and replace our "
           "cloud mask with the cloud layer that comes with Sentinel-2 products.")
    bullet("Use all 13 bands for the pixel comparison. We only used three colour "
           "bands, while the infrared bands show vegetation and water much better.")
