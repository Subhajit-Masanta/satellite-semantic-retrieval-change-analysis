> **Status note (Oct 2026):** this is the original September plan. The project has since been run on real data and some claims changed (change detection is weak at 10 m, cloud mask and speck removal were left off, clustering is demo-only). For current results and scope read `README.md` and `TEAM_NOTES.md`.

# 🛰️ BCSE407L Computer Vision — RIL Execution Plan
### Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery
#### 4-Member Team | Execution & Demonstration Focus | PS SIH26227

---

## 🏗️ How the Team Works (Parallel, Not a Chain)

You **DO NOT** wait for each other. Every person can build and test their part **in parallel** using dummy vectors, hand-edited images, or synthetic masks.

| Person | Stage | How to work independently on Day 1 |
|---|---|---|
| **P1** | Ingestion + Embedding + Index | Download LEVIR-CD, tile it, embed 50 tiles, print vector shapes. Needs nobody. |
| **P2** | Semantic Retrieval | Build FAISS over `np.random.rand(200, 512)` dummy vectors. Verify ranking/top-k/filters work. Swap in P1's real vectors later — same interface. |
| **P3** | Change Analysis + Suppression | Take any image, paint a patch on a copy, confirm your pipeline flags exactly that patch. Then brighten the whole copy and confirm suppression reports *no change*. |
| **P4** | Evaluation + Integration + Demo | Hand-draw a 10×10 ground-truth mask and a prediction mask, verify F1/IoU by hand-arithmetic. Start writing Sections 1–2 immediately. |

*Once everyone's part works, P4 connects them like Lego blocks in `main.py`.*

---

## 👥 Task Division (Pure Execution)

### 👤 Person 1: The Librarian (Data, Embedding & Index)
**Goal:** Turn a pile of satellite images into a searchable vector index.
* **Code to write:** `ingest.py`, `embedder.py`, `index_builder.py`
* **What it does:** Reads GeoTIFF/COG with `rasterio`, tiles images into patches, keeps geo-metadata (CRS, bounds, date, source scene) beside each tile, runs a **frozen** RemoteCLIP/CLIP encoder to get one vector per tile, and builds a FAISS index.
* **Testing:** Embed 50 tiles and print vector shape (should be e.g. `(50, 512)`). Build the index, save it, reload it, confirm it returns neighbours.
* **Record from Day 1:** model name/version/source/licence, index build time, storage size, tile count. The PS requires all of these in the eval report.

### 👤 Person 2: The Search Engine (Semantic Retrieval)
**Goal:** Let someone type *"newly built structures near a river"* and get the right tiles back.
* **Code to write:** `retrieval.py`, `image_search.py`, `filters.py`, *(stretch)* `clustering.py`
* **What it does:** Encodes a text query into the same vector space, ranks all tiles by cosine similarity, returns top-*k*. Also does image-to-image search, and applies AOI/date/sensor metadata filters on top of the semantic ranking.
* **Testing:** Run on dummy random vectors first — you're checking that ranking, filtering and top-*k* mechanics work, not that results are meaningful. Once P1 hands over real vectors, your 5 test queries should return sensible tiles.

### 👤 Person 3: The Detective (Change Analysis & Suppression)
**Goal:** Find what actually changed between two dates — and refuse to be fooled by season, cloud or lighting.
* **Code to write:** `change_detect.py`, `change_classify.py`, `suppress.py`
* **What it does:** Computes the change signal from the **same embeddings P1 built** (cosine distance / vector delta), refines it at pixel level into a clean mask, classifies each changed region (construction / clearance / water / vegetation), then suppresses false alarms via cloud-shadow masking, radiometric normalisation and confidence scoring.
* **Testing:** Paint a patch on a copy of an image → your mask should flag exactly that patch. Brighten the entire copy → your suppressor should report **no change**. That second test *is* the novelty.

### 👤 Person 4: Evaluation, Integration & Demo (The "Prove It" Factor)
**Goal:** Measure everything honestly, glue the pipeline together, and make the demo look sharp.
* **Code to write:** `metrics.py`, `visualizer.py`, `main.py`
* **What it does:**
  1. **Measurement:** Precision@k and mAP for retrieval; Precision, Recall, F1, IoU for change; false-positive rate under confound stress. Plus system stats (build time, latency, storage).
  2. **Integration:** imports P1's index, P2's retrieval, P3's change analysis, runs an end-to-end query → retrieve → compare → report flow.
  3. **Demo:** before/after/change-mask panels, ranked-result grids, baseline comparison bar charts.
* **Testing:** Verify metric math on synthetic masks with known overlap before trusting it on real data.

---

## 🚦 The Three Gates (Run These Early — They Are Not Formalities)

```
GATE A — RETRIEVAL WORKS                                    [P2, Week 2]
  5 held-out text queries over a 200-tile index.
  PASS: ≥3 of 5 return a relevant tile in the top-5.
  FAIL: swap generic CLIP → RemoteCLIP; check tile size/resolution.

GATE B — SHARED-SPACE CHANGE BEATS PIXEL DIFFERENCING       [P3, Week 3]
  5 labelled LEVIR-CD pairs. Compare F1:
    (a) pixel differencing + Otsu   (b) our embedding differencing
  PASS: (b) beats (a) on ≥3 of 5.
  FAIL: tune tile size / distance metric BEFORE building further.

GATE C — SUPPRESSION SUPPRESSES, WITHOUT GOING BLIND  ⚠️     [P3, Week 2]
  TWO HALVES, BOTH REQUIRED:
    Half 1 (false alarm): 3 NO-CHANGE pairs w/ seasonal or illumination shift.
        Compare false-positive area: naive differencing vs. ours.
        PASS: ours is lower on ≥2 of 3.
    Half 2 (sensitivity): real-change pairs with ground truth.
        Compare recall with suppression vs. without.
        PASS: worst-case ≥80% of unsuppressed recall retained.
  FAIL either half: THE NOVELTY CLAIM IS UNSUPPORTED. Fix suppression first.
```

> **Gate C is the important one.** It is the experiment that proves the locked novelty sentence. Run it in Week 2 — not Week 6, when there is no time left to fix it.
>
> **Both halves are mandatory.** A suppressor that outputs an empty mask scores a perfect zero false-alarm area and passes half 1 while detecting nothing whatsoever. We hit exactly this while validating the code (`docs/VALIDATION_NOTES.md`). Half 2 is what stops "improve precision" degenerating into "predict nothing".
>
> Run all three gates with: `python gates.py --synthetic`

---

## 🚀 Execution Steps

1. **Setup (Day 1):** Create the empty `.py` files. Everyone claims their files. P1 starts the LEVIR-CD download immediately (it's ~3.8 GB — don't start this the night before a deadline).
2. **Parallel Build (Days 2–7):** Everyone writes and tests their own functions independently using dummy data. P1 gets real tiles + a real index working.
3. **Gate A & Gate C (Week 2):** P2 proves retrieval returns relevant results. P3 proves suppression works. **Stop and fix if either fails.**
4. **Gate B (Week 3):** P3 proves the shared-space change signal beats pixel differencing.
5. **Integration (Week 4):** P4 writes `main.py` and imports everyone's functions. Dummy vectors get swapped for real ones. The chain connects.
6. **Measurement & Tuning (Weeks 5–6):** Run all 8 baselines (4 retrieval + 4 change). Fill the results tables with real numbers. Tune P3's thresholds for **precision**, not recall.
7. **Final Demo (Weeks 7–8):** P4 polishes the output — ranked result grids, before/after/mask panels, baseline bar charts. Pull 2–3 live Sentinel-2/Bhuvan tiles for a qualitative "it works on real official data" moment. Rehearse the demo.

---

## ⚠️ Standing Rules (Do Not Break These)

1. **Nothing is trained.** All models are frozen and used for inference only. Record model name, version, source URL and licence for every weight.
2. **Quantitative numbers come only from LEVIR-CD and OSCD.** Live Sentinel-2/Bhuvan tiles are qualitative figures — they never enter a metrics table.
3. **Precision over recall.** PS §2.2.3 explicitly asks for analytically useful precision rather than indiscriminate change recall. Tune accordingly and say so.
4. **Declare the scope.** We build PS capabilities 2.2.1 and 2.2.2 fully, 2.2.3 partially, 2.2.4 as a stretch. Capabilities 2.2.5 (analyst workflow/audit trail) and 2.2.6 (offline/air-gapped/incremental indexing) are **out of scope and stated as future work.** Never imply otherwise.
5. **Cache the model weights locally, early.** Do not discover on demo day that a download is needed.

---

*BCSE407L Computer Vision RIL | Execution Plan | Aligned to SIH26227 | Last Updated: 15 September 2026*
