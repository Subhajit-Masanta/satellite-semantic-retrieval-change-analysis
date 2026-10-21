> **Status note (Oct 2026):** this is the original September plan. The project has since been run on real data and some claims changed (change detection is weak at 10 m, cloud mask and speck removal were left off, clustering is demo-only). For current results and scope read `README.md` and `TEAM_NOTES.md`.

# 🛰️ BCSE407L Computer Vision — RIL Team Plan
### Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery
#### Using a Shared Foundation-Model Embedding Space
#### 4-Member Team | Aligned to SIH26227 (MoD / Indian Army DGIS)

---

## 📝 Project Title

> **Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery Using a Shared Foundation-Model Embedding Space**

---

## 🎯 Project Goal (One Line)

Make a satellite archive searchable by meaning *and* by change over time from a single shared embedding space, and prove the change decisions are trustworthy by measuring precision under seasonal/illumination/cloud confounds.

---

## 🔬 Research Question

> *"Can a single pretrained remote-sensing embedding space serve both semantic retrieval and multi-temporal change detection, and does confound-aware suppression in that shared space produce higher-precision change decisions than pixel-domain differencing baselines?"*

---

## 💡 Locked Novelty (Everyone Must Use This)

> *"We couple semantic retrieval and multi-temporal change analysis in a single shared foundation-model embedding space — the same vectors that rank a text query also produce the change signal — and we evaluate change decisions by precision under confound stress (season, illumination, cloud, co-registration error) rather than by raw change recall."*

| ✅ What We Claim | ❌ What We Do NOT Claim |
|---|---|
| One embedding space serves both tasks | We invented a new detection network |
| Precision-first change evaluation | We built the Army's full system |
| Confound suppression on a stated subset | We suppress every possible false alarm |
| Training-free use of pretrained models | We trained or fine-tuned anything |
| Honest comparison against 4 baselines | Offline/air-gapped deployment is solved |

---

## ⚠️ Scope Declaration (Memorise This for the Review)

PS SIH26227 asks for **six** capabilities. We build the CV core and declare the rest out of scope.

| PS § | Capability | Our scope | Owner |
|---|---|---|---|
| 2.2.1 | Semantic + image-to-image retrieval | ✅ Full build | **Person B** |
| 2.2.2 | Change analysis + change typing | ✅ Full build | **Person C** |
| 2.2.3 | False-alarm suppression | ⚠️ Partial (stated subset) | **Person C** |
| 2.2.4 | Discovery / clustering | ⚠️ Stretch goal | **Person B** |
| 2.2.5 | Analyst workflow / audit trail | ❌ Future work | — |
| 2.2.6 | Offline / incremental / production indexing | ❌ Future work | — |

---

## ⚡ How to Work in Parallel (NO CHAIN DEPENDENCY)

You do **NOT** wait for the previous person. Everyone works independently from Day 1.

* **Person A (Data & Index):** Download LEVIR-CD, tile it, and build the embedding index. Test by embedding 50 tiles and confirming vectors come out the right shape. Nobody blocks you.
* **Person B (Retrieval):** Don't wait for A's real index. Generate **random dummy vectors** (e.g. `np.random.rand(200, 512)`), build a FAISS index on those, and verify your search/ranking/filter code returns *k* results in ranked order. Swap in A's real vectors later — same interface.
* **Person C (Change + Suppression):** Don't wait for A or B. Take **any two images** (even a photo and a manually edited copy), run your differencing + masking + confidence code, and verify it produces a sensible mask. Swap in real embeddings later.
* **Person D (Evaluation & Paper):** Start immediately. Write Section 1 (Intro) and Section 2 (Related Work) — no code needed. Build metric functions and test them on **synthetic masks** (a hand-drawn ground truth vs. a hand-drawn prediction) to verify F1/IoU math is right.

**Integration happens in Week 4**, where A's index feeds B and C, and D measures everything.

---

## 🏗️ System Pipeline & Ownership

```
🛰️ GeoTIFF / COG TILE PAIRS (t1, t2)
                │
                ▼
┌───────────────────────────────────────────┐
│ STAGE 1: INGESTION & TILING               │ ← Person A
│ • rasterio read, tiling                    │
│ • geo-metadata + provenance preserved      │
└──────────────────┬────────────────────────┘
                   │
┌──────────────────▼────────────────────────┐
│ STAGE 2: EMBEDDING & INDEXING             │ ← Person A
│ • Frozen RemoteCLIP / CLIP encoder         │
│ • FAISS index (SHARED by Stages 3 and 4)   │
└──────────────────┬────────────────────────┘
                   │
      ┌────────────┴────────────┐
┌─────▼──────────────┐  ┌───────▼─────────────────────┐
│ STAGE 3: RETRIEVAL │  │ STAGE 4: CHANGE ANALYSIS    │
│ ← Person B         │  │ ← Person C                  │
│ • text → embedding │  │ • Δ in the SAME space       │
│ • image-to-image   │  │ • pixel refinement          │
│ • cosine ranking   │  │ • change mask               │
│ • metadata filters │  │ • change TYPE classification│
│ • clustering (⚠️)   │  │                             │
└─────┬──────────────┘  └───────┬─────────────────────┘
      │                         │
      │                 ┌───────▼─────────────────────┐
      │                 │ STAGE 5: FALSE-ALARM         │
      │                 │   SUPPRESSION ← Person C     │
      │                 │ • cloud/shadow/snow mask     │
      │                 │ • radiometric normalisation  │
      │                 │ • confidence scoring         │
      │                 └───────┬─────────────────────┘
      └────────────┬────────────┘
                   │
┌──────────────────▼────────────────────────┐
│ STAGE 6: EVALUATION & VISUALISATION       │ ← Person D
│ • Precision@k, mAP | F1, IoU, Precision    │
│ • Confound stress test (FP rate)           │
│ • Build time, latency, storage             │
└──────────────────┬────────────────────────┘
                   │
                   ▼
        📊 RESULTS + PAPER   ← Person D (All Writing)
```

---

## 👥 Team Work Division

---

### 👤 Person A — Data Pipeline, Embedding & Index

**You own Stages 1 and 2. Everyone else consumes what you produce.**

**What you build:**
* `ingest.py` (~60 lines): read GeoTIFF/COG with `rasterio`, tile into fixed patches, keep geo-metadata (CRS, bounds, acquisition date, source scene) alongside every tile.
* `embedder.py` (~50 lines): load a **frozen** RemoteCLIP/CLIP encoder, turn each tile into one vector. No training.
* `index_builder.py` (~40 lines): build and save a FAISS index + a parallel metadata table.

**Parallel Work Strategy:** Download LEVIR-CD on Day 1 and get tiles + vectors flowing. Print vector shapes and index size. Do not wait for anyone.

**Your deliverable — DAY 2-3:** 200 LEVIR-CD tiles embedded, FAISS index built and saved to disk, and you can report index build time + storage footprint (the PS explicitly asks for these numbers).

**⚠️ Model constraint from Day 1:** cache model weights locally and **record the model name, version, source URL, and licence** for every weight — the PS requires declared provenance, and Person D needs it for the paper.

*(No paper writing. Just hand Person D a text file with model provenance, index stats, and tile counts.)*

---

### 👤 Person B — Semantic Retrieval & Discovery

**You own Stage 3. You make the archive searchable by meaning.**

**What you build:**
* `retrieval.py` (~60 lines): free-text query → text embedding → cosine similarity → ranked top-*k* results.
* `image_search.py` (~30 lines): image-to-image search (given a tile, find visually/semantically similar tiles).
* `filters.py` (~30 lines): apply AOI / date-range / sensor metadata filters on top of semantic ranking.
* *(Stretch, PS §2.2.4)* `clustering.py` (~30 lines): k-means or nearest-neighbour grouping so an analyst who finds one site discovers similar ones.

**Parallel Work Strategy:** Do **not** wait for Person A. Build a FAISS index over `np.random.rand(200, 512)` dummy vectors and confirm your ranking, filtering and top-*k* logic all work. Swapping in A's real vectors is a one-line change.

**Your test queries (write these down early):** *"newly built structures near a river"*, *"large vehicle concentrations on open ground"*, *"dense urban rooftops"*, *"bare cleared land"*, *"water body"*. These become Gate A and your Section 5 table.

*(No paper writing. Hand Person D your query list, ranked-result screenshots, and retrieval timings.)*

---

### 👤 Person C — Change Analysis & False-Alarm Suppression

**You own Stages 4 and 5. This is where the novelty lives — take Gate C seriously.**

**What you build:**
* `change_detect.py` (~70 lines): compute the change signal from **the same embeddings Person A produced** (cosine distance / vector delta per tile), plus pixel-level refinement for a cleaner mask.
* `change_classify.py` (~40 lines): label each changed region as construction / clearance / water-extent / vegetation (rules on colour, texture and shape are fine — say so honestly).
* `suppress.py` (~60 lines): cloud/shadow/snow masking, radiometric normalisation across dates, and a per-candidate confidence score. **Precision first** — PS §2.2.3 explicitly prefers useful precision over indiscriminate recall.

**Parallel Work Strategy:** Do **not** wait for A or B. Take any image, make a copy with a patch painted over it, and verify your pipeline flags exactly that patch. Then test suppression by brightening the whole second image (a fake "illumination change") — a good suppressor reports *no change*.

**⚠️ Your novelty gate (Gate C) — two halves, both required:**
1. **False alarm:** on 3 **no-change** pairs with seasonal/illumination differences, your false-positive area must be lower than naive pixel differencing on ≥2 of 3.
2. **Sensitivity:** on real-change pairs with ground truth, suppression must retain ≥80% of the unsuppressed recall.

**If either half fails, the project has no contribution.** Run it in Week 2, not Week 6. Check with `python gates.py --synthetic`.

> Half 2 exists because half 1 alone is gameable — a suppressor that returns an empty mask has zero false alarms and "passes" while finding nothing. Two real bugs of this shape were already found and fixed in `suppress.py`; read `docs/VALIDATION_NOTES.md` before you touch that file, especially the rule that **cloud masking runs on the raw images, never the radiometrically normalised one.**

*(No paper writing. Hand Person D your masks, confound-test numbers, and the list of which confounds you actually handle.)*

---

### 👤 Person D — Evaluation, Visualisation & Paper

**You own Stage 6 and 100% of the paper and presentation.**

**What you build:**
* `metrics.py` (~50 lines): Precision@k, mAP (retrieval); Precision, Recall, F1, IoU (change); false-positive rate under confound stress.
* `visualizer.py` (~60 lines): before/after/mask panels, ranked-result grids, metric bar charts.
* The **entire paper** (Sections 1–7) and the final slide deck. A, B and C write no paper text — they give you numbers, figures and formulas.

**Parallel Work Strategy:** Start Week 1. Write Sections 1 and 2 with zero code. Test your metric functions on **synthetic masks** you draw by hand (a 10×10 array with a known overlap) so you can verify F1/IoU by hand-arithmetic before real data arrives.

**⚠️ Critical metric rule:**
> Quantitative numbers (F1, IoU, Precision@k) come **only** from LEVIR-CD and OSCD. Live Sentinel-2/Bhuvan tiles are **qualitative figures only** and never enter a metrics table.

**Also your job:** the PS requires a reproducible evaluation report stating **indexed area, number of tiles, build time, storage footprint, query latency, and hardware used**. Collect these from A and B and put them in Section 4.

**Paper outline:**
| Section | Content | Who Writes |
|---|---|---|
| 1. Introduction | Problem + motivation + 3 contributions (locked novelty sentence) | **D** |
| 2. Related Work | 6–8 papers, 2–3 lines each | **D** |
| 3. Methodology | Pipeline + embedding math + suppression rules (data from A/B/C) | **D** |
| 4. Experimental Setup | Datasets + metrics + baselines + **scope declaration** + hardware | **D** |
| 5. Results & Discussion | Tables + qualitative figures (data from D's own eval) | **D** |
| 6. Ablation Study | Drop suppression; CLIP vs RemoteCLIP; tile size sweep | **D** |
| 7. Conclusion | Summary + limitations (§2.2.5/2.2.6 out of scope) + future work | **D** |

---

## 📅 8-Week Timeline (Parallel Execution)

| Week | Person A | Person B | Person C | Person D |
|---|---|---|---|---|
| **Week 1** | Download LEVIR-CD + OSCD. Tile 200 images. | Build `retrieval.py` on dummy vectors | Build `change_detect.py` on hand-edited image pairs | Set up paper template. Write Sec 1 (Intro) |
| **Week 2** | Build embedder + FAISS index. **Report build time/size.** | Write 5 test queries. Run **Gate A** | Build `suppress.py`. Run **Gate C** ⚠️ | Build `metrics.py` on synthetic masks. Write Sec 2 |
| **Week 3** | Add OSCD ingestion + GeoTIFF/COG read | Add metadata filters + image-to-image search | Build `change_classify.py`. Run **Gate B** | Build `visualizer.py`. Draft Sec 3 |
| **Week 4** | **INTEGRATION:** hand real index to B and C | **INTEGRATION:** swap dummy → real vectors | **INTEGRATION:** run on real embeddings | Polish Sec 1–3 |
| **Week 5** | Pull 2–3 live Sentinel-2/Bhuvan tiles for demo | Fix ranking bugs; clustering stretch goal | Tune suppression thresholds for precision | Generate all final tables & graphs. Write Sec 4 |
| **Week 6** | Help D with index stats/provenance | Help D with retrieval numbers | Help D with change numbers + confound test | Write Sec 5 & 6 (Results & Ablation) |
| **Week 7** | Review code | Review code | Review code | Write Sec 7 + build presentation |
| **Week 8** | Final polish | Final polish | Final polish | Final polish & demo rehearsal |

---

## 📁 Datasets

| Dataset | Where to Get | What to Use It For |
|---|---|---|
| **LEVIR-CD** | https://justchenhao.github.io/LEVIR/ | **Primary** — all headline F1/IoU/Precision@k numbers |
| **OSCD (Onera)** | https://rcdaudt.github.io/oscd/ · HF `blanchon/OSCD_MSI` | **Secondary** — real Sentinel-2 multispectral validation |
| **Copernicus Sentinel-1/2** | https://dataspace.copernicus.eu/ | Qualitative demo tiles only (free account) |
| **USGS Landsat C2** | https://earthexplorer.usgs.gov/ | Optional qualitative tiles (free account) |
| **NRSC/ISRO Bhuvan** | https://bhuvan-app3.nrsc.gov.in/data/download/ | Qualitative demo — shows Indian EO source coverage |

---

## 📊 Baselines to Compare Against

**Retrieval**
| Method | P@1↑ | P@5↑ | mAP↑ | Latency↓ |
|---|---|---|---|---|
| Random ranking | — | — | — | — |
| Metadata-only filter | — | — | — | — |
| Generic CLIP | — | — | — | — |
| **RemoteCLIP (Ours)** | **—** | **—** | **—** | **—** |

**Change Detection**
| Method | Precision↑ | Recall↑ | F1↑ | IoU↑ | FP under confound↓ |
|---|---|---|---|---|---|
| Pixel differencing + Otsu | — | — | — | — | — |
| Change Vector Analysis | — | — | — | — | — |
| SSIM differencing | — | — | — | — | — |
| **Embedding Δ + suppression (Ours)** | **—** | **—** | **—** | **—** | **—** |

---

## 🛠️ Code Files Summary

| File | Owner | Lines | Purpose |
|---|---|---|---|
| `ingest.py` | A | ~60 | GeoTIFF/COG read, tiling, geo-metadata |
| `embedder.py` | A | ~50 | Frozen encoder → tile vectors |
| `index_builder.py` | A | ~40 | FAISS index + metadata table |
| `retrieval.py` | B | ~60 | Text query → ranked results |
| `image_search.py` | B | ~30 | Image-to-image search |
| `filters.py` | B | ~30 | AOI / date / sensor filtering |
| `clustering.py` | B | ~30 | *(stretch)* Site discovery grouping |
| `change_detect.py` | C | ~70 | Shared-space change signal + mask |
| `change_classify.py` | C | ~40 | Change type labelling |
| `suppress.py` | C | ~60 | Confound masking + confidence |
| `metrics.py` | D | ~50 | P@k, mAP, F1, IoU, FP rate |
| `visualizer.py` | D | ~60 | Figures, panels, charts |
| `main.py` | All | ~40 | End-to-end run (Week 4) |
| **Total** | | **~620 lines** | |

---

## ✅ Submission Checklist

- [ ] Gate A passed (retrieval returns relevant results)
- [ ] Gate B passed (shared-space beats pixel differencing)
- [ ] **Gate C passed — BOTH halves: false alarms cut AND ≥80% recall retained** ← validates the novelty
- [ ] All 4 retrieval + 4 change baselines run and measured
- [ ] Results tables filled with real numbers
- [ ] Quantitative metrics reported **only** on LEVIR-CD / OSCD
- [ ] Scope declaration table in Section 4 (which PS capabilities we built)
- [ ] Model provenance + licences declared for every pretrained weight
- [ ] Index build time, storage, query latency, hardware reported
- [ ] Novelty sentence appears in abstract + Section 1
- [ ] 7-section paper complete (Person D)
- [ ] Code cleaned and commented
- [ ] Presentation slides done
- [ ] Individual contribution statement written (required by RIL)

---

*BCSE407L Computer Vision RIL | 4-Member Team Plan | Aligned to SIH26227 | Last Updated: 15 September 2026*
