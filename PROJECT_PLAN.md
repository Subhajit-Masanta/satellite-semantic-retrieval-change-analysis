> **Status note (Oct 2026):** this is the original September plan. The project has since been run on real data and some claims changed (change detection is weak at 10 m, cloud mask and speck removal were left off, clustering is demo-only). For current results and scope read `README.md` and `TEAM_NOTES.md`.

# 🛰️ Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery
### Using a Shared Foundation-Model Embedding Space
#### BCSE407L – Computer Vision | Research Integrated Learning (RIL)
#### Aligned to SIH 2026 · PS **SIH26227** · Ministry of Defence (Indian Army, DGIS) · Theme: Space Technology

---

## 📝 Project Title

> **Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery Using a Shared Foundation-Model Embedding Space**

---

## 🎯 One-Line Goal

Make a satellite-image archive searchable **by meaning** ("newly built structures near a river") and **by change over time**, using one shared embedding space for both tasks — and prove the change decisions are *trustworthy* by measuring precision under seasonal/illumination/cloud confounds, not just raw change recall.

---

## 🔬 Research Question

> *"Can a single pretrained remote-sensing embedding space serve both semantic retrieval and multi-temporal change detection, and does confound-aware suppression in that shared space produce higher-precision change decisions than pixel-domain differencing baselines?"*

---

## 💡 Locked Novelty Sentence

> **All team members must use this exact framing in every section they write. Do not drift from it.**

*"We couple semantic retrieval and multi-temporal change analysis in a single shared foundation-model embedding space — the same vectors that rank a text query also produce the change signal — and we evaluate change decisions by precision under confound stress (season, illumination, cloud, co-registration error) rather than by raw change recall."*

**What this means in plain words:**
- We embed every image tile **once**. Those vectors do double duty: they answer search queries *and* they detect change.
- We do **not** report "we found lots of changes." We report "the changes we reported were real."
- A seasonal colour shift or a cloud shadow is a **confound to suppress**, not a change to celebrate.

**What this is NOT:**
- Not "we invented a new change-detection network" — we use pretrained models, inference only.
- Not "we built a new retrieval algorithm" — CLIP-style retrieval exists.
- The novelty is: **one embedding space for both tasks**, plus **precision-under-confound as the evaluation target**.

---

## ⚠️ Scope Declaration — What We Build vs. What We Don't

The official PS SIH26227 lists **six** required capabilities. A 4-member, 8-week RIL project cannot deliver an air-gapped production system. We implement the **core computer-vision capabilities** and explicitly declare the operational/infrastructure ones out of scope. **State this in the paper's Section 4 and say it out loud in the review.**

| PS § | Capability | Our scope |
|---|---|---|
| 2.2.1 | Semantic + image-to-image retrieval | ✅ **Full build** — free-text query, image-to-image search, ranked results, metadata filters |
| 2.2.2 | Multi-temporal change analysis + change typing | ✅ **Full build** — change mask + classification into construction / clearance / water / vegetation |
| 2.2.3 | False-alarm suppression & quality handling | ⚠️ **Partial** — cloud/shadow masking, radiometric normalisation, confidence scores. We handle a *subset* of confounds and say which. |
| 2.2.4 | Discovery & clustering | ⚠️ **Stretch goal** — k-means / nearest-neighbour grouping over the existing index (cheap to add if time allows) |
| 2.2.5 | Analyst workflow, provenance, audit trail | ❌ **Out of scope** — documented as future work |
| 2.2.6 | Offline/air-gapped deployment, incremental indexing, COG ingestion at scale | ❌ **Out of scope** — we use a static FAISS index in Colab. GeoTIFF read support is demonstrated, not productionised. |

> This is a **strength, not a weakness.** Declaring scope honestly is what separates a research prototype from an overclaim.

---

## 🏗️ What the System Does (Pipeline)

```
🛰️ SATELLITE IMAGE ARCHIVE  (GeoTIFF / COG tile pairs, t1 and t2)
                │
                ▼
┌───────────────────────────────────────────┐
│  STAGE 1: INGESTION & TILING              │
│  • Read GeoTIFF/COG (rasterio)            │
│  • Tile into fixed patches                │
│  • Preserve geo-metadata + provenance     │
│    (source scene, CRS, bounds, date)      │
└──────────────────┬────────────────────────┘
                   │
┌──────────────────▼────────────────────────┐
│  STAGE 2: EMBEDDING & INDEXING            │
│  • RemoteCLIP / CLIP encoder (FROZEN)     │
│  • One vector per tile per date           │
│  • FAISS index over all vectors           │
│  ← THIS INDEX IS SHARED BY STAGE 3 AND 4  │
└──────────────────┬────────────────────────┘
                   │
      ┌────────────┴────────────┐
      │                         │
┌─────▼──────────────┐  ┌───────▼─────────────────────┐
│ STAGE 3: SEMANTIC  │  │ STAGE 4: CHANGE ANALYSIS    │
│         RETRIEVAL  │  │                             │
│ • Free-text query  │  │ • Δ = f(v_t1, v_t2) in the  │
│   → text embedding │  │   SAME embedding space      │
│ • Image-to-image   │  │ • Pixel-level refinement     │
│ • Cosine ranking   │  │ • Change mask                │
│ • Metadata filters │  │ • Change TYPE classification │
│   (AOI/date/sensor)│  │   (construction/clearance/   │
│ • Top-k results    │  │    water/vegetation)        │
└─────┬──────────────┘  └───────┬─────────────────────┘
      │                         │
      └────────────┬────────────┘
                   │
┌──────────────────▼────────────────────────┐
│  STAGE 5: FALSE-ALARM SUPPRESSION         │
│  • Cloud / shadow / snow masking          │
│  • Radiometric normalisation across dates │
│  • Confidence score per candidate         │
│  • Precision-first thresholding           │
└──────────────────┬────────────────────────┘
                   │
┌──────────────────▼────────────────────────┐
│  STAGE 6: EVALUATION & VISUALISATION      │
│  • Retrieval: Precision@k, mAP            │
│  • Change: Precision, Recall, F1, IoU     │
│  • Confound stress test (FP rate)         │
│  • Index build time, query latency, size  │
└──────────────────┬────────────────────────┘
                   │
                   ▼
        📊 RANKED RESULTS + CHANGE MAPS + METRICS
```

---

## 📊 What We Compare

### Retrieval baselines
| Method | Type | Description |
|---|---|---|
| Random ranking | Baseline | Sanity floor |
| Metadata-only filter | Baseline | What a conventional catalogue already does (date/AOI/sensor) |
| Generic CLIP (ViT-B/32) | Baseline | Off-the-shelf, natural-image domain |
| **RemoteCLIP (ours)** | **Proposed** | **Remote-sensing-tuned embeddings, same pipeline** |

### Change-detection baselines
| Method | Type | Description |
|---|---|---|
| Pixel differencing + Otsu | Baseline | Classical, no features |
| Change Vector Analysis (CVA) | Baseline | Classical multi-band standard |
| SSIM-based differencing | Baseline | Structural, reuses classical CV |
| **Embedding differencing (ours)** | **Proposed** | **Shared-space feature delta + confound suppression** |

**Headline metrics:** retrieval **Precision@5**, change-detection **F1** and **false-positive rate under confound stress**.

---

## 📁 Datasets

| Dataset | Size | Labels? | What We Use It For |
|---|---|---|---|
| **LEVIR-CD** | 637 pairs, 1024×1024, 0.5 m | ✅ Pixel-level change masks | **Primary** — change F1/IoU, retrieval index, all headline numbers |
| **OSCD (Onera)** | 24 pairs, real Sentinel-2 multispectral | ✅ Pixel-level change masks | **Secondary** — proves it works on *genuine* Sentinel-2, not just aerial crops |
| **Live Sentinel-2 / Bhuvan tiles** | 2–3 hand-picked AOI pairs | ❌ No labels | **Qualitative demo only** — shows we ingest the PS's own official sources |

> **Rule:** All quantitative numbers (F1, IoU, Precision@k) come from **LEVIR-CD and OSCD only**. Live Sentinel-2/Bhuvan tiles are shown as **qualitative figures** and never appear in a metrics table. State this explicitly in Section 4. *(Same discipline as the old project's PSNR/SSIM rule.)*

**Download sources:**
- LEVIR-CD — https://justchenhao.github.io/LEVIR/ (Google Drive / Kaggle mirrors)
- OSCD — https://rcdaudt.github.io/oscd/ · HuggingFace: `blanchon/OSCD_MSI`
- Copernicus Sentinel-1/2 — https://dataspace.copernicus.eu/ (free account + token)
- USGS Landsat Collection 2 — https://earthexplorer.usgs.gov/ (free account)
- NRSC/ISRO Bhuvan — https://bhuvan-app3.nrsc.gov.in/data/download/ · https://bhoonidhi.nrsc.gov.in/

---

## 📏 Metrics

| Metric | Applies To | Direction | What It Proves |
|---|---|---|---|
| **Precision@k (k=1,5,10)** | Retrieval | ↑ Higher = better | Top results are actually relevant |
| **mAP** | Retrieval | ↑ Higher = better | Ranking quality overall |
| **F1 Score** | Change detection | ↑ Higher = better | Balance of catching change vs. crying wolf |
| **IoU** | Change detection | ↑ Higher = better | Change mask overlaps ground truth spatially |
| **Precision** | Change detection | ↑ Higher = better | **Primary** — PS §2.2.3 demands precision over recall |
| **False-Positive Rate under confound stress** | Change detection | ↓ Lower = better | **Novelty metric** — seasonal/illumination shifts are NOT reported as change |
| **Index build time / query latency / storage** | System | ↓ Lower = better | PS explicitly requires these in the eval report |

---

## ⚠️ What We Will NOT Claim

| ❌ Do NOT Say This | ✅ Say This Instead |
|---|---|
| "We built the Army's geospatial intelligence system" | "We implement the core CV capabilities of PS SIH26227 as a research prototype" |
| "We invented a new change-detection model" | "We use frozen pretrained encoders; our contribution is the shared-space coupling and evaluation protocol" |
| "Our system works fully offline on-premises" | "Offline/air-gapped deployment (PS §2.2.6) is out of scope and noted as future work" |
| "We suppress all false alarms" | "We suppress a stated subset of confounds: cloud, shadow, illumination, radiometric drift" |
| "We tested on real Army imagery" | "We used only public benchmark and open Copernicus/Bhuvan data" |
| "Higher change recall proves it works" | "We optimise precision; recall trade-offs are reported honestly" |

---

## ✅ Non-Negotiable Pilot Gate (Run Before Anything Else)

**Before any further build, run all three gates. This is a gate, not a formality.**

```
GATE A — RETRIEVAL WORKS
  5 held-out text queries over a 200-tile LEVIR-CD index.
  PASS: ≥3 of 5 queries return a relevant tile in the top-5.
  FAIL: swap generic CLIP → RemoteCLIP, re-check tile size/resolution.

GATE B — SHARED-SPACE CHANGE BEATS PIXEL DIFFERENCING
  5 labelled LEVIR-CD pairs. Compute F1 for:
    (a) pixel differencing + Otsu
    (b) our embedding differencing
  PASS: (b) beats (a) on ≥3 of 5 pairs.
  FAIL: tune tile size / distance metric BEFORE building further.

GATE C — CONFOUND SUPPRESSION SUPPRESSES, WITHOUT GOING BLIND  ← the novelty gate
  TWO HALVES, BOTH REQUIRED:

  Half 1 (false alarm):  3 NO-CHANGE pairs with seasonal/illumination shifts.
      Measure false-positive area, naive differencing vs. ours.
      PASS: our false-positive area is lower on ≥2 of 3.

  Half 2 (sensitivity):  real-change pairs with ground truth.
      Measure recall with suppression vs. recall without it.
      PASS: worst-case ≥80% of the unsuppressed recall is retained.

  FAIL either half: the novelty claim is unsupported. Fix before proceeding.
```

> Gate C is the one that validates the **novelty sentence**. If it fails, the paper has no contribution. Run it early.
>
> **Why two halves.** With only the false-alarm half, the gate is gameable: a suppressor that returns an empty mask scores a perfect zero false-alarm area and "passes" while detecting nothing at all. We found this by actually running it — see `docs/VALIDATION_NOTES.md`. A precision-oriented metric must always be paired with a sensitivity floor, or "improving precision" quietly collapses into "predict nothing".

---

## 🧠 Model Constraint (Important from Day 1)

We use **frozen, pretrained, inference-only** models. No training, no fine-tuning.

- Declare model **origin and licence** for every weight used (the PS requires this, and it's good practice regardless).
- Prefer **RemoteCLIP** or **DINOv2** over generic CLIP — generic CLIP was trained on natural photos and has a real domain gap on overhead imagery.
- Cache weights locally early. Don't discover on demo day that a download is required.

---

## 📰 Paper Structure

| Section | Content |
|---|---|
| **1. Introduction** | Problem (archives searchable by metadata, not meaning) + why it matters + 3 contributions (state the locked novelty sentence here) |
| **2. Related Work** | 6–8 papers: CLIP, RemoteCLIP/remote-sensing foundation models, siamese change detection, CVA, FAISS/vector search, recent 2024–2026 EO retrieval |
| **3. Methodology** | Pipeline diagram + embedding formulation + change signal derivation + confound suppression rules |
| **4. Experimental Setup** | Datasets (with the quantitative-vs-qualitative rule stated) + metrics + baselines + **scope declaration table** + hardware |
| **5. Results & Discussion** | Retrieval table + change table + confound stress test + qualitative figures (before/after/mask) |
| **6. Ablation Study** | Remove confound suppression; swap RemoteCLIP→CLIP; vary tile size — prove each choice matters |
| **7. Conclusion** | Summary + limitations (scope §2.2.5/2.2.6 not built, subset of confounds) + future work |

---

## 🎯 Key Papers to Cite

| Paper | Year | Why Cite It |
|---|---|---|
| Radford et al. — *CLIP* | 2021 | Your retrieval backbone / text-image alignment |
| Liu et al. — *RemoteCLIP* | 2024 | Remote-sensing-tuned embeddings; closes the domain gap |
| Oquab et al. — *DINOv2* | 2023 | Alternative frozen visual encoder |
| Daudt et al. — *Fully Convolutional Siamese Networks for Change Detection* (OSCD) | 2018 | Your change-detection baseline + dataset |
| Chen & Shi — *STANet / LEVIR-CD* | 2020 | Your primary dataset + benchmark numbers |
| Malila — *Change Vector Analysis* | 1980 | Classical change baseline |
| Johnson et al. — *FAISS* | 2019 | Vector indexing / scalable similarity search |
| 2–3 recent (2024–2026) papers on EO foundation models / semantic EO retrieval | — | Show the current gap |

---

## 🛠️ Tools

- Python 3.10+
- **PyTorch** (inference only) + `open_clip` / HuggingFace `transformers`
- **FAISS** — vector index and similarity search
- **rasterio** / GDAL — GeoTIFF & COG reading, geo-metadata preservation
- NumPy, OpenCV, scikit-image (SSIM, morphology), scikit-learn (k-means, metrics)
- Matplotlib — figures, before/after/mask panels
- *(Optional)* Streamlit or Gradio — query UI for the live demo
- Google Colab — free tier is sufficient; GPU optional, CPU workable

> ⚠️ **Change from the previous project:** we now use deep-learning models. But **inference only — nothing is trained.** Frame it as "training-free use of pretrained foundation models," not "we avoided deep learning."

---

## 📌 Pivot Note

This project replaces the earlier *"Adaptive Image Quality Assessment and Enhancement for Long-Range Optical Surveillance"* idea, which clashed with **BCSE407J010 — Long Range Optical Target Locator**. All previous work is preserved in `archive_old_idea/`. Reusable carry-over: SSIM/metrics code, visualisation and side-by-side plotting, and the evaluation-discipline mindset (locked novelty sentence, scope honesty, pilot gates).

---

*BCSE407L Computer Vision RIL | Aligned to SIH26227 | Last Updated: 15 September 2026*
