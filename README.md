# Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery
### Using a Shared Foundation-Model Embedding Space

**BCSE407L – Computer Vision | Research Integrated Learning (RIL)**
Aligned to **SIH 2026 · PS SIH26227** · Ministry of Defence (Indian Army, DGIS) · Theme: Space Technology

Akanksha Bharti (23BCE0257) · Arjun Pratap Singh (23BCE2027) · Gontla Venkata Siva Sai Nikhil (23BCE0318) · Subhajit Masanta (23BCE0292)

---

## The idea in one paragraph

Satellite archives can be searched by **metadata** (place, date, sensor) but not by
**what is in the picture**. We cut images into tiles and embed every tile **once**
with a frozen pretrained model (OpenCLIP / RemoteCLIP / DINOv2). Those vectors go in
one shared index. The same index does **two jobs**: it ranks tiles against a text or
example-image query, and the distance between a tile's vectors at two dates gives a
change signal. **Nothing is trained; inference only.**

> *"We couple semantic retrieval and multi-temporal change analysis in a single
> shared foundation-model embedding space — the same vectors that rank a text query
> also produce the change signal — and we evaluate change decisions by precision
> under confound stress (season, illumination, cloud, co-registration error) rather
> than by raw change recall."*

## Results so far (real runs, frozen models, GTX 1650 Ti)

All numbers are in `results/*.json`. Choices (thresholds, prompts, model weights) were
made on train/tune splits and reported on held-out data.

| Tested | Result | Verdict |
|---|---|---|
| Same-place image search (OSCD, 2,139 tiles) | 46% top-1 (random 0.05%) | Works |
| Text search, EuroSAT (27,000 patches, human labels) | mAP 0.44, P@10 69% (random 0.10) | Works |
| EuroSAT prompt rewording for 2 weak classes (tuned on 50%, scored on the other 50%) | OpenCLIP mAP 0.438 → 0.444; RemoteCLIP 0.306 → 0.336 | Small gain (mostly SeaLake) |
| Text search, OSCD (formula tags) | mAP 0.16 (random 0.12) | Barely |
| Tile-level change from embeddings, **LEVIR-CD 0.5 m** | AUC 0.82 whole tile, **0.92** patch features (pixel diff 0.70) | Works |
| Text-query change on **LEVIR-CD** (score = rise in similarity to "dense residential area with roofs", prompt picked on val) | test AUC **0.946**, top 50 all correct (OpenCLIP) | Works when the change type is known |
| Tile-level change, **SYSU-CD 0.5 m** (Hong Kong, mixed change types, 4,000 test pairs) | AUC 0.76 patch features (RemoteCLIP), pixel diff 0.70; text query 0.59 | Works, smaller margin |
| Pixel-exact maps, **SYSU-CD** | patch heat map F1 **0.58** vs pixel diff 0.43, CVA 0.16 | Best of the methods tried |
| Tile-level change from embeddings, **OSCD 10 m** | AUC 0.56 whole tile, 0.70 patch features (pixel diff 0.64) | Weak |
| Ensembling CLIP / RemoteCLIP / DINOv2 patch scores (OSCD) | test AUC 0.70–0.74 for every combination | No reliable gain |
| Pixel-exact change maps | best F1 0.35 OSCD (CVA baseline), 0.37 LEVIR-CD | Weak |
| Brightness matching | false alarms 24.4% → 8.8% (about 3×), 90% of recall kept | Works |
| Cloud mask / speck removal | cloud mask flags bright roofs as cloud; speck removal cuts recall 0.52 → 0.34 | Hurt, off by default |
| Index-based change types (OSCD) | best F1 0.19 | Rough guide only |
| Multi-year lasting-change filter (10 Indian sites, 2019–2025) | stricter than the two-date map; **no ground truth, so no accuracy claimed** | Qualitative |

**Honest summary:** the shared-embedding idea holds for search and for tile-level change
on sharp imagery. It is weak at 10 m (the resolution of Sentinel-2, the PS's named
source) and does not give exact change shapes. Hard OSCD cities (Valencia, Saclay,
Norcia) are reported as hard cases, not tuned away.

## Run it

### Quick check (no datasets)
```bash
pip install -r requirements.txt
python gates.py --synthetic     # three validation gates, all PASS
python main.py  --demo          # end-to-end pipeline, writes figures to outputs/
```
> The embedder falls back to a hashing stub when no model weights exist. The stub is
> **not a learned model**. Never report numbers made with it.

### Demo app (Streamlit)
```bash
python demo/build_index.py                       # once; builds demo_index/ (needs data/)
streamlit run demo/app.py --global.developmentMode false
```
Windows: double-click `run_demo.bat` (uses the project-local `env/` libraries, if you have them).
Tabs: map of all data · text search · check the search (EuroSAT, results marked right/wrong) ·
image search · change between two dates · search by GPS · change over years (India) ·
discover clusters · about. Sidebar filters: resolution, collection, country, city.

Local runs use every image in `data/`. The deployed site has no `data/`, so it reads the
small copies in `demo_data/` (6 OSCD cities, 200 EuroSAT patches per class), made by
`python demo/make_web_data.py`.

Best demo: **Change over years → Jewar airport**, slide 2019 → 2025.

### Experiments
`run.bat <script>` (or `python <script>` with libraries installed):
`experiments/run_oscd.py`, `extra_analysis.py`, `run_oscd_v2.py`, `run_eurosat.py`,
`run_levir.py`, `run_levir_text.py`, `run_sysu.py`, `run_indices.py`, `run_series.py`,
`make_figures.py`.

## Data

Not in the repo (see `.gitignore`). `python download_data.py` prints sources and `--check` verifies them.

| Dataset | Used for | Link |
|---|---|---|
| OSCD (24 Sentinel-2 pairs, 10 m) | change + search tests | https://rcdaudt.github.io/oscd/ · mirror https://huggingface.co/datasets/blanchon/OSCD_MSI |
| EuroSAT (27,000 patches) | text-search scoring | https://github.com/phelber/EuroSAT |
| LEVIR-CD (0.5 m building change) | change where it works best | https://justchenhao.github.io/LEVIR/ |
| SYSU-CD (Hong Kong 2007 vs 2014, 0.5 m, mixed change) | change test, demo | https://huggingface.co/datasets/ericyu/SYSU_CD (val + test parquet only) |
| Indian Sentinel-2 L2A (10 sites + Vellore, VIT, 2019–2025) | qualitative demo | read by `demo/fetch_india.py` and `demo/fetch_series.py` from the AWS open-data mirror via https://earth-search.aws.element84.com/v1/search (no account) |
| ISRO Bhoonidhi, Resourcesat-2/2A LISS-IV (5 m), VIT pair 21 Mar 2023 vs 4 Jan 2026 (path 101, sub-scene D) | qualitative demo | https://bhoonidhi.nrsc.gov.in/ (free account; `demo/fetch_bhoonidhi.py` uses the API) |

`demo/fetch_bhoonidhi.py` reads your login from the environment variables
`BHOONIDHI_USER` and `BHOONIDHI_PASS`. **Never commit credentials.** Scenes marked
"Online = N" cannot be fetched by the API and must come from the web cart.

**Metric rule:** numbers come only from datasets with ground truth (OSCD, EuroSAT,
LEVIR-CD). Indian Sentinel-2 and ISRO imagery are qualitative only.

## Repository layout

```
├── src/                 the pipeline (one file per job)
│   ├── ingest.py            GeoTIFF/COG read, tiling, geo-metadata
│   ├── embedder.py          frozen encoder (RemoteCLIP -> OpenCLIP -> stub)
│   ├── index_builder.py     the shared index (FAISS or NumPy)
│   ├── retrieval.py         text and image search
│   ├── filters.py           area / date / sensor filters
│   ├── clustering.py        site discovery (shown in the demo, no metric)
│   ├── change_detect.py     embedding change, pixel baselines, suppression
│   ├── change_classify.py   change typing (rule based)
│   ├── change_types.py      spectral-index change types
│   ├── time_series.py       multi-year lasting-change filter
│   ├── suppress.py          brightness matching, cloud/shadow mask
│   ├── metrics.py           P@k, mAP, precision, recall, F1, IoU, AUC
│   └── visualizer.py        figures
├── demo/                Streamlit app, index builder, data fetchers
├── experiments/         every experiment that produced a number
├── results/             saved numbers (*.json) and logs
├── gates.py, main.py    synthetic validation gates and end-to-end runner
├── download_data.py     dataset sources and checks
├── report_tools/        scripts that build the course report (.docx)
├── docs/VALIDATION_NOTES.md   bugs found by running the code
├── PROJECT_PLAN.md, TEAM_PLAN.md, EXECUTION_PLAN.md   original Sept plans (see banner)
├── TEAM_NOTES.md        plain-language notes for the report
└── archive_old_idea/    the abandoned image-enhancement project
```

## Scope (PS SIH26227 has six capabilities)

| PS § | Capability | What we did |
|---|---|---|
| 2.2.1 | Semantic + image retrieval | Full, evaluated |
| 2.2.2 | Change analysis + typing | Done; strong on sharp imagery, weak at 10 m; typing is a rough guide |
| 2.2.3 | False-alarm suppression | Partial: brightness matching works; cloud mask and speck removal tested and left off |
| 2.2.4 | Discovery / clustering | Implemented, shown qualitatively in the demo, no metric |
| 2.2.5 | Analyst workflow, audit trail | Not built |
| 2.2.6 | Offline / incremental indexing | Not built |

## What we do not claim

| Not this | This |
|---|---|
| We built the Army's intelligence system | A research prototype of the core CV capabilities of PS SIH26227 |
| We invented a new detector | Frozen pretrained encoders; the contribution is the shared-space coupling and the honest evaluation |
| Change detection works on Sentinel-2 | It works on sharp imagery; at 10 m it is weak and pixel-exact maps do not beat the classic CVA baseline |
| Our multi-year filter is more accurate | It is stricter; with no ground truth we cannot say it is more accurate |
| Ensembles or prompt tuning gave big gains | Gains were small; reported as measured |

---
*Replaces the earlier "Adaptive Image Quality Assessment and Enhancement for Long-Range
Optical Surveillance" project, which clashed with BCSE407J010. Kept in `archive_old_idea/`.*
