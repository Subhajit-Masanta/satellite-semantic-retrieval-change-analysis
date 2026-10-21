# Team notes — read this, then write your report part in your own words

Rough notes only. Not for submission. Every number is from our real runs (files in `results/`). Do not change the numbers; change the words.

---

## What our project is (say this in 3 lines)

- Satellite archives can be searched only by date / place / satellite name, not by what is in the picture.
- We cut images into small tiles, turn each tile into a vector with a ready-made model (OpenCLIP / RemoteCLIP / DINOv2, we train nothing) and keep all of them in one index.
- That one index does two jobs: (1) search tiles by words or by an example tile, (2) check if a place changed between two dates.

## The honest result (know this before the viva)

| What | Result | Worked? |
|---|---|---|
| Search by example tile (OSCD) | right place first 46% of the time out of 2,139 (random 0.05%) | Yes |
| Text search on EuroSAT (27,000 patches, human labels) | mAP 0.44, top-10 correct 69% (random 0.10) | Yes |
| Prompt rewording for SeaLake and Herbaceous (tuned on half, scored on other half) | mAP 0.438 to 0.444 (OpenCLIP), 0.306 to 0.336 (RemoteCLIP) | Small gain |
| Text search on OSCD tiles | mAP 0.16 (random 0.12) | Barely |
| Tile change, LEVIR-CD (0.5 m, sharp) | AUC 0.92 with patch features, pixel difference 0.70 | Yes |
| Tile change, OSCD (10 m) | AUC 0.70 with patch features, pixel difference 0.64 | Weak |
| Ensemble of models on OSCD | test AUC 0.70 to 0.74 for every combination | No gain |
| Exact change shapes | best F1 0.35 (old CVA method beat us), LEVIR-CD 0.37 | Weak |
| Brightness matching | false alarms 24.4% to 8.8%, keeps 90% of real change | Yes |
| Cloud mask, speck removal | cloud mask called bright roofs "cloud"; speck removal cut recall 0.52 to 0.34 | Hurt, off by default |
| Change types from NDVI / NDWI / NDBI | best F1 0.19 | Rough guide only |
| Multi-year (2019 to 2025, 10 Indian sites) | stricter than two-date map; no ground truth so no accuracy | Qualitative |

If ma'am asks "so did it work?" -> "Search yes. Change works on sharp images (AUC 0.92) but is weak at 10 m, and we found out why."

## Why change is weak at 10 m (in our words)

- One vector describes a whole tile (640 m x 640 m). A few new buildings barely move it.
- Only 3.2% of OSCD pixels changed. At 10 m a building is 2-3 pixels.
- The two dates are in different months, so fields and light look different, and the vector reacts to that more than to buildings.
- On LEVIR-CD (0.5 m) buildings are big in the picture, and the same method reaches AUC 0.92.
- Hard OSCD cities: valencia, saclay, norcia. Almost no structural change there; we report them as hard cases and did not tune for them.

## Demo

- Run `run_demo.bat` (or `streamlit run demo/app.py`), open http://localhost:8501.
- Best moment: "Change over years" -> Jewar (Noida) airport -> slide 2019 to 2025: farmland, cleared ground in 2022, airport from 2023.
- "Discover clusters" tab is qualitative only (no labels to score it).
- Indian data is Sentinel-2 and has no ground truth, so we show it but never put it in a results table.

## Report chapters (current structure, do not re-add removed parts)

1. Introduction 2. Project Description and Goals 3. Design 4. Dataset 5. Implementation 6. Results 7. Dataset Links.
No literature review, no technical specification chapter, no DFD / use-case diagrams, no table of contents, no references (ma'am's request). Max 20 pages.

## Scope (say it clearly)

PS SIH26227 asks for 6 features. We built search and change analysis, did false-alarm removal partly (brightness matching works; cloud mask and speck removal tested and left off), and showed clustering in the demo without a metric. We did NOT build the analyst review screen, audit trail, or offline / incremental server.

## Files to know (each person should explain their own)

- `src/embedder.py`, `index_builder.py`, `retrieval.py` - embed, index, search
- `src/change_detect.py`, `suppress.py` - change and false-alarm removal (`despeckle` is off by default)
- `src/time_series.py`, `change_types.py` - multi-year and index-based change types
- `experiments/` - one script per result; numbers go to `results/`
- `demo/` - the app and the data fetchers

## Still to do before submitting

- [ ] Supervisor name and designation on the title page
- [ ] Each person rewrites their chapter in their own words
- [ ] Contribution statement (who did what)
- [ ] Update the PPT - it still states the old claim that the embedding idea works everywhere
- [ ] ISRO LISS-IV 2025 scene (a 2023 scene is downloaded; a clean 2025 date is still needed for a pair)
- [ ] Never commit passwords; Bhoonidhi login goes in environment variables only
