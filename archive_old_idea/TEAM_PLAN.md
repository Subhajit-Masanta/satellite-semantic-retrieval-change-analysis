# 🎯 BCSE407L Computer Vision — RIL Team Plan
### Adaptive Image Quality Assessment & Enhancement for Long-Range Optical Surveillance
#### 4-Member Team | 8 Weeks | 30 Marks

---

## 📝 Project Title

> **Adaptive Image Quality Assessment and Enhancement for Long-Range Optical Surveillance Using Classical Computer Vision**

---

## 🎯 Project Goal (One Line)

Build a system that looks at a degraded surveillance image, automatically detects what is wrong with it (haze, darkness, noise, blur), tunes classical enhancement algorithms accordingly, and proves the result is better by showing more ORB features get detected — not just that the image "looks nicer."

---

## 🔬 Research Question

> *"Can adaptive parameter optimization of classical image enhancement algorithms improve feature detection and target visibility in long-range surveillance images compared to fixed-parameter baselines?"*

---

## 💡 Locked Novelty (Everyone Must Use This)

> *"We evaluate enhancement quality using downstream feature detection performance — ORB keypoint count and distribution — as the primary optimization target, rather than reference-based perceptual metrics alone."*

| ✅ What We Claim | ❌ What We Do NOT Claim |
|---|---|
| Adaptive parameter tuning per image | We invented a new algorithm |
| Enhancement judged by ORB features | We built the Army's full system |
| Preprocessing module for surveillance | Results are guaranteed to be better |
| Honest comparison with 4 baselines | Works perfectly in all conditions |

---

## ⚡ How to Work in Parallel (NO CHAIN DEPENDENCY)

You do **NOT** need to wait for the previous person to finish. Everyone works independently from Day 1.

*   **Person A (Quality/Parameters):** Test your code by passing raw images and printing out the parameter decisions. You don't need Person B's enhancement code to verify your logic.
*   **Person B (Enhancement Pipeline):** Test your code by hardcoding parameters (e.g., clip=2, sigma=50) and passing sample images. You don't need Person A's decision logic to build and test the filters.
*   **Person C (Metrics/Evaluation):** Test your metric functions by passing ANY two images (e.g., a raw image and a manually blurred version of it) to ensure the math works. You don't need A or B's final output to write your metric scripts.
*   **Person D (Paper/Presentation):** Start writing the Introduction, Related Work, and setting up the paper template immediately. You don't need any code to be finished for this.

**Integration happens in Week 4**, where you link A → B → C together in main.py.

---

## 🏗️ System Pipeline

`
📷 INPUT IMAGE
       │
       ▼
┌──────────────────────────────────┐
│  STAGE 1: QUALITY ANALYSIS       │  ← Person A owns this
│  • Contrast Score   (RMS)        │
│  • Haze Level       (Dark Ch.)   │
│  • Brightness       (mean px)    │
│  • Sharpness        (Lap. var)   │
│  • Noise Level      (std dev)    │
└──────────────────┬───────────────┘
                   │
┌──────────────────▼───────────────┐
│  STAGE 2: PARAMETER DECISION     │  ← Person A owns this
│  Hazy?  → raise DCP strength     │
│  Dark?  → raise CLAHE clip       │
│  Noisy? → raise bilateral sigma  │
│  Blur?  → apply sharpen kernel   │
└──────────────────┬───────────────┘
                   │
┌──────────────────▼───────────────┐
│  STAGE 3: ENHANCEMENT PIPELINE   │  ← Person B owns this
│  Step 1: Dark Channel Prior      │
│          + guided filter         │
│  Step 2: CLAHE (adaptive clip)   │
│  Step 3: Bilateral Filter        │
│  Step 4: Unsharp Masking         │
└──────────────────┬───────────────┘
                   │
┌──────────────────▼───────────────┐
│  STAGE 4: EVALUATION             │  ← Person C owns this
│  • ORB keypoint count            │
│  • Canny edge map                │
│  • PSNR / SSIM (O-HAZE only)     │
│  • Entropy + Edge Preservation   │
│  • Runtime measurement           │
└──────────────────┬───────────────┘
                   │
                   ▼
        📊 RESULTS + PAPER          ← Person D owns this (All Writing)
`

---

## 👥 Team Work Division

---

### 👤 Person A — Quality Analysis + Parameter Decision

**You own Stages 1 and 2. The entire project depends on this working correctly.**

**What you build:**
*   quality_analyzer.py (~50 lines): Measures 5 image quality scores.
*   param_selector.py (~40 lines): Returns parameters (CLAHE clip, bilateral sigma, etc.) based on the scores.

**Parallel Work Strategy:** Just pass 3 raw images into your script and print the parameter outputs to verify the logic. Do not wait for Person B.

**Your deliverable — PILOT TEST (must be done in Week 1, Day 2-3):**
*   Pick 3 images (1 hazy O-HAZE, 1 dark ExDark, 1 mixed RESIDE).
*   Run them through your quality logic.
*   If your logic correctly identifies the dark image as needing a higher CLAHE clip, and the hazy image as needing dehazing, you pass the gate.

*(No paper writing for you. Just provide your parameter decision formulas to Person D in a text file).*

---

### 👤 Person B — Enhancement Pipeline

**You own Stage 3. You build the actual image processing modules.**

**What you build:**
*   dehazer.py (~80 lines): Dark Channel Prior + guided filter.
*   enhancer.py (~20 lines): CLAHE, bilateral filter, unsharp masking.

**Parallel Work Strategy:** Do not wait for Person A. Write your functions to accept parameters as arguments (def apply_clahe(img, clip_limit):). Test them by passing manual values like clip_limit=3.0 to see if the image enhances correctly.

**⚠️ DCP Constraint — from Day 1:**
> When selecting test images for Dark Channel Prior testing, **avoid images with large bright sky regions or predominantly white/bright objects**.

*(No paper writing for you. Just provide bullet points explaining how your pipeline works to Person D).*

---

### 👤 Person C — Evaluation & Metrics

**You own Stage 4. You measure everything and generate all graphs.**

**What you build:**
*   eature_eval.py (~15 lines): ORB keypoint counting.
*   metrics.py (~30 lines): PSNR, SSIM, Entropy, Edge Preservation.
*   isualizer.py (~50 lines): Bar charts and comparison plots.

**Parallel Work Strategy:** Do not wait for A or B. Build your metric functions to accept image1 and image2. Test them by taking a raw image (image1), blurring it yourself (image2), and verifying your metrics output the correct numbers (e.g., PSNR/SSIM drops). 

**⚠️ Critical metric rule:**
Never report PSNR/SSIM on ExDark or RESIDE-real. Only O-HAZE and synthetic RESIDE.

*(No paper writing for you. Just generate the final graphs, tables, and metric numbers, and hand them to Person D).*

---

### 👤 Person D — Paper Writing + Presentation

**You own 100% of the research paper and final presentation.**

**Your responsibilities:**
1. Write the **ENTIRE** paper (Sections 1 through 7). A, B, and C will not write paper text, they only write code and give you raw data/formulas.
2. Build the final presentation slides.
3. Prepare the demo video / screenshots.

**Parallel Work Strategy:** Do not wait for the code! You can write Section 1 (Intro) and Section 2 (Related Work) in Weeks 1-2. You can set up the IEEE/Springer paper template and write the skeleton for Section 3 (Methodology) based on this plan.

**Paper outline:**
| Section | Content | Who Writes |
|---|---|---|
| 1. Introduction | Problem + motivation + 3 contributions (use novelty sentence) | **D** |
| 2. Related Work | 6-8 papers, 2-3 lines each | **D** |
| 3. Methodology | Pipeline + quality analysis math + parameter rules (Data from A/B) | **D** |
| 4. Experimental Setup | Datasets (PSNR scope stated) + metrics + baselines | **D** |
| 5. Results & Discussion | Tables + side-by-side images + analysis (Data from C) | **D** |
| 6. Ablation Study | Remove one stage at a time, show ORB count drops (Data from C) | **D** |
| 7. Conclusion | Summary + DCP sky limitation + future work | **D** |

---

## 📅 8-Week Timeline (Parallel Execution)

| Week | Person A | Person B | Person C | Person D |
|---|---|---|---|---|
| **Week 1** | Download datasets. Run Pilot Test. | Build enhancer.py (CLAHE, bilateral) with dummy params | Build metrics.py (PSNR, SSIM, Entropy) using dummy images | Setup paper template. Write Sec 1 (Intro) |
| **Week 2** | Build quality_analyzer.py (measure 5 metrics) | Build dehazer.py (DCP + guided filter) | Build eature_eval.py (ORB counter) | Write Sec 2 (Related Work) |
| **Week 3** | Build param_selector.py (threshold logic) | Test enhancement pipeline on dummy dark/hazy images | Build isualizer.py (plotting scripts) | Draft Sec 3 (Methodology) structure |
| **Week 4** | **INTEGRATION:** Connect A → B | **INTEGRATION:** Receive params from A, run pipeline | **INTEGRATION:** Receive final images from B, run metrics | Polish Sec 1-3 |
| **Week 5** | Fix bugs in A | Fix bugs in B | Generate final tables & graphs | Write Sec 4 (Exp Setup) |
| **Week 6** | Help D with formulas | Help D with pipeline details | Help D with graphs/tables | Write Sec 5 & 6 (Results & Ablation) |
| **Week 7** | Review code | Review code | Review code | Write Sec 7 (Conclusion) + Presentation |
| **Week 8** | Final Polish | Final Polish | Final Polish | Final Polish & Demo |

---

## 📁 Datasets

| Dataset | Where to Get | What to Use It For |
|---|---|---|
| **O-HAZE** | https://data.vision.ee.ethz.ch/cvl/ntire18/o-haze/ | PSNR, SSIM, ORB, Entropy |
| **RESIDE (ITS)** | https://sites.google.com/view/reside-dehaze-datasets | ORB, Entropy (synthetic subset) |
| **ExDark** | https://github.com/cs-chan/Exclusively-Dark-Image-Dataset | ORB, Entropy only |

---

## 📊 Baselines to Compare Against

| Method | PSNR↑ | SSIM↑ | Entropy↑ | ORB Features↑ | Runtime↓ |
|---|---|---|---|---|---|
| Raw Input | — | — | — | — | — |
| Histogram Equalization | — | — | — | — | — |
| CLAHE fixed (clip=2) | — | — | — | — | — |
| Dark Channel Prior | — | — | — | — | — |
| **Ours (Adaptive)** | **—** | **—** | **—** | **—** | **—** |

---

## 🛠️ Code Files Summary

| File | Owner | Lines | Purpose |
|---|---|---|---|
| quality_analyzer.py | A | ~50 | Measure 5 image quality scores |
| param_selector.py | A | ~40 | Decide parameters per image |
| dehazer.py | B | ~80 | Dark Channel Prior + guided filter |
| enhancer.py | B | ~20 | CLAHE, bilateral, sharpen |
| eature_eval.py | C | ~15 | ORB keypoint counting |
| metrics.py | C | ~30 | PSNR, SSIM, Entropy, Edge Preservation |
| isualizer.py | C | ~50 | Plots, comparison figures |
| main.py | All | ~30 | Run full pipeline end to end (Week 4) |
| **Total** | | **~315 lines** | |

---

## ✅ Submission Checklist

- [ ] Pilot gate passed
- [ ] All 4 baselines run and measured
- [ ] Results table filled with real numbers
- [ ] PSNR/SSIM only reported on O-HAZE
- [ ] Novelty sentence appears in abstract + Section 1
- [ ] 7-section paper complete (Written by Person D)
- [ ] Code cleaned and commented
- [ ] Presentation slides done
- [ ] Individual contribution statement written (required by RIL)

---

*BCSE407L Computer Vision RIL | 4-Member Team Plan | Last Updated: September 2026*
