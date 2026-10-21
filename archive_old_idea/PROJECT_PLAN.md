# 🎯 Adaptive Image Quality Assessment & Enhancement for Long-Range Optical Surveillance
### BCSE407L – Computer Vision | Research Integrated Learning (RIL)

---

## 📝 Project Title

> **Adaptive Image Quality Assessment and Enhancement for Long-Range Optical Surveillance Using Classical Computer Vision**

---

## 🎯 One-Line Goal

Build a system that looks at a degraded surveillance image, automatically figures out what is wrong with it (haze, darkness, noise, blur), tunes classical enhancement algorithms accordingly, and proves the result is better by showing more ORB features get detected — not just that the image "looks nicer."

---

## 🔬 Research Question

> *"Can adaptive parameter optimization of classical image enhancement algorithms improve feature detection and target visibility in long-range surveillance images compared to fixed-parameter baselines?"*

---

## 💡 Locked Novelty Sentence

> **All team members must use this exact framing in every section they write. Do not drift from it.**

*"We evaluate enhancement quality using downstream feature detection performance (ORB keypoint count and distribution) as the primary optimization target, rather than reference-based perceptual metrics alone."*

**What this means in plain words:**
- We are NOT making images look prettier
- We ARE making images yield more useful keypoints for a CV/targeting task
- Enhancement is judged by whether ORB detects more features — not by PSNR alone

**What this is NOT:**
- Not "we automatically choose which algorithm to use" — that exists already
- Not "we chain filters together" — that exists already
- The novelty is: adaptive tuning of parameters, evaluated by a downstream CV task

---

## 🏗️ What the System Does (Pipeline)

`
📷 INPUT IMAGE
       │
       ▼
┌──────────────────────────────────┐
│       STAGE 1: QUALITY ANALYSIS  │
│                                  │
│  Measure 5 things:               │
│  • Contrast Score    (RMS)       │
│  • Haze Level        (Dark Ch.)  │
│  • Brightness        (mean px)   │
│  • Sharpness         (Lap. var)  │
│  • Noise Level       (std dev)   │
└──────────────────┬───────────────┘
                   │
┌──────────────────▼───────────────┐
│    STAGE 2: PARAMETER DECISION   │
│                                  │
│  Hazy?  → increase DCP strength  │
│  Dark?  → raise CLAHE clip       │
│  Noisy? → increase bilateral σ   │
│  Blur?  → apply unsharp mask     │
│                                  │
│  Parameters adapt per image,     │
│  not fixed for all images.       │
└──────────────────┬───────────────┘
                   │
┌──────────────────▼───────────────┐
│      STAGE 3: ENHANCEMENT        │
│                                  │
│  Step 1: Dehazing  (if hazy)     │
│          Dark Channel Prior      │
│          + guided filter fix     │
│  Step 2: CLAHE (adaptive clip)   │
│  Step 3: Bilateral Filter        │
│  Step 4: Unsharp Masking         │
└──────────────────┬───────────────┘
                   │
┌──────────────────▼───────────────┐
│   STAGE 4: DOWNSTREAM EVALUATION │
│                                  │
│  • ORB keypoint detection        │  ← PRIMARY success metric
│  • Canny edge map                │
│  • Target region highlighting    │
└──────────────────┬───────────────┘
                   │
                   ▼
        📊 METRICS + RESULTS
`

---

## 📊 What We Compare

| Method | Type | Description |
|---|---|---|
| Raw Input | Baseline | No enhancement at all |
| Histogram Equalization | Baseline | Fixed global contrast stretch |
| CLAHE (fixed, clip=2) | Baseline | Standard fixed-parameter CLAHE |
| Dark Channel Prior | Baseline | Standard DCP, no tuning |
| **Ours (Adaptive)** | **Proposed** | **Adaptive parameter tuning per image** |

**Headline metric:** ORB feature count (does our enhanced image yield more detectable keypoints?)

**Secondary metrics:** Entropy, Edge Preservation Index, Runtime

**Reference metrics (O-HAZE only):** PSNR, SSIM

---

## 📁 Datasets

| Dataset | Images | PSNR/SSIM Valid? | What We Use It For |
|---|---|---|---|
| **O-HAZE** | 45 pairs | ✅ YES — paired ground truth | PSNR, SSIM, Entropy, ORB count |
| **RESIDE (ITS/SOTS)** | ~1,000 | ⚠️ Synthetic subset only | Entropy, ORB count |
| **ExDark** | 7,363 | ❌ NO — no reference | Entropy, ORB count only |
| **D-HAZY** | 1,400 | ⚠️ VERIFY before using | Drop it if synthetic/real is mixed and unclear |

> **Rule:** PSNR and SSIM are only computed on O-HAZE. For all other datasets, report Entropy, Edge Preservation Index, ORB feature count, and runtime only. State this explicitly in your paper's Section 4.

---

## 📏 Metrics

| Metric | Valid On | Direction | What It Proves |
|---|---|---|---|
| **ORB Feature Count** | All datasets | ↑ Higher = better | Enhancement helps downstream CV task |
| **Entropy** | All datasets | ↑ Higher = better | More information content in image |
| **Edge Preservation Index** | All datasets | ↑ Higher = better | Edges not blurred out |
| **Runtime (ms)** | All datasets | ↓ Lower = better | Practical speed |
| **PSNR** | O-HAZE only | ↑ Higher = better | Pixel-level accuracy vs. ground truth |
| **SSIM** | O-HAZE only | ↑ Higher = better | Structural similarity vs. ground truth |

---

## ⚠️ What We Will NOT Claim

| ❌ Do NOT Say This | ✅ Say This Instead |
|---|---|
| "We built the Army's surveillance system" | "We propose a preprocessing module for long-range surveillance" |
| "We invented a new algorithm" | "We propose adaptive tuning of existing classical algorithms" |
| "Results are guaranteed to improve" | "We test the hypothesis and report results honestly" |
| "This works in all conditions" | "Evaluated on four benchmark datasets with stated limitations" |
| "We automatically choose which algorithm to use" | "We automatically tune the parameters of a fixed pipeline" |

---

## ✅ Non-Negotiable Checkpoint (Run Before Anything Else)

**Before any other work begins:**

Run 3 test images through raw → fixed CLAHE → adaptive pipeline. Count ORB features for each.

`
Image 1: 1 hazy image from O-HAZE     (pick one with no large sky area)
Image 2: 1 dark image from ExDark     (indoor or street scene)
Image 3: 1 mixed image from RESIDE

For each image, record:
  - ORB count: raw image
  - ORB count: fixed CLAHE (clip=2)
  - ORB count: your adaptive pipeline

PASS:  adaptive beats fixed CLAHE on ≥ 2 out of 3 images → proceed
FAIL:  adaptive does not beat fixed on ≥ 2 out of 3 → fix quality-analysis
       thresholds first, then re-run. Do not build further until gate passes.
`

> This is a **gate**, not a formality.

---

## 🔧 DCP Constraint (Important from Day 1)

Dark Channel Prior has known failures with:
- Large bright sky regions
- Predominantly white/bright objects

**From Day 1, select test images that have:**
- Foreground objects (vehicles, people, buildings)
- No large open sky areas
- Acknowledge sky-region limitation in your paper's limitations section

---

## 📰 Paper Structure

| Section | Content |
|---|---|
| **1. Introduction** | Problem + why existing methods fail + 3 contributions (list the locked novelty sentence here) |
| **2. Related Work** | 6–8 papers on: CLAHE, DCP, Retinex, bilateral filter, adaptive enhancement, ORB |
| **3. Methodology** | Pipeline flowchart + quality analysis formulas + parameter decision rules |
| **4. Experimental Setup** | Datasets (with PSNR/SSIM scope stated), metrics, baselines, software |
| **5. Results & Discussion** | Tables (real numbers only) + side-by-side image comparisons |
| **6. Ablation Study** | Remove each stage one at a time — proves every step matters |
| **7. Conclusion** | Summary + limitations (incl. DCP sky failure) + future work |

---

## 🎯 Key Papers to Cite

| Paper | Year | Why Cite It |
|---|---|---|
| He et al. — *Dark Channel Prior* | 2011 | Your dehazing baseline |
| Zuiderveld — *CLAHE* | 1994 | Your contrast enhancement |
| Tomasi & Manduchi — *Bilateral Filter* | 1998 | Your noise filter |
| Rublee et al. — *ORB* | 2011 | Your evaluation metric |
| Wang et al. — *SSIM* | 2004 | Your image quality metric |
| Land — *Retinex* | 1977 | Related enhancement approach |
| 2–3 recent papers (2020–2024) on adaptive surveillance image enhancement | — | Show current gap |

---

## 🛠️ Tools

- Python 3.10+
- OpenCV (cv2)
- NumPy
- Matplotlib
- Scikit-image (SSIM, PSNR)
- Google Colab (free, no GPU needed)

---

*BCSE407L Computer Vision RIL | Last Updated: September 2026*
