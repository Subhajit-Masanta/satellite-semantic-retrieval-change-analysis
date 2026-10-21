# 🎯 BCSE407L Computer Vision — RIL Execution Plan
### Adaptive Image Quality Assessment & Enhancement for Long-Range Optical Surveillance
#### 4-Member Team | Execution & Demonstration Focus

---

## 🏗️ How the Team Works (Parallel vs. Chain)

You **DO NOT** need to wait for each other. You can build and test your parts **in parallel** by using placeholder images or dummy outputs. 

Here is how you unblock each other:

| Person | Stage | How to work independently |
|---|---|---|
| **P1** | Quality Analysis | Feed any raw image into your code. Just print the 5 scores to terminal. You don't need P2's enhancement code to do this. |
| **P2** | Enhancement | Hardcode fake parameters (e.g., clip=3.0, sigma=60) to test your filters on a raw image. You don't need P1's analysis to build this. |
| **P3** | Metric Evaluation | Feed any two random images into your metric functions (treat one as "raw", one as "enhanced"). Just make sure the numbers output correctly. You don't need P1 or P2. |
| **P4** | Downstream CV & Integration | Use standard OpenCV algorithms (like Canny or Haar Cascades) on raw images to prove that enhancement helps downstream CV tasks. |

*Once everyone finishes their individual parts, P4 connects them all together like Lego blocks.*

---

## 👥 Task Division (No Paper Writing, Pure Execution)

### 👤 Person 1: The Analyzer (Quality & Decision)
**Goal:** Build the brain that looks at an image and decides what parameters to use.
* **Code to write:** quality_analyzer.py and param_selector.py
* **What it does:** Measures Brightness, Contrast, Sharpness, Haze, Noise. Then outputs 4 variables: clahe_clip, ilateral_s, pply_dehaze, pply_sharp.
* **Testing:** Just run it on 5 different images and show that it picks different, sensible parameters for each.

### 👤 Person 2: The Enhancer (Filters & Processing)
**Goal:** Build the actual image processing functions.
* **Code to write:** dehazer.py and enhancer.py
* **What it does:** Implements Dark Channel Prior (DCP), CLAHE, Bilateral Filter, and Unsharp Masking as separate, callable functions.
* **Testing:** Hardcode some parameters and run them on a raw image to make sure the filters don't crash and actually change the image.

### 👤 Person 3: The Evaluator (Metrics & Measurement)
**Goal:** Build the ruler that measures if the enhancement actually worked.
* **Code to write:** eature_eval.py and metrics.py
* **What it does:** Counts ORB keypoints. Calculates Entropy, Edge Preservation Index, PSNR, SSIM, and measures runtime.
* **Testing:** Run it on any two images and make sure it spits out the right numbers.

### 👤 Person 4: Downstream CV & Integration (The "So What?" Factor)
**Goal:** Prove that the enhancement actually helps a real computer vision task, and tie the pipeline together.
* **Code to write:** downstream_tasks.py, main.py and isualizer.py
* **What it does:** 
  1. **Image Processing (IP) Task:** Implement a downstream CV task that usually fails on hazy/dark images (e.g., Target Region Highlighting using contour detection, or Canny Edge mapping). You run this on the raw image vs. the enhanced image to prove the enhanced one works better.
  2. **Integration:** Imports P1, P2, and P3's code. Passes an image to P1, takes the parameters, gives them to P2 to enhance the image, passes both to P3 to get metrics.
  3. **Demo:** Plots a beautiful side-by-side comparison using matplotlib.
* **Testing:** Run your downstream task (like contour detection) on a raw dark image (should fail/find nothing) and then on a manually brightened image (should find objects).

---

## 🚀 Execution Steps (The "Chain" happens at the very end)

1. **Setup (Day 1):** Create the empty .py files. Everyone claims their files.
2. **Parallel Build (Days 2-4):** Everyone writes and tests their own functions independently. P4 builds their downstream contour/edge detection script.
3. **The Pilot Gate (Day 5):** P1 tests their thresholds. If it fails, P1 adjusts.
4. **Integration (Day 6-7):** P4 writes main.py and imports everyone's functions. The chain is connected.
5. **Testing & Tuning (Day 8-10):** Run main.py on 10 different dataset images. Tweak P1's thresholds if the results look bad.
6. **Final Demo (Day 11-14):** P4 makes the output look amazing (side-by-side images, bar charts of ORB features). Prepare the live demonstration.
