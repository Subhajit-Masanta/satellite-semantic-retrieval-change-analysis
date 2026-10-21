# Validation Notes

Bugs and design flaws found by actually running the pipeline, not by reading it.
Each one is worth a sentence in the paper — they are evidence the evaluation
protocol does real work.

*Last updated: 28 September 2026*

---

## 1. A brightness gate deletes bright real change

**Symptom.** A new bright structure (a flat 255 patch standing in for a concrete
or metal roof) was detected correctly by `change_mask` — 6400 px flagged — and
then completely erased by suppression. Final F1: **0.000**.

**Cause.** `cloud_shadow_mask` originally masked out anything with grey value
≥ 230 as "cloud". New construction in satellite imagery *is* bright. The
suppressor was therefore deleting precisely the change type the project cares
about most.

**Fix.** Brightness alone cannot separate cloud from bright change. Two further
cues were added:
- **Texture** — cloud is locally smooth; a roof is edge-rich.
- **Area fraction** — cloud covers a large fraction of a tile; a roof does not.
  Expressed as a fraction of tile area (not a pixel count) so it keeps its
  meaning when tile size changes.

Detected cores are then **buffered outward**, because a cloud's feathered edge
and cast shadow are not smooth enough to be caught directly but are equally
unusable. Buffering a cloud mask is standard practice.

**Result.** Bright change F1 went 0.000 → **1.000**, while a genuine large
smooth cloud is still 100% suppressed.

**Residual limitation, state it in the paper.** A very large, perfectly flat
bright roof is genuinely indistinguishable from cloud by these cues and will be
masked. Nothing here reads spectral signature — a real system would use the
sensor's own cloud product (e.g. the Sentinel-2 SCL band), which this stands in
for.

---

## 2. Masking must run on the raw acquisition, not the normalised one

**Symptom.** After fix 1, cloud suppression still only removed ~22% of a pure
cloud blob, far less than expected.

**Cause.** `analyse_pair` ran `radiometric_normalise` *before* computing the
cloud mask. Normalisation rescales t2 to match t1's mean and standard
deviation, which pulled the cloud core from **250 → 203** — under the 248
detection threshold. The cloud was then never flagged. Silent failure: no
error, just a suppressor that quietly did nothing.

**Fix.** Cloud and shadow are radiometric properties of the **raw**
acquisition, so the mask is computed on the originals. Normalisation is for
*comparing* the two dates; masking must precede it conceptually.

**Result.** Cloud suppression went 21.6% → **100%**, while a real roof in the
same tile survives (recall 0.938, precision 1.000).

---

## 3. Gate C was gameable — it now has two halves

**Symptom.** A deliberately broken suppressor that returns an empty mask for
every input **passed Gate C 3/3**.

**Cause.** Gate C originally only asked "does suppression reduce false-alarm
area on no-change pairs?". A suppressor that detects nothing at all scores a
perfect zero false-alarm area and passes — while being completely useless.
Since Gate C is the experiment that validates the project's novelty claim, a
gameable Gate C meant the novelty claim was unfalsifiable.

**Fix.** Gate C now requires **both**:

| Half | Test | Threshold |
|---|---|---|
| False alarm | On no-change pairs, suppression reduces flagged area vs. naive differencing | ≥ 2 of 3 pairs |
| **Sensitivity** | On real-change pairs, suppression retains recall vs. unsuppressed | worst-case ≥ 80% retained |

**Result.** The honest suppressor passes both halves. The do-nothing suppressor
now fails the sensitivity half (0% recall retained) and correctly fails the gate.

> **Lesson worth stating in the report:** a precision-oriented metric must
> always be paired with a sensitivity floor, or "improving precision" collapses
> into "predict nothing". This is the same trap the original ORB-count idea
> fell into from the other direction, where noise inflated the headline number.

---

## 4. Gate B saturated and could not distinguish the methods

**Symptom.** On the first synthetic test, pixel-differencing + Otsu and our
method both scored **F1 = 1.000**. The gate recorded this as a *loss* (because
`ours > baseline` was false) and failed — while actually telling us nothing.

**Cause.** Two separate problems:
1. A tie was being counted as a loss.
2. More importantly, the test pair was trivially easy — a clean bright square
   on clean noise, with no confounds. Both methods max out, so the comparison
   is uninformative regardless of which way it resolves.

**Fix.**
- Ties are now reported separately and count toward passing, but **only when
  not saturated** — matching on a hard pair is informative, matching at
  F1 ≈ 1.000 on a trivial one is not.
- The gate warns explicitly if every pair saturates.
- The synthetic set now includes **confounded** pairs (real change *plus* an
  illumination shift *plus* a cloud), which is the realistic case and the one
  where plain differencing should struggle.

**Result.** Gate B now discriminates: ties on the trivial pairs, wins on the
confounded ones (F1 0.451 → 0.492).

---

## 5. Noise fragmented the cloud so it escaped the area test

**Symptom.** On a pair containing a real roof *plus* a cloud *plus* an
illumination shift, the cloud was still being reported as change. Precision on
that pair: **0.332**.

**Cause.** The cloud's bright region was 6.45% of the tile — comfortably over
the 4% area threshold — but scene noise showing through its semi-transparent
feathered edge **split it into fragments** (2451 px and 1156 px). Connected
components measured each fragment separately, neither cleared the bar, and the
whole cloud escaped.

**Fix.** A cloud is one object, so the candidate mask is morphologically
**closed** before components are measured. This heals noise-induced gaps
without weakening the discriminator.

---

## 6. The area threshold was an arbitrary guess sitting in the wrong place

**Symptom.** Even after fix 5, the cloud escaped — by 42 pixels.

**Cause.** Measured on the test scene: cloud = 2579 px, roof = 1156 px,
threshold = 2621 px. The size discriminator was working correctly — the cloud
*is* more than twice the roof — but the threshold (an arbitrary 4%) happened to
land just above the cloud rather than in the gap between the two.

**Fix.** Moved to 3%, which sits in the gap with margin on both sides. More
importantly, `min_area_frac` is now documented as **the** tuning knob:

> It is the cloud/change decision boundary and therefore sets the
> precision-sensitivity tradeoff directly. Too high and clouds leak through as
> false change; too low and large real structures get masked as cloud. The
> default is a starting point, **not a calibrated value** — sweep it on
> LEVIR-CD/OSCD and report the sweep in the ablation (Section 6).

**Result on the confounded pair:**

| | before | after |
|---|---|---|
| Precision | 0.332 | **1.000** |
| Recall | 0.946 | 0.946 |
| F1 | 0.492 | **0.972** |

Gate B correspondingly became decisive rather than marginal: ours **0.972** vs
pixel-Otsu **0.451** on confounded pairs.

> **A caution for Person C.** Findings 5 and 6 were both tuned against
> *synthetic* data. The mechanism is sound — cloud is smoother and larger than
> built change — but the specific numbers (3%, 9×9 closing, 25 px buffer) are
> not calibrated. Re-tune them on real LEVIR-CD/OSCD before reporting anything,
> and put the sweep in the ablation.

---

## Current status

Run `python gates.py --synthetic`:

```
GATE A: PASS     retrieval rankings well-formed and query-dependent
GATE B: PASS     3 wins, 2 ties, 0 losses  (ours F1 0.972 vs otsu 0.451)
GATE C: PASS     false-alarm half 2/3; sensitivity 95% worst-case retained
```

Behaviour on the synthetic confound suite:

| Pair | Naive | With suppression | Correct? |
|---|---|---|---|
| Illumination shift only, no real change | 65 536 px flagged | **0 px** | ✅ |
| Cloud only, no real change | 7 270 px flagged | **0 px** | ✅ |
| Clean real change | F1 1.000 | F1 **1.000** | ✅ |
| Real change + cloud + illumination | F1 0.451 | F1 **0.972** | ✅ |
| Bright / dark / mid-tone real change | — | F1 1.000 / 1.000 / 0.999 | ✅ |

**These are mechanism checks on synthetic data, not results.** They prove the
chain works and the gates can fail. Real numbers require LEVIR-CD and OSCD with
a real encoder — the hashing fallback is not a learned model and nothing
produced with it is reportable.
