"""EuroSAT Prompt Tuning & Honest Evaluation.

Protocol to avoid data leakage:
  1. Split each class 50/50 by sorted filename:
     - TUNE half:   first 50% of images (used strictly to test & select candidate prompts)
     - REPORT half: remaining 50% of images (held out; never seen during prompt selection)
  2. Embed all patches using GPU with batched inference.
  3. Evaluate candidate prompts for weak classes (SeaLake, HerbaceousVegetation) on TUNE half.
  4. Select the best prompt per class purely based on TUNE AP.
  5. Form PROMPTS_V2 with the selected prompts.
  6. Evaluate both PROMPTS_V1 (baseline) and PROMPTS_V2 (tuned) on:
     - REPORT split (completely independent test)
     - FULL dataset (for direct comparison with previously published baseline, with disclosure)
  7. Test on both OpenCLIP and RemoteCLIP to assess cross-model generalization.
  8. Save full metrics to results/eurosat_prompt_tuning_results.json.
"""

import json
import os
import sys
import time
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

PROJECT = Path("c:/Users/subha/Desktop/projects/Computer Vision")
sys.path.insert(0, str(PROJECT / "env" / "libs"))
sys.path.insert(0, str(PROJECT / "env" / "libs_gpu"))
sys.path.insert(0, str(PROJECT))

import src.embedder as emb
from src import metrics as M

DATA_DIR = PROJECT / "data" / "EuroSAT" / "2750"
OUT_DIR = PROJECT / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- Prompts
PROMPTS_V1 = {
    "AnnualCrop": "a satellite image of annual crop land",
    "Forest": "a satellite image of a forest",
    "HerbaceousVegetation": "a satellite image of herbaceous vegetation",
    "Highway": "a satellite image of a highway or road",
    "Industrial": "a satellite image of industrial buildings",
    "Pasture": "a satellite image of pasture land",
    "PermanentCrop": "a satellite image of permanent crop land",
    "Residential": "a satellite image of residential buildings",
    "River": "a satellite image of a river",
    "SeaLake": "a satellite image of a sea or lake",
}

CANDIDATES = {
    "SeaLake": [
        "a satellite image of a sea or lake",                      # V1 baseline
        "a satellite image of open ocean or inland lake water",
        "a satellite image of a large body of water",
        "a satellite photo of sea water or a lake surface",
        "a remote sensing image of ocean or lake",
        "a satellite image of water: sea, lake or reservoir",
        "a satellite photo of an ocean, sea, or lake",
        "a satellite image of sea or lake water body",
        "a satellite image of deep blue water, sea or lake",
        "a satellite image of open sea or fresh water lake",
    ],
    "HerbaceousVegetation": [
        "a satellite image of herbaceous vegetation",              # V1 baseline
        "a satellite image of grassland or meadow",
        "a satellite image of open grass fields",
        "a satellite image of low green vegetation or grassland",
        "a satellite image of scrubland and grass",
        "a satellite image of wild grasses and meadow",
        "a satellite image of natural grassland or herbaceous meadow",
        "a satellite image of green herbaceous vegetation and grass",
        "a satellite image of pastures, meadows and herbaceous vegetation",
    ],
}


class FastEuroSAT(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as img:
            return self.transform(img.convert("RGB"))


def compute_ap(scores, labels, target_class_idx):
    order = np.argsort(-scores)
    rel = set(np.flatnonzero(labels == target_class_idx).tolist())
    return float(M.average_precision(order.tolist(), rel))


def get_embeddings(model_name, classes, tune_paths, report_paths):
    cache_path = CACHE_DIR / f"eurosat_embeddings_{model_name}.npz"
    if cache_path.exists():
        print(f"Loading cached {model_name} embeddings from {cache_path} ...")
        data = np.load(cache_path)
        return data["V_tune"], data["L_tune"], data["V_report"], data["L_report"]

    print(f"\nInitializing {model_name} embedder on GPU ...")
    embedder = emb.Embedder(backend=model_name)
    assert embedder.device == "cuda", f"Expected CUDA, got {embedder.device}"

    def encode_files(file_list):
        ds = FastEuroSAT(file_list, embedder.preprocess)
        loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
        vecs = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to("cuda", non_blocking=True)
                f = embedder.model.encode_image(batch)
                vecs.append(embedder._norm(f.cpu().numpy().astype(np.float32)))
        return np.concatenate(vecs)

    print(f"Encoding {len(tune_paths)} TUNE patches ...")
    t0 = time.perf_counter()
    V_tune = encode_files(tune_paths)
    print(f"TUNE encoded in {time.perf_counter()-t0:.1f}s, shape: {V_tune.shape}")

    print(f"Encoding {len(report_paths)} REPORT patches ...")
    t0 = time.perf_counter()
    V_report = encode_files(report_paths)
    print(f"REPORT encoded in {time.perf_counter()-t0:.1f}s, shape: {V_report.shape}")

    # Build labels
    L_tune, L_report = [], []
    for ci, c in enumerate(classes):
        c_tune_n = len(list((DATA_DIR / c).glob("*.jpg"))) // 2
        c_report_n = len(list((DATA_DIR / c).glob("*.jpg"))) - c_tune_n
        L_tune.extend([ci] * c_tune_n)
        L_report.extend([ci] * c_report_n)
    L_tune, L_report = np.array(L_tune), np.array(L_report)

    np.savez_compressed(cache_path, V_tune=V_tune, L_tune=L_tune,
                        V_report=V_report, L_report=L_report)
    print(f"Saved cached embeddings to {cache_path}")
    return V_tune, L_tune, V_report, L_report


def evaluate_split(V, L, classes, prompts, embedder):
    """Compute per-class AP, P@10, and classification accuracy."""
    T = embedder.encode_text([prompts[c] for c in classes])
    S = V @ T.T  # (n_images, n_classes)

    per_class = {}
    for ci, c in enumerate(classes):
        order = np.argsort(-S[:, ci])
        rel = set(np.flatnonzero(L == ci).tolist())
        ap = float(M.average_precision(order.tolist(), rel))
        p10 = float(M.precision_at_k(order, rel, 10))
        per_class[c] = {"ap": round(ap, 4), "p10": round(p10, 4)}

    mAP = float(np.mean([per_class[c]["ap"] for c in classes]))
    mp10 = float(np.mean([per_class[c]["p10"] for c in classes]))

    pred = S.argmax(1)
    acc = float((pred == L).mean())

    return {
        "per_class": per_class,
        "mAP": round(mAP, 4),
        "mean_p10": round(mp10, 4),
        "zero_shot_acc": round(acc, 4),
    }


def main():
    print("=" * 75)
    print("EUROSAT PROMPT TUNING & HONEST VALIDATION PIPELINE")
    print("=" * 75)

    classes = sorted(PROMPTS_V1.keys())
    tune_paths, report_paths = [], []

    print("\n[Step 1] Creating 50/50 class-balanced split (by sorted filename) ...")
    for c in classes:
        files = sorted((DATA_DIR / c).glob("*.jpg"))
        n = len(files)
        tune_paths.extend(files[: n // 2])
        report_paths.extend(files[n // 2 :])
        print(f"  {c:22s} total={n:5d}  tune={n//2:5d}  report={n - n//2:5d}")
    print(f"Total TUNE patches: {len(tune_paths)} | Total REPORT patches: {len(report_paths)}")

    # Load embedder and get OpenCLIP embeddings
    embedder_open = emb.Embedder(backend="openclip")
    V_tune, L_tune, V_rep, L_rep = get_embeddings("openclip", classes, tune_paths, report_paths)

    # -------------------------------------------------------- Prompt selection on TUNE
    print("\n" + "=" * 75)
    print("[Step 2] CANDIDATE PROMPT SELECTION ON TUNE HALF (NEVER looks at REPORT)")
    print("=" * 75)

    chosen_prompts = dict(PROMPTS_V1)
    tune_records = {}

    for c in classes:
        ci = classes.index(c)
        if c not in CANDIDATES:
            # Baseline prompt unchanged
            continue

        print(f"\nTuning candidates for '{c}':")
        cand_list = CANDIDATES[c]
        scores = []
        for prompt in cand_list:
            p_vec = embedder_open.encode_text(prompt)[0]
            sims = V_tune @ p_vec
            cand_ap = compute_ap(sims, L_tune, ci)
            is_orig = (prompt == PROMPTS_V1[c])
            tag = " [ORIGINAL]" if is_orig else ""
            print(f"  AP: {cand_ap:.4f}  |  Prompt: \"{prompt}\"{tag}")
            scores.append((cand_ap, prompt))

        # Best on TUNE
        best_ap, best_prompt = max(scores, key=lambda x: x[0])
        orig_ap = scores[0][0]
        chosen_prompts[c] = best_prompt
        tune_records[c] = {
            "all_candidates": [{"prompt": p, "tune_ap": round(s, 4)} for s, p in scores],
            "best_prompt": best_prompt,
            "orig_tune_ap": round(orig_ap, 4),
            "best_tune_ap": round(best_ap, 4),
            "tune_delta": round(best_ap - orig_ap, 4),
        }
        print(f"  --> WINNER for {c}: \"{best_prompt}\" (TUNE AP: {orig_ap:.4f} -> {best_ap:.4f})")

    # -------------------------------------------------------- Evaluation on REPORT
    print("\n" + "=" * 75)
    print("[Step 3] HONEST REPORT HALF EVALUATION (UNSEEN TEST SET)")
    print("=" * 75)

    res_rep_v1 = evaluate_split(V_rep, L_rep, classes, PROMPTS_V1, embedder_open)
    res_rep_v2 = evaluate_split(V_rep, L_rep, classes, chosen_prompts, embedder_open)

    print(f"\n{'Class':24s} | {'V1 AP (Orig)':>12s} | {'V2 AP (Tuned)':>13s} | {'Delta':>8s} | {'Prompt Changed?'}")
    print("-" * 75)
    for c in classes:
        ap1 = res_rep_v1["per_class"][c]["ap"]
        ap2 = res_rep_v2["per_class"][c]["ap"]
        delta = ap2 - ap1
        changed = "YES (Tuned)" if chosen_prompts[c] != PROMPTS_V1[c] else "No"
        print(f"{c:24s} | {ap1:12.4f} | {ap2:13.4f} | {delta:+8.4f} | {changed}")
    print("-" * 75)
    print(f"{'REPORT mAP':24s} | {res_rep_v1['mAP']:12.4f} | {res_rep_v2['mAP']:13.4f} | {res_rep_v2['mAP'] - res_rep_v1['mAP']:+8.4f} |")
    print(f"{'Zero-Shot Accuracy':24s} | {res_rep_v1['zero_shot_acc']*100:11.1f}% | {res_rep_v2['zero_shot_acc']*100:12.1f}% | {(res_rep_v2['zero_shot_acc'] - res_rep_v1['zero_shot_acc'])*100:+7.1f}% |")

    # -------------------------------------------------------- Full Dataset Comparison
    print("\n" + "=" * 75)
    print("[Step 4] FULL DATASET (27,000 Patches) EVALUATION")
    print("         (Reported with explicit disclosure that prompts were selected on TUNE split)")
    print("=" * 75)

    V_full = np.concatenate([V_tune, V_rep])
    L_full = np.concatenate([L_tune, L_rep])

    res_full_v1 = evaluate_split(V_full, L_full, classes, PROMPTS_V1, embedder_open)
    res_full_v2 = evaluate_split(V_full, L_full, classes, chosen_prompts, embedder_open)

    print(f"\n{'Class':24s} | {'Full V1 AP':>12s} | {'Full V2 AP':>13s} | {'Delta':>8s}")
    print("-" * 65)
    for c in classes:
        ap1 = res_full_v1["per_class"][c]["ap"]
        ap2 = res_full_v2["per_class"][c]["ap"]
        print(f"{c:24s} | {ap1:12.4f} | {ap2:13.4f} | {ap2 - ap1:+8.4f}")
    print("-" * 65)
    print(f"{'FULL mAP':24s} | {res_full_v1['mAP']:12.4f} | {res_full_v2['mAP']:13.4f} | {res_full_v2['mAP'] - res_full_v1['mAP']:+8.4f}")
    print(f"{'FULL Zero-Shot Acc':24s} | {res_full_v1['zero_shot_acc']*100:11.1f}% | {res_full_v2['zero_shot_acc']*100:12.1f}% | {(res_full_v2['zero_shot_acc'] - res_full_v1['zero_shot_acc'])*100:+7.1f}%")

    # -------------------------------------------------------- RemoteCLIP Transfer Test
    print("\n" + "=" * 75)
    print("[Step 5] CROSS-MODEL GENERALIZATION: RemoteCLIP on REPORT Split")
    print("=" * 75)

    embedder_remote = emb.Embedder(backend="remoteclip")
    RV_tune, RL_tune, RV_rep, RL_rep = get_embeddings("remoteclip", classes, tune_paths, report_paths)

    rem_rep_v1 = evaluate_split(RV_rep, RL_rep, classes, PROMPTS_V1, embedder_remote)
    rem_rep_v2 = evaluate_split(RV_rep, RL_rep, classes, chosen_prompts, embedder_remote)

    print(f"\n{'Class':24s} | {'RemoteCLIP V1':>13s} | {'RemoteCLIP V2':>13s} | {'Delta':>8s}")
    print("-" * 65)
    for c in classes:
        ap1 = rem_rep_v1["per_class"][c]["ap"]
        ap2 = rem_rep_v2["per_class"][c]["ap"]
        print(f"{c:24s} | {ap1:13.4f} | {ap2:13.4f} | {ap2 - ap1:+8.4f}")
    print("-" * 65)
    print(f"{'RemoteCLIP REPORT mAP':24s} | {rem_rep_v1['mAP']:13.4f} | {rem_rep_v2['mAP']:13.4f} | {rem_rep_v2['mAP'] - rem_rep_v1['mAP']:+8.4f}")

    # -------------------------------------------------------- Save Final Results
    final_results = {
        "methodology": {
            "protocol": "50/50 balanced split per class by filename. Candidates evaluated on TUNE only; reported on REPORT.",
            "n_tune": len(tune_paths),
            "n_report": len(report_paths),
            "n_total": len(tune_paths) + len(report_paths),
        },
        "prompts_v1": PROMPTS_V1,
        "prompts_v2": chosen_prompts,
        "tune_selection": tune_records,
        "openclip": {
            "report_split": {
                "v1_baseline": res_rep_v1,
                "v2_tuned": res_rep_v2,
            },
            "full_dataset": {
                "v1_baseline": res_full_v1,
                "v2_tuned": res_full_v2,
            },
        },
        "remoteclip": {
            "report_split": {
                "v1_baseline": rem_rep_v1,
                "v2_tuned": rem_rep_v2,
            },
        },
    }

    out_file = OUT_DIR / "eurosat_prompt_tuning_results.json"
    out_file.write_text(json.dumps(final_results, indent=2))
    print(f"\nSuccessfully saved detailed results to: {out_file}")
    print("=" * 75)


if __name__ == "__main__":
    main()
