"""OSCD Patch-Level Model Ensemble Evaluation.

Protocol to ensure honest methodology:
  1. Train set: 14 cities (X.TRAIN).
  2. Test set:  10 cities (X.TEST).
  3. Ground truth: a tile is changed if >= 5% of its pixels changed.
  4. Models evaluated (using their best variant 'patch top-10%'):
     - CLIP ViT-B/32
     - RemoteCLIP ViT-B/32
     - DINOv2 ViT-S/14
  5. Calibrate scores using train statistics (z-score: (x - mean_train) / std_train).
  6. Ensembling decision (which models, what blend weight) is made PURELY on TRAIN AUC.
  7. The winning ensemble is then evaluated on the held-out TEST cities to report TEST AUC.
  8. Save complete metrics to results/oscd_ensemble_results.json.
"""

import json
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "env" / "libs"))
sys.path.insert(0, str(ROOT / "env" / "libs_gpu"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import run_oscd as X
from src import metrics as M

OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
IMNET_MEAN, IMNET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def prep(batch, mean, std, size=224):
    x = torch.from_numpy(np.ascontiguousarray(batch[..., ::-1])).permute(0, 3, 1, 2)
    x = x.float().div(255).to(DEV)
    x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False).clamp(0, 1)
    m = torch.tensor(mean, device=DEV).view(1, 3, 1, 1)
    s = torch.tensor(std, device=DEV).view(1, 3, 1, 1)
    return (x - m) / s


def nrm(t):
    return t / t.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def top(m, share=0.1):
    k = max(1, int(round(m.shape[1] * share)))
    return np.sort(m, axis=1)[:, -k:].mean(1)


def make_clip(arch, pretrained=None, remote=False):
    import open_clip
    if remote:
        model, _, _ = open_clip.create_model_and_transforms(arch)
        from huggingface_hub import hf_hub_download
        ck = hf_hub_download("chendelong/RemoteCLIP", f"RemoteCLIP-{arch}.pt")
        model.load_state_dict(torch.load(ck, map_location="cpu"))
    else:
        model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
    model = model.to(DEV).eval()
    model.visual.output_tokens = True

    def fwd(x):
        out = model.visual(x)
        pooled, tokens = out if isinstance(out, (tuple, list)) else (out, None)
        return pooled.float(), tokens.float()

    return fwd, CLIP_MEAN, CLIP_STD


def make_dino(name):
    model = torch.hub.load("facebookresearch/dinov2", name, verbose=False).to(DEV).eval()

    def fwd(x):
        o = model.forward_features(x)
        return o["x_norm_clstoken"].float(), o["x_norm_patchtokens"].float()

    return fwd, IMNET_MEAN, IMNET_STD


def extract_patch_scores(T1, T2):
    cache_path = CACHE_DIR / "oscd_patch_top10_scores.npz"
    if cache_path.exists():
        print(f"Loading cached OSCD patch scores from {cache_path} ...")
        data = np.load(cache_path)
        return {k: data[k] for k in data.files}

    models = [
        ("CLIP ViT-B/32", lambda: make_clip("ViT-B-32", "laion2b_s34b_b79k")),
        ("RemoteCLIP ViT-B/32", lambda: make_clip("ViT-B-32", remote=True)),
        ("DINOv2 ViT-S/14", lambda: make_dino("dinov2_vits14")),
    ]

    scores = {}
    for mname, factory in models:
        print(f"\nExtracting patch features for {mname} on {DEV} ...")
        t0 = time.perf_counter()
        fwd, mean, std = factory()
        raw_maps = []
        with torch.no_grad(), torch.autocast(DEV, enabled=(DEV == "cuda")):
            for i in range(0, len(T1), 64):
                _, k1 = fwd(prep(T1[i:i + 64], mean, std))
                _, k2 = fwd(prep(T2[i:i + 64], mean, std))
                dist = (1 - (nrm(k1) * nrm(k2)).sum(-1)).cpu().numpy()
                raw_maps.append(dist)
        raw_maps = np.concatenate(raw_maps)
        # Patch top-10% score
        scores[mname] = top(raw_maps, share=0.1)
        print(f"  {mname}: computed for {len(T1)} tiles in {time.perf_counter()-t0:.1f}s")
        del fwd
        if DEV == "cuda":
            torch.cuda.empty_cache()

    np.savez_compressed(cache_path, **scores)
    print(f"Saved extracted scores to {cache_path}")
    return scores


def main():
    print("=" * 75)
    print("OSCD ENSEMBLE EVALUATION PIPELINE")
    print("=" * 75)
    print(f"Device: {DEV} ({torch.cuda.get_device_name(0) if DEV == 'cuda' else 'CPU'})")

    print("\n[Step 1] Loading OSCD cities and constructing tile pairs ...")
    cities = {c: X.load_city(c) for c in X.TRAIN + X.TEST}

    T1, T2, loc_city, frac = [], [], [], []
    for c, d in cities.items():
        h, w = d["label"].shape
        for (y, x) in X.tile_grid(h, w):
            T1.append(d["img1"][y:y + X.TILE, x:x + X.TILE])
            T2.append(d["img2"][y:y + X.TILE, x:x + X.TILE])
            loc_city.append(c)
            frac.append(float((d["label"][y:y + X.TILE, x:x + X.TILE] > 0).mean()))

    T1, T2 = np.stack(T1), np.stack(T2)
    frac = np.array(frac)
    ylab = frac >= X.CHANGED_TILE_FRAC
    is_test = np.array([c in X.TEST for c in loc_city])

    n_train = int((~is_test).sum())
    n_test = int(is_test.sum())
    print(f"Total tile pairs: {len(T1)} (Train: {n_train}, Test: {n_test})")
    print(f"Changed tiles: Total {int(ylab.sum())} (Train: {int(ylab[~is_test].sum())}, Test: {int(ylab[is_test].sum())})")

    # Baseline reference: Pixel difference (raw)
    raw_diff = np.array([float(cv2.absdiff(a, b).mean()) for a, b in zip(T1, T2)])
    pix_train_auc = X.roc_auc(raw_diff[~is_test], ylab[~is_test])[0]
    pix_test_auc = X.roc_auc(raw_diff[is_test], ylab[is_test])[0]

    # [Step 2] Extract or load patch scores
    scores = extract_patch_scores(T1, T2)

    # [Step 3] Individual Model Baseline Performances
    print("\n" + "=" * 75)
    print("[Step 2] INDIVIDUAL MODEL PERFORMANCE (patch top-10%)")
    print("=" * 75)
    print(f"{'Model':24s} | {'Train AUC':>10s} | {'Test AUC':>10s} | {'All AUC':>10s}")
    print("-" * 65)
    print(f"{'Pixel Diff (baseline)':24s} | {pix_train_auc:10.4f} | {pix_test_auc:10.4f} | {X.roc_auc(raw_diff, ylab)[0]:10.4f}")

    indiv_res = {}
    for name, s in scores.items():
        tr_auc = X.roc_auc(s[~is_test], ylab[~is_test])[0]
        te_auc = X.roc_auc(s[is_test], ylab[is_test])[0]
        al_auc = X.roc_auc(s, ylab)[0]
        indiv_res[name] = {"train_auc": round(tr_auc, 4), "test_auc": round(te_auc, 4), "all_auc": round(al_auc, 4)}
        print(f"{name:24s} | {tr_auc:10.4f} | {te_auc:10.4f} | {al_auc:10.4f}")

    # Standardize scores using TRAIN statistics only: z = (s - mean_train) / std_train
    z_scores = {}
    for name, s in scores.items():
        m_tr = s[~is_test].mean()
        std_tr = s[~is_test].std() + 1e-7
        z_scores[name] = (s - m_tr) / std_tr

    # [Step 4] Ensemble Selection on TRAIN ONLY
    print("\n" + "=" * 75)
    print("[Step 3] CANDIDATE ENSEMBLES EVALUATED ON TRAIN AUC ONLY")
    print("=" * 75)

    m_names = list(scores.keys())
    cand_ensembles = {}

    # 1. Simple 50/50 pairs (equal z-score blend)
    for i in range(len(m_names)):
        for j in range(i + 1, len(m_names)):
            m1, m2 = m_names[i], m_names[j]
            pair_name = f"{m1} + {m2} (50/50)"
            ens_score = 0.5 * z_scores[m1] + 0.5 * z_scores[m2]
            tr_auc = X.roc_auc(ens_score[~is_test], ylab[~is_test])[0]
            cand_ensembles[pair_name] = {"score": ens_score, "train_auc": tr_auc, "type": "pair_equal", "models": [m1, m2]}

    # 2. Weighted blend grid search on TRAIN for pairs
    for i in range(len(m_names)):
        for j in range(i + 1, len(m_names)):
            m1, m2 = m_names[i], m_names[j]
            best_w, best_tr = 0.5, -1
            for w in np.linspace(0.1, 0.9, 9):
                ens_score = w * z_scores[m1] + (1 - w) * z_scores[m2]
                tr_auc = X.roc_auc(ens_score[~is_test], ylab[~is_test])[0]
                if tr_auc > best_tr:
                    best_tr = tr_auc
                    best_w = w
            w_name = f"{m1} ({best_w:.1f}) + {m2} ({1-best_w:.1f})"
            ens_score = best_w * z_scores[m1] + (1 - best_w) * z_scores[m2]
            cand_ensembles[w_name] = {"score": ens_score, "train_auc": best_tr, "type": "pair_weighted", "models": [m1, m2], "weight": round(best_w, 2)}

    # 3. Trio (all 3 models) equal average
    trio_score = (z_scores[m_names[0]] + z_scores[m_names[1]] + z_scores[m_names[2]]) / 3.0
    trio_tr = X.roc_auc(trio_score[~is_test], ylab[~is_test])[0]
    cand_ensembles["All 3 Models (equal average)"] = {"score": trio_score, "train_auc": trio_tr, "type": "trio", "models": m_names}

    # Print candidates ranked by TRAIN AUC
    sorted_cands = sorted(cand_ensembles.items(), key=lambda x: x[1]["train_auc"], reverse=True)
    for name, d in sorted_cands:
        print(f"  Train AUC: {d['train_auc']:.4f}  |  {name}")

    winning_name, winning_cand = sorted_cands[0]
    print(f"\n--> WINNER SELECTED ON TRAIN AUC: '{winning_name}' (Train AUC: {winning_cand['train_auc']:.4f})")

    # [Step 5] Evaluate on Held-out TEST Split
    print("\n" + "=" * 75)
    print("[Step 4] HONEST EVALUATION ON HELD-OUT TEST CITIES (10 CITIES)")
    print("=" * 75)

    print(f"{'Method / Ensemble':40s} | {'Train AUC':>10s} | {'Test AUC':>10s} | {'Test AP':>9s}")
    print("-" * 75)

    # Pixel diff baseline
    pix_test_ap = X.roc_auc(raw_diff[is_test], ylab[is_test])[1]
    print(f"{'Pixel Diff (baseline)':40s} | {pix_train_auc:10.4f} | {pix_test_auc:10.4f} | {pix_test_ap:9.4f}")

    # Individual models
    for name in m_names:
        s = scores[name]
        tr_a, tr_ap = X.roc_auc(s[~is_test], ylab[~is_test])
        te_a, te_ap = X.roc_auc(s[is_test], ylab[is_test])
        print(f"{name + ' (individual)':40s} | {tr_a:10.4f} | {te_a:10.4f} | {te_ap:9.4f}")

    print("-" * 75)
    # Winning ensemble
    win_s = winning_cand["score"]
    win_tr_a, win_tr_ap = X.roc_auc(win_s[~is_test], ylab[~is_test])
    win_te_a, win_te_ap = X.roc_auc(win_s[is_test], ylab[is_test])
    print(f"{winning_name + ' [WINNER]':40s} | {win_tr_a:10.4f} | {win_te_a:10.4f} | {win_te_ap:9.4f}")

    # Compare also 50/50 pairs on Test for full transparency
    print("\nAll candidate combinations on Test (for complete transparency):")
    for name, d in sorted_cands:
        s = d["score"]
        te_a, te_ap = X.roc_auc(s[is_test], ylab[is_test])
        mark = " <-- CHOSEN ON TRAIN" if name == winning_name else ""
        print(f"  {name:42s} | Train AUC: {d['train_auc']:.4f} | Test AUC: {te_a:.4f} | Test AP: {te_ap:.4f}{mark}")

    # [Step 6] Save Results
    res_to_save = {
        "methodology": {
            "protocol": "All ensemble model choices and weights selected strictly on 14 TRAIN cities; reported on 10 TEST cities.",
            "tile_grid": f"{X.TILE}x{X.TILE} px",
            "threshold_changed_frac": X.CHANGED_TILE_FRAC,
            "n_train_tiles": n_train,
            "n_test_tiles": n_test,
        },
        "baseline_pixel_diff": {
            "train_auc": round(pix_train_auc, 4),
            "test_auc": round(pix_test_auc, 4),
            "test_ap": round(pix_test_ap, 4),
        },
        "individual_models": indiv_res,
        "winning_ensemble": {
            "name": winning_name,
            "train_auc": round(win_tr_a, 4),
            "test_auc": round(win_te_a, 4),
            "test_ap": round(win_te_ap, 4),
            "delta_over_clip": round(win_te_a - indiv_res["CLIP ViT-B/32"]["test_auc"], 4),
            "delta_over_remoteclip": round(win_te_a - indiv_res["RemoteCLIP ViT-B/32"]["test_auc"], 4),
            "delta_over_dinov2": round(win_te_a - indiv_res["DINOv2 ViT-S/14"]["test_auc"], 4),
        },
        "all_candidates": {
            k: {
                "train_auc": round(v["train_auc"], 4),
                "test_auc": round(X.roc_auc(v["score"][is_test], ylab[is_test])[0], 4),
                "test_ap": round(X.roc_auc(v["score"][is_test], ylab[is_test])[1], 4),
            }
            for k, v in cand_ensembles.items()
        },
    }

    out_file = OUT_DIR / "oscd_ensemble_results.json"
    out_file.write_text(json.dumps(res_to_save, indent=2))
    print(f"\nSaved ensemble evaluation results to: {out_file}")
    print("=" * 75)


if __name__ == "__main__":
    main()
