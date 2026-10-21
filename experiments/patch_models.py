"""Frozen encoders that return BOTH the whole-tile embedding and the grid of
patch features. Shared by the LEVIR-CD experiment. Nothing is trained."""

import numpy as np
import torch
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
IMNET_MEAN, IMNET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def prep(batch, mean, std, size=224):
    """uint8 BGR (N,H,W,3) -> normalised float tensor on the device."""
    x = torch.from_numpy(np.ascontiguousarray(batch[..., ::-1])).permute(0, 3, 1, 2)
    x = x.float().div(255).to(DEV)
    if x.shape[-1] != size or x.shape[-2] != size:
        x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False).clamp(0, 1)
    m = torch.tensor(mean, device=DEV).view(1, 3, 1, 1)
    s = torch.tensor(std, device=DEV).view(1, 3, 1, 1)
    return (x - m) / s


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
        pooled, tokens = model.visual(x)
        return pooled.float(), tokens.float()
    return fwd, CLIP_MEAN, CLIP_STD


def make_dino(name):
    model = torch.hub.load("facebookresearch/dinov2", name, verbose=False).to(DEV).eval()

    def fwd(x):
        o = model.forward_features(x)
        return o["x_norm_clstoken"].float(), o["x_norm_patchtokens"].float()
    return fwd, IMNET_MEAN, IMNET_STD


MODELS = [
    ("CLIP ViT-B/32", lambda: make_clip("ViT-B-32", "laion2b_s34b_b79k")),
    ("RemoteCLIP ViT-B/32", lambda: make_clip("ViT-B-32", remote=True)),
    ("CLIP ViT-B/16", lambda: make_clip("ViT-B-16", "laion2b_s34b_b88k")),
    ("DINOv2 ViT-S/14", lambda: make_dino("dinov2_vits14")),
    ("DINOv2 ViT-B/14", lambda: make_dino("dinov2_vitb14")),
]


def nrm(t):
    return t / t.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def encode_pairs(T1, T2, fwd, mean, std, batch=64, keep_cls=False):
    """Returns whole-tile distance, raw patch-distance maps, centred
    patch-distance maps (each tile's mean token removed first, so a change that
    hits the whole tile such as season or light cancels out) and grid size."""
    cls_d, raw, cen, c1s, c2s = [], [], [], [], []
    with torch.no_grad(), torch.autocast(DEV, enabled=(DEV == "cuda")):
        for i in range(0, len(T1), batch):
            c1, k1 = fwd(prep(T1[i:i + batch], mean, std))
            c2, k2 = fwd(prep(T2[i:i + batch], mean, std))
            cls_d.append((1 - (nrm(c1) * nrm(c2)).sum(-1)).cpu().numpy())
            raw.append((1 - (nrm(k1) * nrm(k2)).sum(-1)).cpu().numpy())
            k1c, k2c = k1 - k1.mean(1, keepdim=True), k2 - k2.mean(1, keepdim=True)
            cen.append((1 - (nrm(k1c) * nrm(k2c)).sum(-1)).cpu().numpy())
            if keep_cls:
                c1s.append(nrm(c1).cpu().numpy()); c2s.append(nrm(c2).cpu().numpy())
    raw, cen = np.concatenate(raw), np.concatenate(cen)
    g = int(round(np.sqrt(raw.shape[1])))
    out = {"cls": np.concatenate(cls_d), "raw": raw, "cen": cen, "grid": g}
    if keep_cls:
        out["v1"], out["v2"] = np.concatenate(c1s), np.concatenate(c2s)
    return out


def variants(enc, share=0.1):
    def top(m):
        k = max(1, int(round(m.shape[1] * share)))
        return np.sort(m, axis=1)[:, -k:].mean(1)
    return {
        "whole-tile embedding": enc["cls"],
        "patch mean": enc["raw"].mean(1),
        "patch max": enc["raw"].max(1),
        "patch top-10%": top(enc["raw"]),
        "centred patch mean": enc["cen"].mean(1),
        "centred patch max": enc["cen"].max(1),
        "centred patch top-10%": top(enc["cen"]),
    }
