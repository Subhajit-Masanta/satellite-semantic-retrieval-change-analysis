"""Text search on LEVIR-CD (0.5 m per pixel).

    python experiments/run_levir_text.py

LEVIR-CD has change masks only (no land-cover labels), so the text test is:
does a text query about new buildings rank the tiles that really changed?
Score of a tile = sim(date-2 tile, query) - sim(date-1 tile, query), all in the
same CLIP space used for search. Prompts are picked on VAL, reported on TEST.
Same-place image search is already in results/levir_results.json.

Saves results/levir_text_results.json.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
os.environ["RUN_LOG"] = "levir_text_log.txt"   # run_levir opens this on import; keep its own log intact
sys.path.insert(0, str(ROOT / "experiments"))
import patch_models as PM                       # noqa: E402
import run_levir as RL                          # noqa: E402

OUT = ROOT / "results"
PROMPTS = [
    "a satellite image of buildings",
    "a satellite image of houses",
    "a satellite image of residential buildings",
    "an aerial image of new houses",
    "a satellite image of a dense residential area with roofs",
    "an aerial image of rooftops",
]
BARE = "a satellite image of bare land"
R = {"tile_px": RL.TILE, "changed_if_fraction_at_least": RL.CHANGED, "prompts": PROMPTS}


def load(remote):
    import open_clip
    if remote:
        model, _, _ = open_clip.create_model_and_transforms("ViT-B-32")
        from huggingface_hub import hf_hub_download
        ck = hf_hub_download("chendelong/RemoteCLIP", "RemoteCLIP-ViT-B-32.pt")
        model.load_state_dict(torch.load(ck, map_location="cpu"))
    else:
        model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    return model.to(PM.DEV).eval(), open_clip.get_tokenizer("ViT-B-32")


@torch.no_grad()
def embed_images(model, T, batch=64):
    out = []
    for i in range(0, len(T), batch):
        f = model.encode_image(PM.prep(T[i:i + batch], PM.CLIP_MEAN, PM.CLIP_STD))
        out.append(PM.nrm(f.float()).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def embed_text(model, tok, texts):
    f = model.encode_text(tok(texts).to(PM.DEV)).float()
    return PM.nrm(f).cpu().numpy()


def main():
    t0 = time.perf_counter()
    val, test = RL.load_split("val"), RL.load_split("test")
    V1, V2, vfr, _ = RL.tiles_of(val)
    S1, S2, sfr, _ = RL.tiles_of(test)
    vy, sy = vfr >= RL.CHANGED, sfr >= RL.CHANGED
    R["tiles"] = {"val": len(V1), "test": len(S1), "test_changed": int(sy.sum())}
    print(f"tiles val {len(V1)}, test {len(S1)} ({int(sy.sum())} changed)")

    for name, remote in (("OpenCLIP ViT-B/32", False), ("RemoteCLIP ViT-B/32", True)):
        model, tok = load(remote)
        ev1, ev2, es1, es2 = (embed_images(model, X) for X in (V1, V2, S1, S2))
        T = embed_text(model, tok, PROMPTS)
        bare = embed_text(model, tok, [BARE])[0]
        res = {}
        for i, p in enumerate(PROMPTS):
            # plain: rise in similarity to the query; contrast: query minus "bare land"
            q2 = T[i] - bare
            res[p] = {
                "plain": {"val": RL.auc(ev2 @ T[i] - ev1 @ T[i], vy),
                          "test": RL.auc(es2 @ T[i] - es1 @ T[i], sy)},
                "vs_bare_land": {"val": RL.auc(ev2 @ q2 - ev1 @ q2, vy),
                                 "test": RL.auc(es2 @ q2 - es1 @ q2, sy)},
            }
        pick = max(((p, k) for p in res for k in res[p]), key=lambda pk: res[pk[0]][pk[1]]["val"])
        # image-only reference in the same space: 1 - cos(date 1, date 2)
        ref = {"val": RL.auc(1 - (ev1 * ev2).sum(1), vy), "test": RL.auc(1 - (es1 * es2).sum(1), sy)}
        # top-50 precision of the picked text query on test
        p, k = pick
        q = T[PROMPTS.index(p)] - (bare if k == "vs_bare_land" else 0)
        top = np.argsort(-(es2 @ q - es1 @ q))[:50]
        R[name] = {"per_prompt": res, "picked_on_val": {"prompt": p, "variant": k},
                   "picked_test_auc": res[p][k]["test"],
                   "picked_test_precision@50": round(float(sy[top].mean()), 4),
                   "image_only_whole_tile_auc": ref}
        print(f"{name}: picked '{p}' ({k}) val {res[p][k]['val']:.3f} test {res[p][k]['test']:.3f}"
              f"  P@50 {sy[top].mean():.2f}  | image-only whole tile test {ref['test']:.3f}")
        del model
        if PM.DEV == "cuda":
            torch.cuda.empty_cache()

    R["random_precision"] = round(float(sy.mean()), 4)
    R["runtime_s"] = round(time.perf_counter() - t0, 1)
    (OUT / "levir_text_results.json").write_text(json.dumps(R, indent=1))
    print("saved results/levir_text_results.json")


if __name__ == "__main__":
    main()
