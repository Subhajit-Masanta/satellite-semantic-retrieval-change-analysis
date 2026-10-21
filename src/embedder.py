"""Stage 2a -- the frozen encoder.   OWNER: Person A

Turns tiles into vectors. NOTHING IS TRAINED. The encoder is loaded pretrained,
put in eval mode, and run under torch.no_grad().

Three backends, tried in order:
  1. RemoteCLIP  - remote-sensing tuned, best quality on overhead imagery
  2. OpenCLIP    - generic CLIP, the ablation baseline
  3. Hashing     - deterministic offline fallback so B, C and D can build and
                   test their code before any model weights are downloaded

PS SIH26227 requires model origin and licence to be declared and weights to be
packaged for offline use -- see `provenance()` and MODEL_PROVENANCE below.
"""

import hashlib

import cv2
import numpy as np

EMBED_DIM = 512

MODEL_PROVENANCE = {
    "remoteclip": {
        "name": "RemoteCLIP ViT-B/32",
        "source": "https://huggingface.co/chendelong/RemoteCLIP",
        "licence": "Apache-2.0",
        "trained_on": "remote-sensing image-text pairs (RET-3, DET-10, SEG-4)",
        "note": "inference only; no fine-tuning performed",
    },
    "openclip": {
        "name": "OpenCLIP ViT-B/32 (laion2b_s34b_b79k)",
        "source": "https://github.com/mlfoundations/open_clip",
        "licence": "MIT",
        "trained_on": "LAION-2B natural images",
        "note": "ablation baseline; natural-image domain gap is the point",
    },
    "hash": {
        "name": "Deterministic hashing fallback",
        "source": "local",
        "licence": "n/a",
        "trained_on": "nothing -- not a learned model",
        "note": "DEVELOPMENT ONLY. Never report results from this backend.",
    },
}


class Embedder:
    """Frozen image/text encoder with a stable interface.

    Contract, relied on by every other module:
      encode_images(list[np.ndarray]) -> (n, EMBED_DIM) float32, L2-normalised
      encode_text(str | list[str])    -> (n, EMBED_DIM) float32, L2-normalised
    """

    def __init__(self, backend="auto", device=None):
        self.backend = None
        self.model = None
        self.dim = EMBED_DIM
        order = ["remoteclip", "openclip", "hash"] if backend == "auto" else [backend]
        for name in order:
            if self._try_load(name, device):
                self.backend = name
                break
        if self.backend is None:
            raise RuntimeError(f"could not initialise any backend from {order}")
        if self.backend == "hash":
            print("[embedder] WARNING: hashing fallback active -- development only, "
                  "results from this backend must not be reported.")

    # ---------------------------------------------------------------- loading
    def _try_load(self, name, device):
        if name == "hash":
            return True
        try:
            import torch
            import open_clip
        except ImportError:
            return False
        try:
            self._torch = torch
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            if name == "remoteclip":
                model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32")
                from huggingface_hub import hf_hub_download
                ckpt = hf_hub_download("chendelong/RemoteCLIP", "RemoteCLIP-ViT-B-32.pt")
                model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            else:
                model, _, preprocess = open_clip.create_model_and_transforms(
                    "ViT-B-32", pretrained="laion2b_s34b_b79k")
            self.model = model.to(self.device).eval()       # FROZEN
            for p in self.model.parameters():
                p.requires_grad = False
            self.preprocess = preprocess
            self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
            self.dim = model.visual.output_dim
            return True
        except Exception as exc:                            # noqa: BLE001
            print(f"[embedder] {name} unavailable ({type(exc).__name__}); trying next")
            return False

    # ---------------------------------------------------------------- encoding
    @staticmethod
    def _norm(v):
        return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9, None)

    def encode_images(self, images):
        if self.backend == "hash":
            return self._norm(np.stack([self._hash_image(i) for i in images]))
        from PIL import Image
        torch = self._torch
        batch = torch.stack([
            self.preprocess(Image.fromarray(cv2.cvtColor(i, cv2.COLOR_BGR2RGB)))
            for i in images
        ]).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_image(batch)
        return self._norm(feats.cpu().numpy().astype(np.float32))

    def encode_text(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        if self.backend == "hash":
            return self._norm(np.stack([self._hash_text(t) for t in texts]))
        torch = self._torch
        tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_text(tokens)
        return self._norm(feats.cpu().numpy().astype(np.float32))

    # ------------------------------------------------------- offline fallback
    def _hash_image(self, img):
        """Colour+texture summary projected to EMBED_DIM. Deterministic, cheap,
        and good enough that similar tiles land near each other -- which is all
        Persons B/C/D need to develop against before real weights exist."""
        small = cv2.resize(img, (32, 32)).astype(np.float32) / 255.0
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        feat = np.concatenate([
            small.reshape(-1),
            cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3).reshape(-1),
            cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3).reshape(-1),
        ])
        rng = np.random.default_rng(0)                       # fixed projection
        proj = rng.standard_normal((feat.size, self.dim), dtype=np.float32) / np.sqrt(feat.size)
        return feat.astype(np.float32) @ proj

    def _hash_text(self, text):
        seed = int(hashlib.sha256(text.lower().encode()).hexdigest()[:8], 16)
        return np.random.default_rng(seed).standard_normal(self.dim, dtype=np.float32)

    # ------------------------------------------------------------- provenance
    def provenance(self):
        """Report for the paper's Experimental Setup section (PS requirement)."""
        return {"backend": self.backend, "dim": self.dim, **MODEL_PROVENANCE[self.backend]}


def embed_tiles(tiles, embedder=None, batch_size=32):
    """Embed a list of Tile objects (or raw arrays). Returns (n, dim) float32."""
    embedder = embedder or Embedder()
    images = [t.image if hasattr(t, "image") else t for t in tiles]
    out = [embedder.encode_images(images[i:i + batch_size])
           for i in range(0, len(images), batch_size)]
    return np.concatenate(out) if out else np.zeros((0, embedder.dim), np.float32)
