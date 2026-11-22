"""Local photo check that never runs out: CLIP ViT-B/32 on the CPU.

Used only when every cloud vision backend is unavailable (Nov 22: Cloudflare's daily neurons were used up and
Gemini was timing out, so no photo could be checked and the run skipped). CLIP cannot know what an obscure
ruin looks like, so it answers a narrower question - is this a PHOTOGRAPH OF A PLACE (a fort, temple, wall,
ruin, street, landscape, interior, carving ...) and not a map, logo, emblem, flag, drawing, painting,
engraving, document, food, a person or an animal? - and is trusted only for photos that are place-accurate
by provenance: geotagged within ~5 km of the place, or with a title that names it (media._place_pool).

Model: openai/clip-vit-base-patch32 (MIT), the ONNX export Xenova/clip-vit-base-patch32 (MIT), pinned to a
commit and checked by SHA-256 when the Docker image is built (see the Dockerfile). Only the vision tower runs
at run time; the label texts were embedded once with the same export's text tower (``python -m
purffle_shorts.clip_local build <dir>``) and ship as clip_labels.json.
"""

from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache
from pathlib import Path

log = logging.getLogger("purffle")

CLIP_REPO = "Xenova/clip-vit-base-patch32"
CLIP_REV = "6b7c523703ae26e39be4b03774966f8571c7f523"
CLIP_VISION_SHA256 = "fd6e1402a588279d1723c7534d4bcba5bc0b14b47dfab0e46f8c47b8270d7d40"   # onnx/vision_model.onnx
CLIP_TEXT_SHA256 = "3f6571f5bad13a97c469c1622e1cfc4d9aef78b79fdbfcff804ca357bfada8cc"     # onnx/text_model.onnx
LABELS_FILE = Path(__file__).with_name("clip_labels.json")
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
LOGIT_SCALE = 100.0                      # CLIP's learned temperature (exp(4.6052))
# Accept when the "photo of a place" labels together hold at least this share of the probability. Calibrated
# on Nov 22 (29 real Commons photos of Bujang Valley, Fort Zeelandia, Pingyao and Gwalior Fort; 13 wrong).
PLACE_THRESHOLD = 0.5

# group -> labels. "place" and "interior" are photographs of a place; the rest are not what a scene needs.
LABELS: dict[str, list[str]] = {
    "place": [
        "an old fort", "a castle", "a ruined castle", "a stone fortress wall", "a city wall", "a city gate",
        "a Hindu temple", "a Buddhist temple", "a stupa", "a pagoda", "a mosque", "a church", "a palace",
        "ancient ruins", "brick ruins of an ancient temple", "an archaeological site", "a tomb or mausoleum",
        "a tower", "a courtyard of a historic building", "an old town street", "traditional old houses",
        "a colonial building", "a stone carving", "a statue", "a landscape", "mountains", "a valley",
        "a river", "a lake", "a forest", "a cave", "a historic building",
    ],
    "interior": ["the interior of a historic building", "a hall with columns", "a mural on a wall"],
    "graphic": [
        "a map", "a logo", "an emblem", "a coat of arms", "a flag", "a seal or stamp", "a coin",
        "a diagram", "a chart", "a page of text", "a sign with text", "a screenshot", "a postage stamp",
    ],
    "artwork": ["a drawing", "a pencil sketch", "a painting", "an old engraving", "an illustration",
                "a cartoon", "a black and white engraving of a fort", "a lithograph", "an old print",
                "a watercolour painting of a building", "a hand-drawn picture of a building"],
    "other": ["food on a plate", "a person", "a portrait of a person", "a group of people", "a selfie",
              "an animal", "a cat", "a dog", "a car"],
}
PHOTO_GROUPS = ("place", "interior")
TEMPLATES = ("a photo of {}.", "a photograph of {}.")


def model_dir() -> Path:
    return Path(os.environ.get("CLIP_DIR", "/opt/clip"))


def available() -> bool:
    return (model_dir() / "vision_model.onnx").is_file() and LABELS_FILE.is_file()


@lru_cache(maxsize=1)
def _session():
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    return ort.InferenceSession(str(model_dir() / "vision_model.onnx"), opts, providers=["CPUExecutionProvider"])


@lru_cache(maxsize=1)
def _labels():
    import numpy as np
    data = json.loads(LABELS_FILE.read_text(encoding="utf-8"))
    names = [x["label"] for x in data["labels"]]
    groups = [x["group"] for x in data["labels"]]
    emb = np.array([x["embed"] for x in data["labels"]], dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return names, groups, emb


def pixels(image_path):
    """CLIP's preprocessing: shortest side to 224 (bicubic), centre crop 224 x 224, scale, normalise."""
    import numpy as np
    from PIL import Image, ImageOps
    with Image.open(image_path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        w, h = im.size
        k = 224 / min(w, h)
        im = im.resize((max(224, round(w * k)), max(224, round(h * k))), Image.BICUBIC)
        w, h = im.size
        left, top = (w - 224) // 2, (h - 224) // 2
        im = im.crop((left, top, left + 224, top + 224))
        a = np.asarray(im, dtype=np.float32) / 255.0
    a = (a - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    return a.transpose(2, 0, 1)[None]


def classify(image_path) -> dict[str, float]:
    """Probability of each label group for ``image_path`` (softmax over all labels, summed per group), plus
    "top": the single most likely label."""
    import numpy as np
    names, groups, emb = _labels()
    img = _session().run(None, {"pixel_values": pixels(image_path)})[0][0]
    img = img / np.linalg.norm(img)
    logits = LOGIT_SCALE * (emb @ img)
    p = np.exp(logits - logits.max())
    p /= p.sum()
    out: dict = {g: 0.0 for g in LABELS}
    for g, x in zip(groups, p):
        out[g] += float(x)
    out["top"] = names[int(p.argmax())]
    return out


def photo_of_a_place(image_path, exterior: bool = False) -> tuple[bool | None, str]:
    """(verdict, why). True when CLIP sees a photograph of a place (an outside view when ``exterior``); False
    when it sees a graphic, an artwork, food, a person or an animal; None when the model is not installed or
    could not read the image."""
    if not available():
        return None, "CLIP model not installed"
    try:
        c = classify(image_path)
    except Exception as e:  # noqa: BLE001 - a local check that fails is "no answer", never a crash
        log.debug("CLIP check failed for %s: %s", image_path, e)
        return None, f"CLIP failed: {e}"
    photo = c["place"] + (0.0 if exterior else c["interior"])
    why = f"CLIP: {c['top']} (place {photo:.2f})"
    return photo >= PLACE_THRESHOLD, why


# --------------------------------------------------------------------------------- building clip_labels.json
def _bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


class Tokenizer:
    """CLIP's byte-level BPE tokenizer (vocab.json + merges.txt of the pinned export), enough for the plain
    English label prompts: lower-case, split into letter runs / single digits / other symbols, BPE each word
    with the "</w>" end-of-word suffix, wrap in <|startoftext|> ... <|endoftext|>."""

    def __init__(self, folder: Path):
        self.vocab = json.loads((folder / "vocab.json").read_text(encoding="utf-8"))
        merges = (folder / "merges.txt").read_text(encoding="utf-8").splitlines()[1:]
        self.ranks = {tuple(m.split()): i for i, m in enumerate(merges) if m.strip()}
        self.byte = _bytes_to_unicode()

    def _bpe(self, word: str) -> list[str]:
        parts = list(word[:-1]) + [word[-1] + "</w>"]
        while len(parts) > 1:
            pairs = [(self.ranks.get((a, b), 1 << 30), i) for i, (a, b) in enumerate(zip(parts, parts[1:]))]
            rank, i = min(pairs)
            if rank == 1 << 30:
                break
            parts = parts[:i] + [parts[i] + parts[i + 1]] + parts[i + 2:]
        return parts

    def encode(self, text: str) -> list[int]:
        text = re.sub(r"\s+", " ", text).strip().lower()
        ids = [self.vocab["<|startoftext|>"]]
        for w in re.findall(r"'s|'t|'re|'ve|'m|'ll|'d|[^\W\d_]+|\d|[^\s\w]+|_+", text):
            w = "".join(self.byte[b] for b in w.encode("utf-8"))
            ids += [self.vocab[t] for t in self._bpe(w)]
        return ids + [self.vocab["<|endoftext|>"]]


def build_labels(folder: str | Path) -> None:
    """Embed every label (averaged over TEMPLATES) with the pinned text tower in ``folder`` and write
    clip_labels.json. Development only; the result is committed."""
    import hashlib

    import numpy as np
    import onnxruntime as ort
    folder = Path(folder)
    sha = hashlib.sha256((folder / "text_model.onnx").read_bytes()).hexdigest()
    if sha != CLIP_TEXT_SHA256:
        raise SystemExit(f"text_model.onnx sha256 {sha} != pinned {CLIP_TEXT_SHA256}")
    tok = Tokenizer(folder)
    sess = ort.InferenceSession(str(folder / "text_model.onnx"), providers=["CPUExecutionProvider"])
    rows = []
    for group, labels in LABELS.items():
        for label in labels:
            vs = []
            for t in TEMPLATES:
                ids = np.array([tok.encode(t.format(label))], dtype=np.int64)
                v = sess.run(None, {"input_ids": ids})[0][0]
                vs.append(v / np.linalg.norm(v))
            v = np.mean(vs, axis=0)
            v /= np.linalg.norm(v)
            rows.append({"group": group, "label": label, "embed": [round(float(x), 5) for x in v]})
    LABELS_FILE.write_text(json.dumps({"model": f"{CLIP_REPO}@{CLIP_REV}", "text_model_sha256": sha,
                                       "templates": list(TEMPLATES), "labels": rows}, separators=(",", ":")),
                           encoding="utf-8")
    print(f"wrote {len(rows)} labels to {LABELS_FILE}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "build":
        build_labels(sys.argv[2])
    else:
        raise SystemExit("usage: python -m purffle_shorts.clip_local build <folder with text_model.onnx, "
                         "vocab.json, merges.txt>")
