"""Depth maps for 2.5D parallax: Depth Anything V2 Small (Apache-2.0) on the CPU through onnxruntime.

The model is the ONNX export of depth-anything/Depth-Anything-V2-Small from onnx-community/depth-anything-v2-small
(Apache-2.0); the Dockerfile downloads it from a pinned commit and checks its SHA-256. (The Base/Large/Giant
models are CC-BY-NC: only Small may be used here.)

maps_for() runs the model in a CHILD process (``python -m purffle_shorts.depth``), like the local voice: it holds
a few hundred MB while loaded, and the render that follows needs that memory. Each map is an 8-bit grayscale
PNG with the image's aspect ratio, 255 = nearest, smoothed so the parallax bends rather than tears at edges.
Anything that fails leaves the image without a map, and render.py falls back to the plain zoom.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

log = logging.getLogger("purffle")

DEPTH_DIR = "/opt/depth"
INPUT_SIDE = 518            # the model's training size: the short side is resized to this
MULTIPLE = 14               # ...and both sides to a multiple of the ViT patch size
MAX_LONG_SIDE = 924         # keep inference fast on a CPU (a 9:16 frame -> 518 x 924)
MAP_LONG_SIDE = 512         # the saved map: smooth anyway, scaled up at render time
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
TIMEOUT = 300


def model_path() -> Path:
    return Path(os.getenv("DEPTH_DIR") or DEPTH_DIR) / "model.onnx"


def available() -> bool:
    return model_path().exists()


def input_size(w: int, h: int) -> tuple[int, int]:
    """(width, height) the model is fed: short side INPUT_SIDE, long side at most MAX_LONG_SIDE, both multiples
    of 14 (DPTImageProcessor's keep_aspect_ratio + ensure_multiple_of)."""
    scale = INPUT_SIDE / min(w, h)
    nw, nh = w * scale, h * scale
    if max(nw, nh) > MAX_LONG_SIDE:
        f = MAX_LONG_SIDE / max(nw, nh)
        nw, nh = nw * f, nh * f
    return max(MULTIPLE, round(nw / MULTIPLE) * MULTIPLE), max(MULTIPLE, round(nh / MULTIPLE) * MULTIPLE)


def to_map(depth, size: tuple[int, int]):
    """Raw model output (relative inverse depth: larger = nearer) -> an 8-bit PIL image of ``size``: robustly
    normalised (2nd-98th percentile), smoothed, 255 = nearest."""
    import numpy as np
    from PIL import Image, ImageFilter
    d = np.asarray(depth, dtype=np.float32).squeeze()
    lo, hi = np.percentile(d, 2), np.percentile(d, 98)
    d = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    img = Image.fromarray((d * 255).astype("uint8"), "L").resize(size, Image.BICUBIC)
    return img.filter(ImageFilter.GaussianBlur(radius=max(2, max(size) // 90)))


def _session():
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.enable_cpu_mem_arena = False
    return ort.InferenceSession(str(model_path()), opts, providers=["CPUExecutionProvider"])


def estimate(sess, src: Path):
    import numpy as np
    from PIL import Image
    im = Image.open(src).convert("RGB")
    w, h = input_size(*im.size)
    x = (np.asarray(im.resize((w, h), Image.BICUBIC), dtype=np.float32) / 255.0 - MEAN) / STD
    x = x.transpose(2, 0, 1)[None].astype(np.float32)
    name = sess.get_inputs()[0].name
    return sess.run(None, {name: x})[0], im.size


def write_maps(pairs: list[tuple[str, str]]) -> None:
    sess = _session()
    for src, dest in pairs:
        try:
            depth, (w, h) = estimate(sess, Path(src))
            f = MAP_LONG_SIDE / max(w, h)
            to_map(depth, (max(8, round(w * f)), max(8, round(h * f)))).save(dest)
        except Exception as e:  # noqa: BLE001 - one bad image only loses its parallax
            print(f"depth failed for {src}: {e}", file=sys.stderr)


def maps_for(images: list[Path], work: Path) -> dict[Path, Path]:
    """{image: depth map} for the images whose map could be made (an empty dict when the model is missing or
    the child process fails: the render then uses the plain zoom)."""
    if not images or not available():
        return {}
    work.mkdir(parents=True, exist_ok=True)
    pairs = [(str(p), str(work / f"depth{i:02d}.png")) for i, p in enumerate(images)]
    try:
        r = subprocess.run([sys.executable, "-m", "purffle_shorts.depth"], input=json.dumps(pairs), text=True,
                           capture_output=True, timeout=TIMEOUT)
    except (OSError, subprocess.SubprocessError) as e:
        log.warning("Depth maps failed (%s); plain zoom instead of parallax", e)
        return {}
    if r.returncode != 0:
        log.warning("Depth maps failed: %s", (r.stderr or "").strip()[-300:])
    out = {Path(s): Path(d) for s, d in pairs if Path(d).exists()}
    if r.stderr.strip():
        log.info("Depth: %s", r.stderr.strip()[-300:])
    return out


if __name__ == "__main__":
    write_maps([tuple(p) for p in json.load(sys.stdin)])
