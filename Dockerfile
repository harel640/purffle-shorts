# PurffleShorts — run the autopilot in a container.
#   docker build -t purffle-shorts .
#   docker run --rm -it --env-file .env -v "$PWD:/work" purffle-shorts run
# Authorize YouTube once on your computer (python -m purffle_shorts auth) so token.json exists in /work.
# Studio in a container:  docker run --rm -it --env-file .env -v "$PWD:/work" -p 127.0.0.1:8765:8765 purffle-shorts \
#                           studio --host 0.0.0.0 --no-browser     (then open http://127.0.0.1:8765)
#   Publish the port as -p 127.0.0.1:8765:8765 so the Studio is only reachable from this computer.
FROM python:3.12-slim
LABEL org.opencontainers.image.source="https://github.com/Chamanrajragu/purffle-shorts" \
      org.opencontainers.image.description="Free AI YouTube Shorts generator and upload autopilot" \
      org.opencontainers.image.licenses="MIT"

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core espeak-ng \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Install dependencies from the hash-locked requirements only, so the image can never
# resolve an unpinned version. Then install the package itself without re-resolving
# dependencies (--no-deps): everything it needs is already pinned above.
COPY requirements.txt ./
RUN pip install --no-cache-dir --require-hashes -r requirements.txt

# Local fallback voice (used when edge-tts is down): Kokoro-82M, Apache-2.0 - the ONNX export from
# onnx-community/Kokoro-82M-v1.0-ONNX (Apache-2.0) with its am_michael voice, and the phoneme vocabulary from
# hexgrad/Kokoro-82M's config.json. Each file is pinned to a commit and checked by SHA-256; the build fails if
# one differs.
ARG KOKORO_ONNX_REV=1939ad2a8e416c0acfeecc08a694d14ef25f2231
ARG KOKORO_REV=f3ff3571791e39611d31c381e3a41a3af07b4987
ENV KOKORO_DIR=/opt/kokoro
RUN python - <<'EOF'
import hashlib, os, urllib.request
onnx = f"https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/{os.environ['KOKORO_ONNX_REV']}/"
base = f"https://huggingface.co/hexgrad/Kokoro-82M/resolve/{os.environ['KOKORO_REV']}/"
files = {
    "model.onnx": (onnx + "onnx/model.onnx", "8fbea51ea711f2af382e88c833d9e288c6dc82ce5e98421ea61c058ce21a34cb"),
    "voices/am_michael.bin": (onnx + "voices/am_michael.bin",
                              "1d1f21dd8da39c30705cd4c75d039d265e9bc4a2a93ed09bc9e1b1225eb95ba1"),
    "config.json": (base + "config.json", "5abb01e2403b072bf03d04fde160443e209d7a0dad49a423be15196b9b43c17f"),
}
for name, (url, want) in files.items():
    dest = os.path.join("/opt/kokoro", name)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    h = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=600) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            h.update(chunk)
            f.write(chunk)
    if h.hexdigest() != want:
        raise SystemExit(f"{name}: sha256 {h.hexdigest()} != pinned {want}")
print("kokoro model ok")
EOF

# 2.5D parallax: Depth Anything V2 SMALL (Apache-2.0; the larger sizes are CC-BY-NC), ONNX export from
# onnx-community/depth-anything-v2-small (Apache-2.0). Ambient beds: CC0 / public-domain recordings from
# Wikimedia Commons (authors and licences in purffle_shorts/ambience.py). Every file is checked by SHA-256.
ARG DEPTH_REV=4472b7362082ad9968fee890ca0f1e5aca36b93d
ENV DEPTH_DIR=/opt/depth AMBIENCE_DIR=/opt/ambience
RUN python - <<'EOF'
import hashlib, os, urllib.request
up = "https://upload.wikimedia.org/wikipedia/commons/"
files = {
    "/opt/depth/model.onnx": (
        f"https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/{os.environ['DEPTH_REV']}/onnx/model.onnx",
        "afb6a5c28f3b6bf1618c6e43f02073ef9dfdc70e937502d51603e57b0a1df10c"),
    "/opt/ambience/temple.ogg": (up + "a/a6/Windglockenspiel.Koshi.ogg",
                                 "1f8b33ad604ef9ff5ba1ad901401b6d182f9db94e28f62a5ed930ff2f281b741"),
    "/opt/ambience/stream.wav": (up + "5/54/433589_jackthemurray_stream-river-water-up-close.wav",
                                 "be201d2fd0d110fc3ef5d8af060e5ac48439fd54e9f26ce6b69825182748a95b"),
    "/opt/ambience/sea.ogg": (up + "6/64/Ocean_Waves_on_a_Tropical_Beach.ogg",
                              "bae2fc0a0c1d36f43494883d9455ad151fc747a0770876b25436422ea8736e55"),
    "/opt/ambience/forest.ogg": (up + "0/0a/20090610_0_ambience.ogg",
                                 "43848c3eda5f42829f1033112c2a91ba3e5c91b79a0532fda264c9f59856d431"),
}
ua = {"User-Agent": "PurffleShorts-build/1.0 (+https://github.com/Chamanrajragu/purffle-shorts)"}
for dest, (url, want) in files.items():
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    h = hashlib.sha256()
    with urllib.request.urlopen(urllib.request.Request(url, headers=ua), timeout=600) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            h.update(chunk)
            f.write(chunk)
    if h.hexdigest() != want:
        raise SystemExit(f"{dest}: sha256 {h.hexdigest()} != pinned {want}")
print("depth model and ambience ok")
EOF

# Local vision fallback (used only when every cloud vision backend is unavailable): CLIP ViT-B/32 - the
# vision tower of openai/clip-vit-base-patch32 (MIT) as exported by Xenova/clip-vit-base-patch32 (MIT), pinned
# to a commit and checked by SHA-256 (the same pin as purffle_shorts/clip_local.py). The label embeddings ship
# in the package (clip_labels.json), so the text tower is not needed at run time.
ARG CLIP_REV=6b7c523703ae26e39be4b03774966f8571c7f523
ENV CLIP_DIR=/opt/clip
RUN python - <<'EOF'
import hashlib, os, urllib.request
url = f"https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/{os.environ['CLIP_REV']}/onnx/vision_model.onnx"
want = "fd6e1402a588279d1723c7534d4bcba5bc0b14b47dfab0e46f8c47b8270d7d40"
os.makedirs("/opt/clip", exist_ok=True)
h = hashlib.sha256()
with urllib.request.urlopen(url, timeout=900) as r, open("/opt/clip/vision_model.onnx", "wb") as f:
    while chunk := r.read(1 << 20):
        h.update(chunk)
        f.write(chunk)
if h.hexdigest() != want:
    raise SystemExit(f"clip vision_model.onnx: sha256 {h.hexdigest()} != pinned {want}")
print("clip model ok")
EOF
COPY pyproject.toml README.md LICENSE ./
COPY purffle_shorts ./purffle_shorts
RUN pip install --no-cache-dir --no-deps .

# Run as an unprivileged user. /work holds the generated videos, the data dir and the
# OAuth token at runtime (mount a host folder over it), so it must be writable by appuser.
RUN useradd --create-home --uid 10001 appuser \
 && mkdir -p /work \
 && chown -R appuser:appuser /app /work

EXPOSE 8765

WORKDIR /work
USER appuser
ENTRYPOINT ["purffle-shorts"]
CMD ["run"]
