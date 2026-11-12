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

# Local fallback voice (Piper, used when edge-tts is down): en_US "norman" (medium), trained on public-domain
# LibriVox audiobook readings. Pinned to one commit of rhasspy/piper-voices and checked by SHA-256; the build
# fails if either file differs.
ARG PIPER_VOICES_REV=ae9d8352b73c1b1750748843e753a504b00e05b0
ENV PIPER_VOICE=/opt/piper-voices/en_US-norman-medium.onnx
RUN python - <<'EOF'
import hashlib, os, urllib.request
rev = os.environ["PIPER_VOICES_REV"]
base = f"https://huggingface.co/rhasspy/piper-voices/resolve/{rev}/en/en_US/norman/medium/"
files = {"en_US-norman-medium.onnx": "b9739443232a80a59c7d18810dd856899bf16a7964725f5ab81ea49b1351cb71",
         "en_US-norman-medium.onnx.json": "6c2db7f558a4a8deb9fe822583c1c5105f6c4e834dd0f9de8ad17a888ee9fe1d"}
os.makedirs("/opt/piper-voices", exist_ok=True)
for name, want in files.items():
    data = urllib.request.urlopen(base + name, timeout=300).read()
    got = hashlib.sha256(data).hexdigest()
    if got != want:
        raise SystemExit(f"{name}: sha256 {got} != pinned {want}")
    with open(f"/opt/piper-voices/{name}", "wb") as f:
        f.write(data)
print("piper voice ok")
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
