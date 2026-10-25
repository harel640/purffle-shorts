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
