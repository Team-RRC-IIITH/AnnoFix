# AnnoFix in a container, shown in the browser (noVNC): a virtual screen (Xvfb) with a small
# window manager, shared over VNC and served as a web page on port 6080.
#   DEVICE=cpu (default): PyTorch from its CPU-only index, ~3 GB image; SAM runs on the CPU.
#   DEVICE=gpu: PyTorch with its CUDA libraries, ~8 GB; needs an NVIDIA GPU and the NVIDIA
#               Container Toolkit on the host.
# Versions come from uv.lock, so the image runs what the tests ran.
FROM python:3.12-slim-bookworm

ARG DEVICE=cpu
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates \
        xvfb x11vnc novnc websockify openbox \
        libxcb-cursor0 libxkbcommon-x11-0 libxkbcommon0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
        libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-xinerama0 libxcb-xfixes0 libxcb-xkb1 \
        libxcb-shm0 libx11-xcb1 libxrender1 libxi6 libsm6 libice6 \
        libgl1 libegl1 libglib2.0-0 libfontconfig1 libdbus-1-3 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/* \
    && printf '<meta http-equiv="refresh" content="0; url=vnc.html?autoconnect=1&resize=scale&reconnect=1">\n' \
        > /usr/share/novnc/index.html

COPY --from=ghcr.io/astral-sh/uv:0.11.13 /uv /usr/local/bin/uv
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1

# Dependencies first (their own layer: editing AnnoFix's code does not reinstall them).
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN set -eux; \
    uv venv --python /usr/local/bin/python3.12 "$VIRTUAL_ENV"; \
    uv export --frozen --no-dev --no-emit-project --no-hashes -o /tmp/locked.txt; \
    if [ "$DEVICE" = cpu ]; then \
        uv pip install --index-url https://download.pytorch.org/whl/cpu \
            "$(grep -E '^torch==' /tmp/locked.txt | cut -d' ' -f1)" \
            "$(grep -E '^torchvision==' /tmp/locked.txt | cut -d' ' -f1)"; \
        grep -v -E '^(torch|torchvision|triton|nvidia-|cuda-)' /tmp/locked.txt > /tmp/rest.txt; \
    else \
        cp /tmp/locked.txt /tmp/rest.txt; \
    fi; \
    grep -v -E '^sam-2 ' /tmp/rest.txt > /tmp/deps.txt; \
    uv pip install -r /tmp/deps.txt; \
    # SAM 2 builds against the torch installed above (its own build would fetch another torch).
    SAM2_BUILD_CUDA=0 uv pip install --no-build-isolation --no-deps "$(grep -E '^sam-2 ' /tmp/rest.txt | cut -d' ' -f1-3)"; \
    rm /tmp/*.txt

COPY annofix ./annofix
COPY docker/start.sh docker/openbox.xml ./docker/
ENV PYTHONPATH=/app

# Runs as any user id (compose passes yours, so the fixes you save belong to you). SAM's
# weights download into HOME once and stay in a volume.
RUN mkdir -p /home/annofix/.cache/huggingface && chmod -R 0777 /home/annofix && chmod +x /app/docker/start.sh
ENV HOME=/home/annofix

EXPOSE 6080
ENTRYPOINT ["/app/docker/start.sh"]
