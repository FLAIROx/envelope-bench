# syntax=docker/dockerfile:1.7
# Build (from project root): docker build --build-arg USERNAME="${USER}" --build-arg UID="$(id -u)" --build-arg GID="$(id -g)" -t "${USER}_envelope_gpu" -f dev/Dockerfile .

##############################################
# Builder stage
##############################################
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS builder
ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# --- Base tools ---
RUN --mount=type=cache,id=apt-cache-builder,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lists-builder,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates gnupg2 tini curl git unzip build-essential cmake libbz2-dev flex bison \
 && rm -rf /var/lib/apt/lists/*

# --- uv ---
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# --- Environment vars ---
ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    HOME=/app \
    XDG_DATA_HOME=/opt/uvdata

# --- Dependencies ---
COPY pyproject.toml uv.lock /app/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install 3.12 && \
    uv lock && \
    uv sync --locked --all-groups --no-install-project -p 3.12

# --- Project install ---
COPY src/ /app/src/
COPY README.md /app/README.md
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --all-groups --editable -p 3.12

RUN uv pip install jax[cuda]


##############################################
# Runtime stage
##############################################
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS runtime
ARG USERNAME=dev
ARG UID=1000
ARG GID=1000
ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# --- Base tools ---
RUN --mount=type=cache,id=apt-cache-runtime,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lists-runtime,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates tini curl git unzip gnupg2 build-essential \
 && rm -rf /var/lib/apt/lists/*

# --- uv ---
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# --- Environment ---
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    HF_HUB_CACHE=/cache \
    HF_HOME=/cache \
    XDG_CACHE_HOME=/cache \
    MPLCONFIGDIR=/cache/matplotlib \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    JAX_COMPILATION_CACHE_DIR=/cache/jax_cache \
    HOME=/app \
    XDG_DATA_HOME=/opt/uvdata

VOLUME ["/cache"]


# --- Copy venv and project from builder ---
COPY --from=builder --chown=${UID}:${GID} /opt/venv /opt/venv
COPY --from=builder --chown=${UID}:${GID} /app /app
COPY --from=builder /bin/uv /bin/uvx /bin/
COPY --from=builder --chown=${UID}:${GID} /opt/uvdata/uv /opt/uvdata/uv

# --- Copy entrypoint script ---
COPY dev/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# --- Craftax asset redirection (link to host-mounted /cache) ---
RUN mkdir -p /cache/craftax_assets && \
    mkdir -p /opt/venv/lib/python3.12/site-packages/craftax/craftax/assets && \
    mkdir -p /opt/venv/lib/python3.12/site-packages/craftax/craftax_classic/assets && \
    ln -sf /cache/craftax_assets/texture_cache.pbz2 \
        /opt/venv/lib/python3.12/site-packages/craftax/craftax/assets/texture_cache.pbz2 && \
    ln -sf /cache/craftax_assets/texture_cache_classic.pbz2 \
        /opt/venv/lib/python3.12/site-packages/craftax/craftax_classic/assets/texture_cache_classic.pbz2

# --- Create user and cache dirs ---
RUN set -eux; \
    if ! getent group "${GID}" >/dev/null; then groupadd -g "${GID}" "${USERNAME}"; fi; \
    if ! getent passwd "${UID}" >/dev/null; then useradd -m -u "${UID}" -g "${GID}" -s /bin/bash "${USERNAME}"; fi;
RUN mkdir -p /cache/craftax_assets /cache/matplotlib

USER ${USERNAME}
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/docker-entrypoint.sh"]
CMD ["python"]
