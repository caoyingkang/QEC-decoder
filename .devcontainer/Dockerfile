FROM debian:bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        git-lfs \
        curl \
        ca-certificates \
        build-essential \
        pkg-config \
        python3 \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

SHELL ["/bin/bash", "-lc"]
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
