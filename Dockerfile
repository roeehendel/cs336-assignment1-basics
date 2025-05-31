FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# system tools
RUN apt-get update && apt-get install -y \
    git htop tmux zsh build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Python
RUN curl -Ls https://astral.sh/uv/install.sh | bash
ENV PATH="/root/.local/bin:${PATH}"

# copy dependency files *first* for layer cache
COPY pyproject.toml uv.lock /tmp/

WORKDIR /workspace
RUN uv pip install --system --requirement /tmp/uv.lock