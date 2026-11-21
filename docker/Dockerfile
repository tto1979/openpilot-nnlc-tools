FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies (includes capnp headers for pycapnp)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    libcapnp-dev \
    capnproto \
    curl \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app

# Upgrade pip for PEP 621 (pyproject.toml) support
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip

# Julia install via juliaup (installs to ~/.juliaup by default)
ENV JULIA_DEPOT_PATH=/usr/local/julia-depot
ENV PATH="/root/.juliaup/bin:${PATH}"
RUN curl -fsSL https://install.julialang.org | sh -s -- -y --default-channel 1.10

# Julia packages (changes rarely — slow layer, cached)
COPY training/install_packages.jl training/install_packages.jl
RUN julia training/install_packages.jl

# There is no GPU while building, so CUDA.jl can't pick a CUDA runtime by itself ("CUDA.jl's JLLs were precompiled
# without an NVIDIA driver present"). Pin one - 12.9 runs RTX 50 cards (needs >= 12.8) on CUDA 12 and 13 drivers -
# then fetch its artifacts and precompile in a fresh process, so `docker run` has everything.
# (cuDNN stays installed but isn't loaded here: CUDNN_jll needs the driver's libcuda.so.1 to load, so it compiles
# on first use at run time.)
RUN julia -e 'using CUDA; CUDA.set_runtime_version!(v"12.9")' \
 && julia -e 'import Pkg; Pkg.instantiate(); using CUDA, Flux'

# Application code (changes often — fast rebuild)
COPY . .

# Install nnlc_tools package (with dev deps for testing)
RUN pip install --no-cache-dir ".[dev]"

# Volume mount points
VOLUME ["/app/data", "/app/output"]

CMD ["bash"]
