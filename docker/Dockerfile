# syntax=docker/dockerfile:1.7

# CUDA 12.6 keeps compatibility with a wider range of NVIDIA GPUs than the
# CUDA 13 images while still using the current stable PyTorch release.
ARG PYTORCH_IMAGE=pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime@sha256:6acf597eeb8e376a96580dde4952f37cc017fef732bb40bfc73f28f25e3f64b4
FROM ${PYTORCH_IMAGE}

ARG PYTORCH_IMAGE
ARG DEBIAN_FRONTEND=noninteractive
ARG PYTC_UID=1000
ARG PYTC_GID=1000
ARG MEDNEXT_REF=ddf7398815d9aa39904e732b8d6fa18b160c9df7
ARG WATERZ_REF=3841cf5adf8b03e81b6f566c439eeee479fdbf27

LABEL org.opencontainers.image.title="PyTorch Connectomics" \
      org.opencontainers.image.description="GPU training environment for PyTorch Connectomics" \
      org.opencontainers.image.source="https://github.com/zudi-lin/pytorch_connectomics" \
      org.opencontainers.image.base.name="${PYTORCH_IMAGE}"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    HOME=/home/pytc \
    XDG_CACHE_HOME=/home/pytc/.cache \
    MPLCONFIGDIR=/home/pytc/.cache/matplotlib

# build-essential covers packages that do not publish a wheel for every
# supported Python version. The shared libraries are required by OpenCV and
# OpenMP-backed scientific packages at runtime.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        libboost-dev \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace

# The base image already provides a CUDA-enabled torch/torchvision pair.
# SNEMI and diSPIM tutorials use the external MedNeXt and waterz packages. Pin
# both because the project metadata intentionally leaves them as optional
# manual installations.
# Installing the project normally leaves the compatible torch pair in place
# while resolving the remaining runtime dependencies from pyproject.toml.
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install \
        "witty[cython]==0.3.1" \
        "git+https://github.com/PytorchConnectomics/MedNeXt.git@${MEDNEXT_REF}" \
        "git+https://github.com/PytorchConnectomics/waterz.git@${WATERZ_REF}" \
    && python -m pip install . \
    && python -m pip check

# Match the common first-user UID/GID on Linux GPU VMs. Recent PyTorch images
# already contain an Ubuntu user with UID/GID 1000, so reuse an existing numeric
# identity and only create the account or group when the requested IDs are free.
RUN if ! getent group "${PYTC_GID}" >/dev/null; then \
        groupadd --gid "${PYTC_GID}" pytc; \
    fi \
    && if ! getent passwd "${PYTC_UID}" >/dev/null; then \
        useradd \
            --create-home \
            --gid "${PYTC_GID}" \
            --shell /bin/bash \
            --uid "${PYTC_UID}" \
            pytc; \
    fi \
    && mkdir -p /data /workspace/outputs "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}" \
    && chown -R "${PYTC_UID}:${PYTC_GID}" \
        /data /workspace/outputs /home/pytc

USER ${PYTC_UID}:${PYTC_GID}

# Keep the image useful for arbitrary scripts while making a bare run
# self-documenting.
CMD ["python", "scripts/main.py", "--help"]
