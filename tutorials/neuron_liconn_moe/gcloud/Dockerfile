# LICONN moe pipeline image: prepare -> affinity -> ABISS -> precomputed, in one
# container. Built on a disposable builder VM by `build.sh`, on top of
# `docker/Dockerfile` from this same source checkout.
#
# NO `# syntax=` DIRECTIVE AND NO HEREDOC `RUN`, deliberately. The builder VM
# installs Debian's `docker.io`, which ships no BuildKit, so `build.sh` runs
# `DOCKER_BUILDKIT=0` — the same thing the proven diSPIM cloud build did.
# Heredoc `RUN <<EOF` is BuildKit-only and would not parse there, which is why
# the build-time smoke test is the separate file `smoke_image.py`.
#
# WHAT THIS LAYER ADDS, and why none of it is in the base image:
#   * the ABISS `ws` binary. `sweep_merge_threshold.py` shells out to it via
#     `scripts/run_abiss_volume.py --abiss-home`; on BC that is a vendored build
#     under `lib/abiss`, which is a compiled artifact and is not in git.
#   * zarr, cloud-volume, igneous. `prepare_volume.py` reads OME-Zarr and
#     `upload_seg_precomputed.py` builds the layer and meshes, but none of the
#     three is a `pyproject.toml` dependency — the core package does not need
#     them, so the base image does not carry them.
#   * huggingface_hub, to pull the `pytc/liconn` checkpoint.
#
# `gcloud` is deliberately NOT installed. The container does compute on local
# paths only; staging in and publishing out are the host's and the VM's job,
# which keeps credentials out of every layer.
ARG PYTC_IMAGE=pytc:gpu
FROM ${PYTC_IMAGE}

# Scripts under tutorials/ must import this copied checkout, not an older
# package in the base image's site-packages.
ENV PYTHONPATH=/workspace

USER root
ARG PYTC_UID=1000
ARG PYTC_GID=1000
# Same revision as tutorials/neuron_snemi_gcloud/Dockerfile: two cloud images
# must not decode with different ABISS builds.
ARG ABISS_REF=452efa5f87f9d3cb241891ee44010d966a33b316
LABEL org.pytorchconnectomics.abiss.revision="${ABISS_REF}"

# Only the single-volume C++ `ws` target. The prepared cube is 213 Mvoxel —
# 9.9% of ABISS's uint32 watershed index cap, ~15 GB peak RSS — so the chunked
# CloudVolume worker stack is not needed. See README "The grid, and why one VM".
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        cmake libboost-iostreams-dev libboost-system-dev libtbb-dev \
    && rm -rf /var/lib/apt/lists/* \
    && git init /opt/abiss \
    && git -C /opt/abiss remote add origin https://github.com/PytorchConnectomics/ABISS.git \
    && git -C /opt/abiss fetch --depth 1 origin "${ABISS_REF}" \
    && git -C /opt/abiss checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/abiss rev-parse HEAD)" = "${ABISS_REF}" \
    && cmake -S /opt/abiss -B /opt/abiss/build \
        -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
        -DABISS_ARCH=x86-64 -DABISS_ALLOCATOR=system \
        -DCMAKE_DISABLE_FIND_PACKAGE_absl=TRUE \
    && cmake --build /opt/abiss/build --target ws --parallel 4 \
    && test -x /opt/abiss/build/ws

# `run_abiss_volume.py --abiss-home` expects a checkout with `build/ws` under
# it, which is the layout above. `sweep_merge_threshold.py` reads this variable
# before falling back to the vendored BC path, which does not exist here.
ENV ABISS_HOME=/opt/abiss

# `click` is pinned to `spin`'s constraint, which the base image already
# carries (spin is a scientific-Python DEVELOPER tool, pulled in transitively;
# it has no runtime role here, but leaving its metadata unsatisfiable means
# `pip check` can never pass again). cloud-files requires bare `click` and
# igneous `click>=6.7`, so 8.2.x satisfies all three. Without the pin,
# cloud-volume drags in click 8.5 and `pip check` fails the build -- measured,
# builder liconn-build-liconn-moe-v1-20260921-190303.
RUN python -m pip install --no-cache-dir \
        "zarr>=3.0" \
        "cloud-volume>=11" \
        "igneous-pipeline>=4" \
        "huggingface_hub>=0.34" \
        "click>=8,!=8.3.0,<8.4" \
    && python -m pip check

# Refresh the checkout on top of the base image's copy, so an image rebuilt
# from an existing base still carries this revision of the tutorial scripts.
COPY connectomics /workspace/connectomics
COPY scripts /workspace/scripts
COPY tutorials /workspace/tutorials

USER ${PYTC_UID}:${PYTC_GID}

RUN python tutorials/neuron_liconn_moe/gcloud/smoke_image.py

WORKDIR /workspace
CMD ["bash", "tutorials/neuron_liconn_moe/gcloud/run_volume.sh", "--help"]
