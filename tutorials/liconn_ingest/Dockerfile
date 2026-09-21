# syntax=docker/dockerfile:1.7
# LICONN ingest: ND2 -> uint8 OME-Zarr at native resolution -> GCS.
#
# CPU-only. Layered on the repo's own base so there is one image lineage, the
# same way tutorials/dispim_gcloud reuses pytc:snemi-abiss. This stage needs
# neither a GPU nor ABISS, so PYTC_IMAGE can be pointed at a slimmer base once
# one exists -- nothing below assumes CUDA.
ARG PYTC_IMAGE=pytc:gpu
FROM ${PYTC_IMAGE}

# Scripts under tutorials/ must import this copied checkout, not the older
# package in the base image's site-packages.
ENV PYTHONPATH=/workspace

USER root
# rclone is the Drive client; nd2/zarr cover the read and write ends.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl unzip ca-certificates \
 && curl -fsSL https://rclone.org/install.sh | bash \
 && apt-get purge -y unzip && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir "zarr>=3" "nd2" "scikit-image"

COPY connectomics /workspace/connectomics
COPY scripts      /workspace/scripts
COPY tutorials    /workspace/tutorials
USER 1000:1000

# Build-time smoke test. The filename grammar and the spacing arithmetic are the
# parts that fail silently rather than loudly, so they gate the image.
RUN python tutorials/liconn_ingest/run_tests.py
RUN python tutorials/liconn_ingest/ingest.py --help > /dev/null

CMD ["python", "tutorials/liconn_ingest/ingest.py", "--help"]
