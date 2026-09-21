# syntax=docker/dockerfile:1.7
# Reuse the tested SNEMI image, including pinned MedNeXt and compiled ABISS.
# Load its verified GCS archive first, or build the preceding tutorial image.
ARG PYTC_IMAGE=pytc:snemi-abiss
FROM ${PYTC_IMAGE}
# Scripts under tutorials/ must import this copied checkout, not the older
# package installed in the base image's site-packages directory.
ENV PYTHONPATH=/workspace
USER root
COPY connectomics /workspace/connectomics
COPY scripts /workspace/scripts
COPY tutorials /workspace/tutorials
USER 1000:1000
RUN python -m connectomics.runtime.dispim_benchmark --help
RUN python tutorials/dispim_gcloud/smoke_cpu.py
CMD ["python", "-m", "connectomics.runtime.dispim_benchmark", "--help"]
