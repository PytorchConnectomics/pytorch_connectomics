# LICONN moe no-GT error analysis: segmentation -> skeletons -> error_analysis.json
# -> split_links.json -> segment_properties/info, for one run directory.
#
# CPU-ONLY and self-contained, like tutorials/liconn_ingest: nothing here runs a
# model. torch and MONAI are present only because `connectomics.metrics` and
# `connectomics.decoding` import them at package level; the CPU wheel keeps that
# to ~200 MB instead of the 18 GB `pytc:gpu` base.
#
# Pins are the BC `pytc` env versions the published ExPID96 S1 run used, so a
# cloud rerun is comparable number for number. kimimaro/edt/cc3d decide the
# skeletons, which decide every caliber and every class.
#
# No gcloud: the container reads and writes local paths only. Stage the run
# directory in and copy results out from the host/VM, which holds credentials.
#
#   podman build -f tutorials/neuron_liconn_moe/error_analysis/Dockerfile -t liconn-ea .
#   podman run --rm -v <run dir>:/data/<name> -v <affinity dir>:/aff \
#     -e LICONN_EA_RUN=/data/<name> -e LICONN_EA_AFFINITY=/aff/raw_x1_ch0-1-2.h5 \
#     liconn-ea all 16
FROM python:3.11-slim

LABEL org.opencontainers.image.title="LICONN moe error analysis" \
      org.opencontainers.image.description="No-GT skeleton error analysis (CPU)"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/workspace \
    MPLBACKEND=Agg

# The second list is the package-__init__ closure found by running every script's
# --help and the tests in a clean venv until nothing was missing: mahotas,
# torchvision, cv2 and zarr are imported on the way to the metric code, and
# kimimaro imports crackle without declaring it, and its parallel path
# (--parallel > 1) lazily imports posix_ipc and psutil, which neither --help nor
# the tests reach -- found by the full-volume parity run. kimimaro's own
# dependency closure is pinned too: an unpinned osteoid resolved to 0.7.3 (BC has
# 0.5.0), whose uint32 vertices crash the parallel path. OpenCV is the headless build
# because this base has no libGL.
RUN python -m pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu "torch==2.6.0" "torchvision==0.21.0" \
    && python -m pip install --no-cache-dir \
      "mahotas==1.4.18" \
      "opencv-python-headless==4.12.0.88" \
      "zarr==3.1.6" \
      "numpy==2.2.6" \
      "scipy==1.17.1" \
      "h5py==3.13.0" \
      "matplotlib==3.10.7" \
      "kimimaro==5.8.0" \
      "crackle-codec==0.36.0" \
      "posix_ipc==1.3.2" \
      "psutil==7.1.3" \
      "osteoid==0.5.0" \
      "dijkstra3d==1.15.2" \
      "fill_voids==2.1.2" \
      "xs3d==1.13.0" \
      "pykdtree==1.4.3" \
      "pathos==0.3.4" \
      "multiprocess==0.70.18" \
      "dill==0.4.0" \
      "ppft==1.7.7" \
      "pox==0.3.6" \
      "tqdm==4.67.1" \
      "click==8.3.1" \
      "connected-components-3d==3.26.1" \
      "fastremap==1.17.7" \
      "edt==3.1.0" \
      "scikit-image==0.25.2" \
      "networkx==3.6.1" \
      "monai==1.5.1" \
      "torchmetrics==1.8.2" \
      "omegaconf==2.3.0" \
      "einops==0.8.1" \
      "pytest==8.3.5"

WORKDIR /workspace
COPY connectomics /workspace/connectomics
COPY tutorials/neuron_liconn_moe/error_analysis /workspace/tutorials/neuron_liconn_moe/error_analysis
# run_volume.sh's `semantic` step calls the catalog CLI one directory up.
COPY tutorials/neuron_liconn_moe/build_semantic_catalog.py /workspace/tutorials/neuron_liconn_moe/
COPY tests/unit/test_continuity_metrics.py tests/unit/test_morphology_metrics.py \
     tests/unit/test_segment_classification.py tests/unit/test_split_links.py \
     /workspace/tests/unit/

# Build-time gates: the whole import closure resolves without a GPU stack, and
# the metric code the catalog is built from passes its own tests.
RUN python -c "import sys; sys.path.insert(0, 'tutorials/neuron_liconn_moe/error_analysis'); \
import kimimaro, cc3d, fastremap, edt, h5py, matplotlib.pyplot; \
import connectomics.metrics.unsupervised.classification, connectomics.metrics.unsupervised.continuity, \
connectomics.metrics.unsupervised.pieces, connectomics.metrics.unsupervised.morphology, \
connectomics.decoding.error_correction.split_links, connectomics.decoding.error_correction.skeletonize; \
import torch; assert not torch.cuda.is_available(); print('imports ok', torch.__version__)"
RUN python -m pytest -q -p no:cacheprovider tests/unit

ENTRYPOINT ["bash", "/workspace/tutorials/neuron_liconn_moe/error_analysis/run_volume.sh"]
CMD ["all", "16"]
