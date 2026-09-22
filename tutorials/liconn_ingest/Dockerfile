# syntax=docker/dockerfile:1.7
# LICONN preprocessing: ND2 -> uint8 OME-Zarr pyramid at native resolution.
#
# SELF-CONTAINED AND CPU-ONLY, on purpose. The earlier revision layered on
# `pytc:gpu`, which meant an 18 GB CUDA image had to exist before a job that
# never touches a GPU could run -- and it did not exist on the machine this
# stage is actually driven from. Nothing below needs torch, MONAI or CUDA.
#
# SCOPE IS PREPROCESS ONLY. `fetch` (rclone) and `publish`/`verify`/`prune`
# (gcloud) stay on the host, where the Drive and GCS credentials already live;
# putting them in here would mean mounting two credential stores to save
# nothing. The host has rclone and gcloud; what it lacks is `nd2` and `zarr`,
# and that gap is exactly what this image fills.
FROM python:3.11-slim

LABEL org.opencontainers.image.title="LICONN ingest" \
      org.opencontainers.image.description="ND2 -> uint8 OME-Zarr pyramid (CPU)"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/workspace

# NO apt LAYER ON PURPOSE. This build runs behind an egress policy that
# allows pypi.org and files.pythonhosted.org but not deb.debian.org, so an
# `apt-get install` here fails the build outright. Nothing needs it: the
# manylinux wheels for numpy, scipy and scikit-image vendor their own libgomp,
# and the import smoke test below is what proves that rather than assuming it.

# Pinned: this image decides the pixels that land in donglai_public, so an
# unpinned CLAHE or rounding change would silently fork the published set.
RUN python -m pip install --no-cache-dir \
      "numpy==2.1.3" \
      "zarr==3.0.6" \
      "numcodecs==0.15.1" \
      "nd2==0.10.1" \
      "dask==2025.3.0" \
      "scikit-image==0.25.2" \
      "imageio==2.37.0" \
      "h5py==3.13.0"

WORKDIR /workspace
COPY tutorials/liconn_ingest /workspace/tutorials/liconn_ingest
COPY scripts/preprocess_liconn.py /workspace/scripts/preprocess_liconn.py

# Import smoke test: proves the wheels brought their own shared libraries,
# since there is no apt layer to supply them.
RUN python -c "import numpy, zarr, numcodecs, nd2, dask.array, skimage.exposure; \
print('imports ok', numpy.__version__, zarr.__version__, nd2.__version__)"

# Build-time gates. The filename grammar and the spacing arithmetic fail
# silently rather than loudly, and the zarr writer has to reproduce a layout
# that already exists in the bucket -- so both gate the image.
RUN python tutorials/liconn_ingest/run_tests.py
RUN python tutorials/liconn_ingest/ingest.py --help > /dev/null

CMD ["python", "tutorials/liconn_ingest/ingest.py", "--help"]
