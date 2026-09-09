# syntax=docker/dockerfile:1.7
# Build docker/Dockerfile as pytc:gpu first, from this same source checkout.
ARG PYTC_IMAGE=pytc:gpu
FROM ${PYTC_IMAGE}

USER root
ARG PYTC_UID=1000
ARG PYTC_GID=1000
ARG ABISS_REF=452efa5f87f9d3cb241891ee44010d966a33b316
LABEL org.pytorchconnectomics.abiss.revision="${ABISS_REF}"

# Only the single-volume C++ ws target is needed: no CloudVolume worker stack.
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
    && cmake --build /opt/abiss/build --target ws --parallel 2 \
    && test -x /opt/abiss/build/ws

USER ${PYTC_UID}:${PYTC_GID}

# Fail the build if the real C++ decoder, affinity layout, HDF5 stage boundary,
# or challenge evaluator cannot recover two perfect synthetic objects on CPU.
RUN python - <<'PY'
import argparse
import tempfile
from pathlib import Path

import numpy as np
import yaml

from connectomics.data.io import write_hdf5
from connectomics.runtime.snemi_benchmark import run_benchmark

with tempfile.TemporaryDirectory(prefix="snemi_abiss_build_") as folder:
    root = Path(folder)
    labels = np.ones((16, 32, 32), dtype=np.uint32)
    labels[:, :, 16:] = 2
    affinities = np.zeros((3, *labels.shape), dtype=np.float32)
    for channel, axis in enumerate((2, 1, 0)):
        current = [slice(None)] * 3
        previous = [slice(None)] * 3
        current[axis] = slice(1, None)
        previous[axis] = slice(None, -1)
        affinities[channel][tuple(current)] = labels[tuple(current)] == labels[tuple(previous)]
    write_hdf5(str(root / "labels.h5"), labels)
    write_hdf5(str(root / "affinities.h5"), affinities)
    config = yaml.safe_load(Path("tutorials/neuron_snemi_gcloud/neuron_snemi_gcloud.yaml").read_text())
    config["_base_"] = "/workspace/tutorials/neuron_snemi/neuron_snemi_v1.yaml"
    config["test"] = {"data": {"test": {"path": "", "label": str(root / "labels.h5")}}}
    (root / "config.yaml").write_text(yaml.safe_dump(config))
    score = run_benchmark(argparse.Namespace(
        config=root / "config.yaml", output=root / "run",
        prediction=root / "affinities.h5", checkpoint=None, resume_to=None,
    ))
    assert abs(score) < 1e-10, f"Perfect-affinity ABISS smoke failed: {score}"
PY

CMD ["python", "-m", "connectomics.runtime.snemi_benchmark", "--help"]
