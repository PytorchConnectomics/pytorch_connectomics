"""Launch one disposable GPU or CPU VM for the staged diSPIM benchmark.

The run prefix must already contain the built image, prepared data, and
vm_stage.sh. Run GPU first; verify its COMPLETE marker, delete that VM/disk,
then launch CPU. This command never overwrites input data or deletes a VM.
"""

import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path
from shlex import quote


def launch(stage: str, prefix: str, project: str, zone: str, name: str, service_account: str):
    if not prefix.startswith("gs://"):
        raise ValueError("run-prefix must be a GCS URI")
    prefix = prefix.rstrip("/")
    # Refuse to allocate expensive compute before its inputs are durable.
    required = ["build/image.sha256", "prepared/provenance.json", "vm_stage.sh", "config.yaml"]
    if stage == "cpu":
        required += ["gpu/COMPLETE", "gpu/infer/affinities.h5"]
    for path in required:
        subprocess.run(["gcloud", "storage", "ls", f"{prefix}/{path}"], check=True)
    prepared = json.loads(subprocess.check_output(
        ["gcloud", "storage", "cat", f"{prefix}/prepared/provenance.json"], text=True
    ))
    if prepared["status"] != "complete" or len(prepared["files"]) != 6:
        raise ValueError("Expected six completed prepared image/label files")
    voxels = math.prod(prepared["sources"]["test"]["shape"])
    # ABISS holds full-volume affinities and a region graph in memory.
    cpu_type = "n2-highmem-32" if voxels > 200_000_000 else "e2-standard-16"
    disk_gb = 500 if voxels > 200_000_000 else 200
    subprocess.run([
        "gcloud", "storage", "ls", *[f"{prefix}/prepared/{name}" for name in prepared["files"]]
    ], check=True)
    script = (
        "#!/bin/bash\nset -Eeuo pipefail\n"
        f"gcloud storage cp {quote(prefix + '/vm_stage.sh')} /tmp/vm_stage.sh\n"
        f"exec bash /tmp/vm_stage.sh {quote(stage)} {quote(prefix)}\n"
    )
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "startup.sh"
        path.write_text(script)
        command = [
            "gcloud", "compute", "instances", "create", name,
            f"--project={project}", f"--zone={zone}",
            f"--machine-type={'g2-standard-48' if stage == 'gpu' else cpu_type}",
            f"--boot-disk-size={disk_gb}GB", "--boot-disk-type=pd-ssd",
            f"--service-account={service_account}", "--scopes=cloud-platform",
            f"--metadata-from-file=startup-script={path}",
            f"--max-run-duration={'12h' if stage == 'gpu' else '6h'}",
            "--instance-termination-action=DELETE", "--no-restart-on-failure", "--quiet",
        ]
        if stage == "gpu":
            command += [
                "--image=common-cu129-ubuntu-2204-nvidia-580-v20260831",
                "--image-project=deeplearning-platform-release",
            ]
        else:
            command += ["--image-family=debian-12", "--image-project=debian-cloud"]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("gpu", "cpu"))
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--zone", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--service-account", required=True)
    args = parser.parse_args()
    launch(args.stage, args.run_prefix, args.project, args.zone, args.name, args.service_account)
