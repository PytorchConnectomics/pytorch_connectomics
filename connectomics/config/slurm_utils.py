"""SLURM cluster resource detection and management utilities."""

import subprocess
import re
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class NodeResources:
    """Resources available on a compute node."""

    cpus: int
    gpus: int
    memory_gb: int
    gpu_type: Optional[str] = None


@dataclass
class PartitionInfo:
    """Information about a SLURM partition."""

    name: str
    nodes: Dict[str, NodeResources]
    available: bool
    max_time: str
    state: str


def detect_slurm_resources() -> Dict[str, PartitionInfo]:
    """
    Auto-detect SLURM partitions and their resources.

    Returns:
        Dictionary mapping partition names to PartitionInfo objects.

    Raises:
        RuntimeError: If SLURM commands are not available.
    """
    try:
        # Test if SLURM is available
        subprocess.run(["sinfo", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("SLURM not available, returning empty resource dict")
        return {}

    partitions = {}

    try:
        # Get partition list: partition|state|timelimit|nodelist
        result = subprocess.run(
            ["sinfo", "-h", "-o", "%R|%A|%l|%N"], capture_output=True, text=True, check=True
        )

        for line in result.stdout.strip().split("\n"):
            if not line:
                continue

            parts = line.split("|")
            if len(parts) < 4:
                continue

            name, state, time_limit, nodelist = parts

            # Parse node resources for this partition
            nodes = _parse_partition_nodes(name, nodelist)

            # Determine if partition is available
            available = any(s in state.lower() for s in ["idle", "mix", "alloc"])

            partitions[name] = PartitionInfo(
                name=name, nodes=nodes, available=available, max_time=time_limit, state=state
            )

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to detect SLURM resources: {e}")
        return {}

    return partitions


def _parse_partition_nodes(partition: str, nodelist: str) -> Dict[str, NodeResources]:
    """
    Parse node resources for a given partition.

    Args:
        partition: Partition name
        nodelist: Comma-separated list of nodes (may include ranges)

    Returns:
        Dictionary mapping node names to NodeResources
    """
    nodes = {}

    try:
        # Expand nodelist (e.g., "node[01-03]" -> ["node01", "node02", "node03"])
        expanded_nodes = _expand_nodelist(nodelist)

        for node_name in expanded_nodes:
            # Get detailed node information
            result = subprocess.run(
                ["scontrol", "show", "node", node_name], capture_output=True, text=True, check=True
            )

            node_info = result.stdout

            # Parse CPU count
            cpu_match = re.search(r"CPUTot=(\d+)", node_info)
            cpus = int(cpu_match.group(1)) if cpu_match else 0

            # Parse memory (convert MB to GB)
            mem_match = re.search(r"RealMemory=(\d+)", node_info)
            memory_gb = int(mem_match.group(1)) // 1024 if mem_match else 0

            # Parse GPU count and type from GRES
            gpus = 0
            gpu_type = None
            gres_match = re.search(r"Gres=([^\s]+)", node_info)
            if gres_match:
                gres = gres_match.group(1)
                # Parse formats like "gpu:4", "gpu:a100:2", "gpu:v100:4(S:0-1)"
                gpu_match = re.search(r"gpu:(\w+)?:?(\d+)", gres)
                if gpu_match:
                    if gpu_match.group(1) and gpu_match.group(1).isdigit():
                        # Format: gpu:4
                        gpus = int(gpu_match.group(1))
                    else:
                        # Format: gpu:a100:2
                        gpu_type = gpu_match.group(1)
                        gpus = int(gpu_match.group(2)) if gpu_match.group(2) else 0

            nodes[node_name] = NodeResources(
                cpus=cpus, gpus=gpus, memory_gb=memory_gb, gpu_type=gpu_type
            )

    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to parse nodes for partition {partition}: {e}")

    return nodes


def _expand_nodelist(nodelist: str) -> List[str]:
    """
    Expand SLURM nodelist notation to individual node names.

    Args:
        nodelist: SLURM nodelist (e.g., "node[01-03,05]")

    Returns:
        List of expanded node names
    """
    try:
        # Use scontrol to expand nodelist
        result = subprocess.run(
            ["scontrol", "show", "hostnames", nodelist], capture_output=True, text=True, check=True
        )
        return result.stdout.strip().split("\n")
    except subprocess.CalledProcessError:
        # Fallback: return as-is if expansion fails
        return [nodelist]


# Cache configuration
CACHE_FILE = Path.home() / ".pytorch_connectomics_slurm_cache.json"
CACHE_VALIDITY = 3600  # 1 hour in seconds


def get_cluster_config(force_refresh: bool = False) -> Dict[str, PartitionInfo]:
    """
    Get cluster configuration from cache or refresh if stale.

    Args:
        force_refresh: If True, ignore cache and detect resources

    Returns:
        Dictionary mapping partition names to PartitionInfo objects
    """
    if not force_refresh and CACHE_FILE.exists():
        cache_age = time.time() - CACHE_FILE.stat().st_mtime
        if cache_age < CACHE_VALIDITY:
            try:
                with open(CACHE_FILE, "r") as f:
                    cached_data = json.load(f)
                    # Reconstruct dataclass objects from dict
                    return _dict_to_partition_info(cached_data)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load cache: {e}, refreshing...")

    # Refresh cache
    logger.info("Detecting SLURM cluster resources...")
    config = detect_slurm_resources()

    # Save to cache
    try:
        cache_data = _partition_info_to_dict(config)
        with open(CACHE_FILE, "w") as f:
            json.dump(cache_data, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")

    return config


def _partition_info_to_dict(partitions: Dict[str, PartitionInfo]) -> dict:
    """Convert PartitionInfo objects to JSON-serializable dict."""
    result = {}
    for name, info in partitions.items():
        result[name] = {
            "name": info.name,
            "available": info.available,
            "max_time": info.max_time,
            "state": info.state,
            "nodes": {node_name: asdict(resources) for node_name, resources in info.nodes.items()},
        }
    return result


def _dict_to_partition_info(data: dict) -> Dict[str, PartitionInfo]:
    """Reconstruct PartitionInfo objects from dict."""
    result = {}
    for name, info in data.items():
        result[name] = PartitionInfo(
            name=info["name"],
            available=info["available"],
            max_time=info["max_time"],
            state=info["state"],
            nodes={
                node_name: NodeResources(**resources)
                for node_name, resources in info["nodes"].items()
            },
        )
    return result


def filter_partitions(
    partitions: Dict[str, PartitionInfo],
    min_cpus: int = 1,
    min_gpus: int = 0,
    min_memory_gb: int = 0,
    available_only: bool = True,
    gpu_type: Optional[str] = None,
) -> Dict[str, PartitionInfo]:
    """
    Filter partitions based on resource requirements.

    Args:
        partitions: Dictionary of partition information
        min_cpus: Minimum CPUs per node
        min_gpus: Minimum GPUs per node
        min_memory_gb: Minimum memory in GB per node
        available_only: Only return available partitions
        gpu_type: Required GPU type (e.g., "a100", "v100")

    Returns:
        Filtered dictionary of partitions
    """
    filtered = {}

    for name, info in partitions.items():
        # Check availability
        if available_only and not info.available:
            continue

        # Check if any node meets requirements
        valid_nodes = {}
        for node_name, resources in info.nodes.items():
            if (
                resources.cpus >= min_cpus
                and resources.gpus >= min_gpus
                and resources.memory_gb >= min_memory_gb
            ):
                # Check GPU type if specified
                if gpu_type and resources.gpu_type != gpu_type:
                    continue

                valid_nodes[node_name] = resources

        if valid_nodes:
            filtered[name] = PartitionInfo(
                name=info.name,
                nodes=valid_nodes,
                available=info.available,
                max_time=info.max_time,
                state=info.state,
            )

    return filtered


def get_best_partition(
    partition_preferences: Optional[List[str]] = None,
    min_cpus: int = 1,
    min_gpus: int = 0,
    min_memory_gb: int = 0,
    gpu_type: Optional[str] = None,
) -> Optional[str]:
    """
    Get the best available partition based on preferences and requirements.

    Args:
        partition_preferences: Ordered list of preferred partition names
        min_cpus: Minimum CPUs per node
        min_gpus: Minimum GPUs per node
        min_memory_gb: Minimum memory in GB per node
        gpu_type: Required GPU type

    Returns:
        Name of best partition, or None if no suitable partition found
    """
    partitions = get_cluster_config()

    # Filter by requirements
    valid_partitions = filter_partitions(
        partitions,
        min_cpus=min_cpus,
        min_gpus=min_gpus,
        min_memory_gb=min_memory_gb,
        gpu_type=gpu_type,
    )

    if not valid_partitions:
        logger.warning("No partitions meet resource requirements")
        return None

    # Return first match from preferences
    if partition_preferences:
        for pref in partition_preferences:
            if pref in valid_partitions:
                return pref

    # Fallback: return partition with most GPUs
    best = max(
        valid_partitions.items(), key=lambda x: sum(node.gpus for node in x[1].nodes.values())
    )
    return best[0]


def print_cluster_resources():
    """Print formatted cluster resource information."""
    partitions = get_cluster_config()

    if not partitions:
        print("No SLURM partitions detected")
        return

    print("\n" + "=" * 80)
    print("SLURM Cluster Resources")
    print("=" * 80)

    for name, info in partitions.items():
        print(f"\nPartition: {name}")
        print(f"  State: {info.state}")
        print(f"  Available: {info.available}")
        print(f"  Max Time: {info.max_time}")
        print(f"  Nodes: {len(info.nodes)}")

        # Aggregate resources
        total_cpus = sum(node.cpus for node in info.nodes.values())
        total_gpus = sum(node.gpus for node in info.nodes.values())
        total_memory = sum(node.memory_gb for node in info.nodes.values())

        print(f"  Total CPUs: {total_cpus}")
        print(f"  Total GPUs: {total_gpus}")
        print(f"  Total Memory: {total_memory} GB")

        # GPU types
        gpu_types = set(node.gpu_type for node in info.nodes.values() if node.gpu_type)
        if gpu_types:
            print(f"  GPU Types: {', '.join(gpu_types)}")

        # Sample node
        if info.nodes:
            sample_node = next(iter(info.nodes.items()))
            print(f"  Sample Node: {sample_node[0]}")
            print(f"    CPUs: {sample_node[1].cpus}")
            print(f"    GPUs: {sample_node[1].gpus}")
            print(f"    Memory: {sample_node[1].memory_gb} GB")
            if sample_node[1].gpu_type:
                print(f"    GPU Type: {sample_node[1].gpu_type}")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    # CLI usage
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--refresh":
        print("Refreshing cluster resource cache...")
        get_cluster_config(force_refresh=True)

    print_cluster_resources()
