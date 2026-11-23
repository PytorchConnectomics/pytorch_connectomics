"""Artifact contracts shared by the morphology error-correction workflow."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np

FORBIDDEN_PATH_PARTS = frozenset(
    {
        "arm096_lut",
        "evaluation_gt_only",
        "ffn_pseudogt",
        "matchguard_lut",
        "oracle_gt",
        "test_50",
        "valid_12",
    }
)


def reject_evaluation_path(path: str | Path) -> None:
    """Reject evaluation or pseudo-label artifacts at the decoder boundary."""

    value = str(path)
    if "://" in value:
        parts = value.replace("\\", "/").lower().split("/")
    else:
        parts = [part.lower() for part in Path(value).expanduser().resolve().parts]
    forbidden = FORBIDDEN_PATH_PARTS.intersection(parts)
    if forbidden:
        raise ValueError(f"evaluation/GT path is forbidden in error correction: {path}")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_gt_free_npz(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    reject_evaluation_path(path)
    with np.load(path, allow_pickle=False) as data:
        result = {key: np.asarray(data[key]) for key in data.files}
    if "gt_free" not in result or not bool(result["gt_free"].item()):
        raise ValueError(f"{path} is not explicitly marked GT-free")
    return result


class UnionFind:
    """Deterministic union-find whose component representative is its smallest label."""

    def __init__(self, labels: Iterable[int] = ()) -> None:
        self.parent = {int(label): int(label) for label in labels if int(label) != 0}

    def find(self, label: int) -> int:
        label = int(label)
        root = self.parent.setdefault(label, label)
        while root != self.parent[root]:
            root = self.parent[root]
        while label != self.parent[label]:
            parent = self.parent[label]
            self.parent[label] = root
            label = parent
        return root

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if right_root < left_root:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root


def load_frozen_merge_roots(
    proposals_path: str | Path, report_path: str | Path
) -> tuple[np.ndarray, np.ndarray, str]:
    """Validate a frozen proposal and return sorted label-to-root arrays."""

    proposals_path = Path(proposals_path)
    report_path = Path(report_path)
    reject_evaluation_path(proposals_path)
    reject_evaluation_path(report_path)
    digest = sha256_file(proposals_path)
    report = json.loads(report_path.read_text())
    if report.get("proposal_sha256") != digest:
        raise ValueError("proposal hash differs from its frozen report")
    proposal = load_gt_free_npz(proposals_path)
    if not bool(proposal.get("frozen_before_evaluation", np.asarray(False)).item()):
        raise ValueError("proposal was not frozen before evaluation")
    pairs = sorted(
        {
            tuple(sorted((int(left), int(right))))
            for left, right in zip(proposal["left"].tolist(), proposal["right"].tolist())
            if int(left) and int(right) and int(left) != int(right)
        }
    )
    labels = np.unique(np.asarray(pairs, dtype=np.uint64)) if pairs else np.zeros(0, np.uint64)
    union = UnionFind(labels)
    for left, right in pairs:
        union.union(left, right)
    roots = np.asarray([union.find(int(label)) for label in labels], dtype=np.uint64)
    return labels, roots, digest


def relabel_sorted(
    values: np.ndarray, source_labels: np.ndarray, target_labels: np.ndarray
) -> np.ndarray:
    """Relabel an integer array from sorted sparse source/target tables."""

    array = np.asarray(values)
    if len(source_labels) == 0:
        return array.copy()
    index = np.searchsorted(source_labels, array)
    safe = np.minimum(index, len(source_labels) - 1)
    matched = (index < len(source_labels)) & (source_labels[safe] == array)
    output = array.copy()
    output[matched] = target_labels[safe[matched]]
    return output
