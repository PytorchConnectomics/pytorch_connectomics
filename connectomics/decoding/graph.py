"""Decoder graph normalization, validation, and execution."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, List, Mapping, Sequence

import numpy as np

from ..utils.channel_slices import (
    normalize_channel_range_selector,
    resolve_channel_indices,
    resolve_channel_range,
)
from .base import DecodeGraph, DecodeNode
from .pipeline import (
    _apply_spatial_transpose,
    _invert_axis_permutation,
    _prepare_batched_input,
    normalize_decode_modes,
)
from .registry import DEFAULT_DECODER_REGISTRY, DecoderRegistry, ensure_builtin_decoders_registered

RAW_REF = "raw"


def _is_raw_ref(ref: str) -> bool:
    return ref == RAW_REF or (ref.startswith("raw[") and ref.endswith("]"))


def _coerce_kwargs(kwargs: Any) -> dict[str, Any]:
    if kwargs is None:
        return {}
    if hasattr(kwargs, "items"):
        return dict(kwargs)
    raise TypeError(f"Decode node kwargs must be a mapping, got {type(kwargs).__name__}.")


def _get_value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _normalize_node(node: Any) -> DecodeNode:
    if isinstance(node, DecodeNode):
        return DecodeNode(
            enabled=bool(node.enabled),
            name=str(node.name),
            op=str(node.op),
            inputs=list(node.inputs),
            kwargs=_coerce_kwargs(node.kwargs),
        )

    enabled = bool(_get_value(node, "enabled", True))
    name = _get_value(node, "name", "")
    op = _get_value(node, "op", "")
    inputs = _get_value(node, "inputs", [])
    kwargs = _coerce_kwargs(_get_value(node, "kwargs", {}))
    return DecodeNode(
        enabled=enabled,
        name=str(name),
        op=str(op),
        inputs=[str(ref) for ref in (inputs or [])],
        kwargs=kwargs,
    )


def normalize_graph(cfg: Any) -> DecodeGraph:
    """Normalize dict/dataclass/OmegaConf graph config into a ``DecodeGraph``."""
    if isinstance(cfg, DecodeGraph):
        return DecodeGraph(
            nodes=[_normalize_node(node) for node in cfg.nodes],
            output=str(cfg.output),
        )
    if cfg is None:
        raise ValueError("Decode graph configuration is missing.")

    nodes = _get_value(cfg, "nodes", [])
    output = _get_value(cfg, "output", "")
    return DecodeGraph(nodes=[_normalize_node(node) for node in (nodes or [])], output=str(output))


def steps_to_graph(steps: Iterable[Any]) -> DecodeGraph:
    """Lower a linear decode step list into a chain-shaped graph."""
    normalized = [step for step in normalize_decode_modes(steps) if step.enabled]
    nodes: List[DecodeNode] = []
    previous = RAW_REF
    for idx, step in enumerate(normalized):
        name = f"node_{idx}"
        nodes.append(
            DecodeNode(
                enabled=True,
                name=name,
                op=step.name,
                inputs=[previous],
                kwargs=dict(step.kwargs),
            )
        )
        previous = name

    output = nodes[-1].name if nodes else ""
    return DecodeGraph(nodes=nodes, output=output)


def _validate_input_ref(ref: str, kept_names: set[str], disabled_names: set[str]) -> str | None:
    if ref == RAW_REF:
        return None
    if ref.startswith("raw[") and ref.endswith("]"):
        selector = ref[4:-1]
        if not selector:
            raise ValueError("Raw input slice must not be empty.")
        normalize_channel_range_selector(selector, context=f"graph input reference {ref}")
        return None
    if ref.startswith("raw[") or ref.endswith("]"):
        raise ValueError(f"Invalid graph input reference {ref!r}.")
    if ref in kept_names:
        return ref
    if ref in disabled_names:
        raise ValueError(f"Graph input reference {ref!r} points to a disabled node.")
    raise ValueError(f"Graph input reference {ref!r} is unknown.")


def _toposort(nodes: Sequence[DecodeNode]) -> List[DecodeNode]:
    by_name = {node.name: node for node in nodes}
    deps = {
        node.name: [ref for ref in node.inputs if ref in by_name]
        for node in nodes
    }
    state: dict[str, str] = {}
    ordered: List[DecodeNode] = []

    def visit(name: str, stack: list[str]) -> None:
        marker = state.get(name)
        if marker == "done":
            return
        if marker == "visiting":
            cycle = " -> ".join([*stack, name])
            raise ValueError(f"Decode graph contains a cycle: {cycle}.")
        state[name] = "visiting"
        for dep in deps[name]:
            visit(dep, [*stack, name])
        state[name] = "done"
        ordered.append(by_name[name])

    for node in nodes:
        visit(node.name, [])
    return ordered


def validate_graph(graph: Any) -> DecodeGraph:
    """Validate a graph and return a graph containing kept nodes in topological order."""
    normalized = normalize_graph(graph)
    names = [node.name for node in normalized.nodes]
    for idx, node in enumerate(normalized.nodes):
        if not node.name:
            raise ValueError(f"Decode graph node {idx} is missing required field 'name'.")
        if node.name == RAW_REF:
            raise ValueError("'raw' is a reserved graph input reference, not a legal node name.")
        if not node.op:
            raise ValueError(f"Decode graph node {node.name!r} is missing required field 'op'.")

    duplicate_names = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicate_names:
        raise ValueError(
            f"Decode graph node names must be unique, got duplicates {duplicate_names}."
        )

    kept = [node for node in normalized.nodes if node.enabled]
    if not kept:
        raise ValueError("Decode graph must contain at least one enabled node.")
    kept_names = {node.name for node in kept}
    disabled_names = {node.name for node in normalized.nodes if not node.enabled}

    if not normalized.output:
        raise ValueError("Decode graph is missing required field 'output'.")
    if normalized.output in disabled_names:
        raise ValueError(f"Decode graph output {normalized.output!r} points to a disabled node.")
    if normalized.output not in kept_names:
        raise ValueError(
            f"Decode graph output {normalized.output!r} does not name an enabled node."
        )

    for node in kept:
        if not node.inputs:
            raise ValueError(f"Decode graph node {node.name!r} must declare at least one input.")
        for ref in node.inputs:
            _validate_input_ref(ref, kept_names, disabled_names)

    return DecodeGraph(nodes=_toposort(kept), output=normalized.output)


def _resolve_raw_ref(ref: str, raw: np.ndarray) -> np.ndarray:
    if ref == RAW_REF:
        return raw
    if not (ref.startswith("raw[") and ref.endswith("]")):
        raise KeyError(ref)
    selector = ref[4:-1]
    start, stop = resolve_channel_range(
        selector,
        num_channels=int(raw.shape[0]),
        context=f"graph input reference {ref}",
    )
    return raw[start:stop]


def _resolve_inputs(
    node: DecodeNode,
    raw: np.ndarray,
    values: Mapping[str, np.ndarray],
) -> list[np.ndarray]:
    inputs: list[np.ndarray] = []
    for ref in node.inputs:
        if _is_raw_ref(ref):
            inputs.append(_resolve_raw_ref(ref, raw))
        else:
            inputs.append(values[ref])
    return inputs


def _resolve_decoder_kwargs(node: DecodeNode, inputs: Sequence[np.ndarray]) -> dict[str, Any]:
    kwargs = dict(node.kwargs)
    for key, value in list(kwargs.items()):
        if not key.endswith("_channels"):
            continue
        if not inputs:
            raise ValueError(f"Cannot resolve {node.op}.{key}: node has no inputs.")
        first = inputs[0]
        if first.ndim < 1:
            raise ValueError(
                f"Cannot resolve {node.op}.{key}: first input has no channel axis."
            )
        kwargs[key] = resolve_channel_indices(
            value,
            num_channels=int(first.shape[0]),
            context=f"decode kwargs {node.op}.{key}",
        )
    return kwargs


def _node_dependencies(node: DecodeNode) -> list[str]:
    return [ref for ref in node.inputs if not _is_raw_ref(ref)]


def _consumer_counts(nodes: Sequence[DecodeNode]) -> Counter:
    counts: Counter = Counter()
    for node in nodes:
        counts.update(_node_dependencies(node))
    return counts


def run_decode_graph(
    data: np.ndarray,
    graph: Any,
    registry: DecoderRegistry | None = None,
    *,
    on_node_complete: Any = None,
) -> np.ndarray:
    """Run a decoder graph over raw prediction data."""
    validated = validate_graph(graph)
    if registry is None:
        ensure_builtin_decoders_registered()
        registry = DEFAULT_DECODER_REGISTRY

    batched, batch_size = _prepare_batched_input(data)
    results: list[np.ndarray] = []
    for batch_idx in range(batch_size):
        raw = batched[batch_idx]
        values: dict[str, np.ndarray] = {}
        consumers = _consumer_counts(validated.nodes)

        for node in validated.nodes:
            inputs = _resolve_inputs(node, raw, values)
            try:
                op = registry.get(node.op)
            except KeyError as exc:
                available = ", ".join(registry.available())
                raise ValueError(
                    f"Unknown decode function '{node.op}'. "
                    f"Available functions: [{available}]."
                ) from exc

            try:
                kwargs = _resolve_decoder_kwargs(node, inputs)
                spatial_transpose = kwargs.pop("spatial_transpose", None)
                use_original_input = bool(kwargs.pop("use_original_input", False))
                original_input_kwarg = kwargs.pop("original_input_kwarg", "affinities")
                op_inputs = list(inputs)
                original_for_decoder = raw
                if spatial_transpose:
                    op_inputs = [
                        _apply_spatial_transpose(input_arr, spatial_transpose)
                        for input_arr in op_inputs
                    ]
                    original_for_decoder = _apply_spatial_transpose(
                        original_for_decoder,
                        spatial_transpose,
                    )
                if use_original_input:
                    kwargs[original_input_kwarg] = original_for_decoder
                output = op(op_inputs, **kwargs)
                if spatial_transpose:
                    output = _apply_spatial_transpose(
                        output,
                        _invert_axis_permutation(spatial_transpose),
                    )
            except Exception as exc:
                raise RuntimeError(f"Error applying decode function '{node.op}': {exc}") from exc

            values[node.name] = output
            if on_node_complete is not None:
                on_node_complete(batch_idx, node, output)

            for dep in _node_dependencies(node):
                consumers[dep] -= 1
                if consumers[dep] <= 0 and dep != validated.output:
                    values.pop(dep, None)

        results.append(values[validated.output])

    if len(results) == 1:
        return results[0]
    return np.stack(results, axis=0)


__all__ = [
    "RAW_REF",
    "normalize_graph",
    "run_decode_graph",
    "steps_to_graph",
    "validate_graph",
]
