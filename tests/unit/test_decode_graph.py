"""Unit tests for decoder graph execution."""

from __future__ import annotations

import numpy as np
import pytest

from connectomics.config.pipeline.config_io import from_dict
from connectomics.decoding.base import DecodeGraph, DecodeNode
from connectomics.decoding.decoders.combine import combine_split
from connectomics.decoding.graph import run_decode_graph, steps_to_graph, validate_graph
from connectomics.decoding.pipeline import apply_decode_pipeline
from connectomics.decoding.registry import DecoderRegistry


def test_steps_lower_to_graph_executor_with_legacy_hooks():
    registry = DecoderRegistry()
    registry.register("plus", lambda data, offset=1: data + offset)
    registry.register("times", lambda data, factor=2: data * factor)
    registry.register("identity", lambda data: data.copy())
    registry.register("make_seg", lambda data: np.zeros(data.shape[1:], dtype=np.uint64))

    def correction(seg, affinities=None):
        assert affinities is not None
        return seg + (affinities[0] > 0).astype(np.uint64)

    registry.register("correction", correction)
    data = np.ones((3, 2, 3, 4), dtype=np.float32)
    cases = [
        [
            {"name": "plus", "kwargs": {"offset": 2}},
            {"name": "times", "kwargs": {"factor": 3}},
        ],
        [
            {"name": "identity", "kwargs": {"spatial_transpose": [2, 1, 0]}},
        ],
        [
            {"name": "make_seg", "kwargs": {}},
            {"name": "correction", "kwargs": {"use_original_input": True}},
        ],
    ]

    for steps in cases:
        linear = apply_decode_pipeline(data, steps, registry=registry)
        graph = run_decode_graph(data, steps_to_graph(steps), registry)
        np.testing.assert_array_equal(linear, graph)


def test_apply_decode_pipeline_keeps_legacy_step_callback_objects():
    registry = DecoderRegistry()
    registry.register("plus", lambda data, **_kw: data + 1)
    captured = []

    result = apply_decode_pipeline(
        np.zeros((1, 2, 2, 2), dtype=np.float32),
        [{"name": "plus", "kwargs": {"tag": "debug"}}],
        registry=registry,
        on_step_complete=lambda batch, step, sample: captured.append((batch, step, sample.copy())),
    )

    assert captured[0][0] == 0
    assert captured[0][1].name == "plus"
    assert captured[0][1].kwargs == {"tag": "debug"}
    np.testing.assert_array_equal(captured[0][2], result)


def test_raw_ref_grammar_preserves_channel_axis():
    registry = DecoderRegistry()
    registry.register("identity", lambda data: data.copy())
    data = np.arange(4 * 2 * 3 * 4, dtype=np.float32).reshape(4, 2, 3, 4)
    graph = DecodeGraph(
        nodes=[
            DecodeNode(name="single", op="identity", inputs=["raw[2]"]),
            DecodeNode(name="window", op="identity", inputs=["raw[1:3]"]),
        ],
        output="window",
    )
    shapes = {}

    result = run_decode_graph(
        data,
        graph,
        registry,
        on_node_complete=lambda _batch, node, sample: shapes.setdefault(node.name, sample.shape),
    )

    assert shapes["single"] == (1, 2, 3, 4)
    assert shapes["window"] == (2, 2, 3, 4)
    np.testing.assert_array_equal(result, data[1:3])


@pytest.mark.parametrize(
    ("graph", "match"),
    [
        (
            {
                "nodes": [
                    {"name": "a", "op": "identity", "inputs": ["raw"]},
                    {"name": "a", "op": "identity", "inputs": ["raw"]},
                ],
                "output": "a",
            },
            "unique",
        ),
        (
            {"nodes": [{"name": "raw", "op": "identity", "inputs": ["raw"]}], "output": "raw"},
            "reserved",
        ),
        (
            {
                "nodes": [
                    {"name": "a", "op": "identity", "inputs": ["b"]},
                    {"name": "b", "op": "identity", "inputs": ["a"]},
                ],
                "output": "a",
            },
            "cycle",
        ),
        (
            {"nodes": [{"name": "a", "op": "identity", "inputs": ["missing"]}], "output": "a"},
            "unknown",
        ),
        (
            {
                "nodes": [
                    {"name": "disabled", "op": "identity", "inputs": ["raw"], "enabled": False},
                    {"name": "out", "op": "identity", "inputs": ["disabled"]},
                ],
                "output": "out",
            },
            "disabled",
        ),
        (
            {
                "nodes": [
                    {"name": "disabled", "op": "identity", "inputs": ["raw"], "enabled": False},
                    {"name": "out", "op": "identity", "inputs": ["raw"]},
                ],
                "output": "disabled",
            },
            "disabled",
        ),
    ],
)
def test_validate_graph_rejects_invalid_graphs(graph, match):
    with pytest.raises(ValueError, match=match):
        validate_graph(graph)


def test_config_rejects_graph_and_steps_together():
    with pytest.raises(ValueError, match="graph.*steps"):
        from_dict(
            {
                "decoding": {
                    "graph": {
                        "nodes": [{"name": "out", "op": "identity", "inputs": ["raw"]}],
                        "output": "out",
                    },
                    "steps": [{"name": "identity"}],
                }
            }
        )


def test_register_graph_op_keeps_multi_input_ops_unwrapped():
    registry = DecoderRegistry()
    a = np.array([[1, 1, 0], [2, 2, 2]], dtype=np.uint32)
    b = np.array([[3, 4, 4], [3, 3, 0]], dtype=np.uint32)
    registry.register("label_a", lambda _raw: a)
    registry.register("label_b", lambda _raw: b)
    registry.register_graph_op("combine_split", combine_split)

    graph = DecodeGraph(
        nodes=[
            DecodeNode(name="a", op="label_a", inputs=["raw"]),
            DecodeNode(name="b", op="label_b", inputs=["raw"]),
            DecodeNode(name="out", op="combine_split", inputs=["a", "b"]),
        ],
        output="out",
    )

    result = run_decode_graph(np.zeros((1, 2, 3), dtype=np.float32), graph, registry)

    assert set(np.unique(result)) == {0, 1, 2, 3}


def test_combine_split_background_deterministic_dtype_and_overflow():
    a = np.array([[1, 1, 1, 0], [2, 2, 0, 2]], dtype=np.uint32)
    b = np.array([[5, 5, 6, 6], [5, 6, 6, 0]], dtype=np.uint32)
    expected = np.array([[1, 1, 2, 0], [3, 4, 0, 0]], dtype=np.uint16)

    result = combine_split([a, b], output_dtype="uint16")

    assert result.dtype == np.uint16
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(result, combine_split([a, b], output_dtype="uint16"))

    many = np.arange(1, 257, dtype=np.uint16)
    with pytest.raises(OverflowError, match="output dtype"):
        combine_split([many, np.ones_like(many)], output_dtype="uint8")

    too_large = np.array([np.iinfo(np.uint64).max], dtype=np.uint64)
    with pytest.raises(OverflowError, match="uint64"):
        combine_split([too_large, np.array([1], dtype=np.uint64)])


def test_banis2_style_graph_runs_and_combines_common_refinement():
    predictions = np.zeros((5, 5, 5, 5), dtype=np.float32)
    predictions[:3, 1:4, 1:4, 1:4] = 1.0
    predictions[3, 1:4, 1:4, 1:4] = 1.0
    predictions[4, 1:4, 1:4, 1:2] = 1.0
    predictions[4, 1:4, 1:4, 3:4] = 1.0
    graph = {
        "nodes": [
            {
                "name": "g2",
                "op": "channel_gate",
                "inputs": ["raw"],
                "kwargs": {"signal_channels": [0, 1, 2], "gate_channel": 3},
            },
            {
                "name": "n1",
                "op": "decode_affinity_cc",
                "inputs": ["g2"],
                "kwargs": {"threshold": 0.5, "backend": "cc3d"},
            },
            {
                "name": "g10",
                "op": "channel_gate",
                "inputs": ["raw"],
                "kwargs": {"signal_channels": [0, 1, 2], "gate_channel": 4},
            },
            {
                "name": "n2",
                "op": "decode_affinity_cc",
                "inputs": ["g10"],
                "kwargs": {"threshold": 0.5, "backend": "cc3d"},
            },
            {"name": "out", "op": "combine_split", "inputs": ["n1", "n2"]},
        ],
        "output": "out",
    }
    captured = {}

    result = run_decode_graph(
        predictions,
        graph,
        on_node_complete=lambda _batch, node, sample: captured.setdefault(
            node.name, sample.copy()
        ),
    )

    assert captured["g2"].shape == (3, 5, 5, 5)
    assert np.all(result[(captured["n1"] == 0) | (captured["n2"] == 0)] == 0)
    assert set(np.unique(result)) == {0, 1, 2}
    for label in np.unique(result):
        if label == 0:
            continue
        mask = result == label
        n1_labels = np.unique(captured["n1"][mask])
        n2_labels = np.unique(captured["n2"][mask])
        assert len(n1_labels) == 1 and int(n1_labels[0]) > 0
        assert len(n2_labels) == 1 and int(n2_labels[0]) > 0
