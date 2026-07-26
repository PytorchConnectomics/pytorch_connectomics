from __future__ import annotations

from types import SimpleNamespace

from connectomics.decoding.experiment_log import log_decode_experiment


def test_graph_decode_is_recorded_with_pruned_operation_chain(tmp_path):
    graph = SimpleNamespace(
        nodes=[
            {
                "name": "sections",
                "op": "seg_2d",
                "inputs": ["raw"],
            },
            {
                "name": "tracklets",
                "op": "branch_link",
                "inputs": ["raw", "sections"],
            },
            {
                "name": "split",
                "op": "branch_split",
                "inputs": ["raw", "tracklets"],
            },
        ],
        output="tracklets",
    )
    cfg = SimpleNamespace(
        decoding=SimpleNamespace(steps=[], graph=graph),
        inference=SimpleNamespace(),
    )

    log_decode_experiment(
        cfg=cfg,
        output_dir=tmp_path,
        volume_name="sample",
        timestamp="20260725_120000",
        metrics_dict={"nerl": 0.5, "nerl_oracle_merge": 0.9},
    )

    lines = (tmp_path / "decode_experiments.tsv").read_text().splitlines()
    header = lines[0].split("\t")
    row = dict(zip(header, lines[1].split("\t")))
    assert row["decoder"] == "seg_2d+branch_link"
    assert row["nerl"] == "0.500000"
    assert row["nerl_oracle_merge"] == "0.900000"
