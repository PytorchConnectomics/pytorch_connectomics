import numpy as np

from connectomics.decoding.decoders.branch import split as split_module


def test_link_cut_change_cuts_persistent_real_iou_dip():
    seg = np.zeros((24, 12, 12), dtype=np.uint32)
    seg[:12, 2:6, 2:6] = 1
    seg[12:, 2:6, 7:11] = 1

    result, n_cut = split_module.link_cut_change(seg, min_size=1, recover=1.1)

    assert n_cut == 1
    assert np.all(result[:12][seg[:12] == 1] == 1)
    assert np.all(result[12:][seg[12:] == 1] == 2)


def test_link_cut_change_rejects_cut_inside_minimum_fragment_gate():
    seg = np.zeros((24, 12, 12), dtype=np.uint32)
    seg[:3, 2:6, 2:6] = 1
    seg[3:, 2:6, 7:11] = 1

    result, n_cut = split_module.link_cut_change(seg, min_size=1, recover=1.1)

    assert n_cut == 0
    np.testing.assert_array_equal(result, seg)


def test_split_pair_skips_existing_anchor_slice():
    seg = np.zeros((3, 24, 24), dtype=np.uint32)
    seg[0, 5:15, 5:15] = 1
    seg[1, :2, :2] = 2
    seg[1, 5:15, 5:15] = 3
    seg[2, 5:15, 5:15] = 4
    zr, sizes, _ = split_module.seg_stats(seg)
    pair = {
        "H": 2,
        "S1": 1,
        "S2": 4,
        "z1": 0,
        "z2": 2,
        "c1": np.array([9.5, 9.5]),
        "c2": np.array([9.5, 9.5]),
        "a1": 100,
        "a2": 100,
        "off": 0.0,
        "cal": 100.0,
    }

    n_carved = split_module.split_pair(seg, pair, zr, sizes)

    assert n_carved == 1
    assert np.all(seg[1, 5:15, 5:15] == 1)
    assert np.all(seg[2, 5:15, 5:15] == 1)
    assert np.all(seg[1, :2, :2] == 2)


def test_branch_split_uses_validated_stage_order_and_defaults(monkeypatch):
    calls = []
    seg = np.zeros((2, 2, 2), dtype=np.uint32)

    def fake_link(value, **kwargs):
        calls.append(("link", kwargs))
        return value + 1, 3

    def fake_confident(value, **kwargs):
        calls.append(("confident", kwargs))
        np.testing.assert_array_equal(value, seg + 1)
        return value + 1, 2, []

    monkeypatch.setattr(split_module, "link_cut_change", fake_link)
    monkeypatch.setattr(split_module, "confident_parallel_split", fake_confident)

    result = split_module.branch_split(np.zeros((3, 2, 2, 2)), seg)

    np.testing.assert_array_equal(result, seg + 2)
    assert calls[0][0] == "link"
    assert calls[0][1]["recover"] == 1.1
    assert calls[0][1]["inplace"] is True
    assert calls[1][0] == "confident"
    assert calls[1][1]["host_both"] is False
    assert calls[1][1]["inplace"] is True


def test_branch_split_reuses_stats_until_link_cut_mutates(monkeypatch):
    original_seg_stats = split_module.seg_stats
    calls = []

    def counting_seg_stats(value):
        calls.append(value)
        return original_seg_stats(value)

    monkeypatch.setattr(split_module, "seg_stats", counting_seg_stats)
    uncut = np.zeros((24, 12, 12), dtype=np.uint32)
    uncut[:, 2:6, 2:6] = 1
    split_module.branch_split(None, uncut, min_size=1)
    assert len(calls) == 1

    calls.clear()
    cut = np.zeros((24, 12, 12), dtype=np.uint32)
    cut[:12, 2:6, 2:6] = 1
    cut[12:, 2:6, 7:11] = 1
    split_module.branch_split(None, cut, min_size=1)
    assert len(calls) == 2


def test_branch_split_inplace_skips_defensive_copy():
    seg = np.zeros((24, 12, 12), dtype=np.uint32)
    seg[:12, 2:6, 2:6] = 1
    seg[12:, 2:6, 7:11] = 1
    stats = split_module.seg_stats(seg)

    result = split_module.branch_split(None, seg, min_size=1, stats=stats, inplace=True)

    assert result is seg
    assert result.max() == 2
