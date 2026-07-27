"""Seeded branch-split postprocessing for instance segmentations.

===============================================================================
CUE LADDER — what to trust when deciding "is this one label actually 2 neurons,
and where does the boundary go?"
===============================================================================
Ordered MOST → LEAST robust, from measurements on MIT-LiCONN DL288B. Splitting
is ORACLE-SAFE (cutting can only raise the false-merge-free ceiling om), so be
braver here than when merging — but a bad *boundary* still leaks voxels into a
long neuron, and that costs more than the split gains.

DETECTION — is there a second tube in here?

 1. TWO SEPARABLE 2D COMPONENTS on a run of slices.                 [strongest]
    (measured in research, not vendored here — no cue-1 detector in this module)
    If the cross-section separates, the evidence is direct. Require a LOCAL RUN
    of consecutive 2-lobe slices: a foreign axon crossing for 15-40 of a
    400-slice tube is invisible to any whole-segment fraction test.

 2. SUSTAINED AREA BUMP — interior area ≥1.5× the running median for ≥3
    slices, excluding the tapered ends. (also not vendored here)
    Catches a swap/crossover where the tubes coexist for a stretch. The naive
    max-deviation version FAILS (it flags normal endpoint taper).

 3. REAL-IoU CHANGE-POINT at a z-seam — cut where consecutive-slice IoU dips
    below the median of its neighbours (adaptive, NOT a fixed threshold).
    fn: `link_cut_change`
    Catches sequential mis-links, which have no 2-lobe or area signature.
    Gate with local-minimum + a minimum fragment spacing, or it over-cuts.

 4. ONE-SIDED CONTAINMENT — |S∩H|/|S| high while |S∩H|/|H| is low.
    fn: `detect_confident` (`_shifted_iomm` vs `IOMIN_HI`/`IOMAX_LO`)
    A branch entering a host (vs a continuation, where both are high).

 5. BLENDED-CENTROID TRAJECTORY — NOT USABLE for detection.
    Once two tubes fuse, the centroid is a mixture of both, so a polynomial fit
    to it is self-contradictory (it matched tube A for 10 slices, then tube B,
    and disagreed with ground truth about which one continues).

 6. IN-PLANE FUSION with sep = 0 on every slice — no in-plane cue exists.
    This is the model floor: there is nothing in XY affinity to cut on. Needs
    an orthogonal signal (XZ/YZ sections, membrane channel) or a learned
    scorer. Leaving it merged is the correct action.

BOUNDARY PLACEMENT — once you know to split, where?

 A. TWO-SIDED ANCHORING (both ends of the fused stretch known) → INTERPOLATE
    the centre-line between them, and propagate each tube's ACTUAL
    cross-section mask inward from its nearer end.
    fn: `split_pair` (`gather_anchors` + `_traj` + per-slice 2-marker watershed)
    Point/disk seeds leak (~5.2k voxels in one case); propagating the real mask
    cut that to ~1.6k.

 B. ONE-SIDED CARVE (only one end known) → EXTRAPOLATE. Dangerous: with no
    exit anchor it drifts along the host's geometry and absorbs whatever it
    meets. Prefer to leave the object merged over a one-sided carve.

RULE OF THUMB: detect with 1-3, place the boundary with A, and decline (5, 6,
B) rather than guess — a leak into a long neuron is scored harder than the
split you were trying to fix.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    center_of_mass,
    distance_transform_edt,
)
from scipy.ndimage import label as cc_label
from skimage.segmentation import watershed

from connectomics.data.processing.bbox import seg_stats

__all__ = ["branch_split"]


# link-cut gates
DROP_THR, W, MIN_SIZE, MIN_SPAN, MIN_FRAG, RECOVER = 0.25, 4, 10000, 20, 6, 0.4

# tunnel-split gates
CAND_MIN, CAND_SPAN, HOST_MIN = 500, 8, 20000
IOMIN_HI, IOMAX_LO, MAX_SHIFT = 0.6, 0.7, 6
MAX_GAP, CAL_RATIO, COLLINEAR_NM = 40, 2.2, 900
TUBE_THR = 0.7
STEP_LO, STEP_HI = 0.4, 2.0
ANCHOR_NM = 150
DRIFT_HI = 1.6


def _shift_align(a, b, max_shift):
    """Roll b onto a by the centroid offset, clamped to ``+-max_shift`` voxels."""
    ca, cb = center_of_mass(a), center_of_mass(b)
    dy = int(np.clip(round(ca[0] - cb[0]), -max_shift, max_shift))
    dx = int(np.clip(round(ca[1] - cb[1]), -max_shift, max_shift))
    return np.roll(np.roll(b, dy, 0), dx, 1), int(a.sum()), int(b.sum())


def _shifted_iou(a, b, *, max_shift=5):
    """Shifted (clamped-centroid-aligned) IoU."""
    if not a.any() or not b.any():
        return 0.0
    b2, _, _ = _shift_align(a, b, max_shift)
    return float((a & b2).sum() / max((a | b2).sum(), 1))


def _shifted_iomm(a, b, *, max_shift=5):
    """Shifted intersection-over-min and intersection-over-max."""
    if not a.any() or not b.any():
        return 0.0, 0.0
    b2, aa, bb = _shift_align(a, b, max_shift)
    inter = float((a & b2).sum())
    return inter / max(min(aa, bb), 1), inter / max(max(aa, bb), 1)


def _bbox2d(mask):
    """Return an exclusive-high 2D bounding box for a non-empty mask."""
    ys = np.where(mask.any(1))[0]
    xs = np.where(mask.any(0))[0]
    return int(ys[0]), int(ys[-1]) + 1, int(xs[0]), int(xs[-1]) + 1


def link_cut_change(
    seg,
    *,
    drop_thr=DROP_THR,
    w=W,
    min_size=MIN_SIZE,
    min_frag=MIN_FRAG,
    recover=RECOVER,
    verbose=False,
    trace=frozenset(),
    stats=None,
    inplace=False,
):
    """Cut real-IoU change points, preserving the validated adaptive gates."""
    if not inplace:
        seg = np.array(seg, dtype=np.uint32, copy=True)
    zr, sizes = stats[:2] if stats is not None else seg_stats(seg)[:2]
    Z = seg.shape[0]  # noqa: F841
    next_id = int(seg.max()) + 1
    n_cut = 0
    cands = [
        int(L)
        for L in np.unique(seg)
        if L > 0 and sizes[L] >= min_size and zr[L][1] - zr[L][0] + 1 >= MIN_SPAN
    ]
    for L in cands:
        z0, z1, y0, y1, x0, x1 = zr[L]
        sub = seg[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1] == L
        pres = [zi for zi in range(sub.shape[0]) if sub[zi].any()]
        if len(pres) < 2 * w + 2:
            continue
        iou = np.full(len(pres) - 1, 1.0)
        for i in range(len(pres) - 1):
            m0, m1 = sub[pres[i]], sub[pres[i + 1]]
            inter = int((m0 & m1).sum())
            union = int((m0 | m1).sum())
            iou[i] = inter / max(union, 1)
        K = len(pres)
        cand = []
        for i in range(len(iou)):
            lo = max(0, i - w)
            hi = min(len(iou), i + w + 1)
            nbr = np.concatenate([iou[lo:i], iou[i + 1 : hi]])
            if len(nbr) == 0:
                continue
            drop = float(np.median(nbr)) - iou[i]
            islocalmin = (i == 0 or iou[i] <= iou[i - 1]) and (
                i == len(iou) - 1 or iou[i] <= iou[i + 1]
            )
            if not (drop > drop_thr and islocalmin):
                continue
            if i + 2 < len(pres):
                mb, ma = sub[pres[i]], sub[pres[i + 2]]
                rec = int((mb & ma).sum()) / max(int((mb | ma).sum()), 1)
                if rec > recover:
                    continue
            cand.append((drop, i + 1, iou[i], float(np.median(nbr))))
        cand.sort(reverse=True)
        accepted = []
        for drop, pos, iouv, nm in cand:
            if pos < min_frag or (K - pos) < min_frag:
                continue
            if all(abs(pos - a) >= min_frag for a in accepted):
                accepted.append(pos)
                if verbose or L in trace:
                    print(
                        f"  cut {L} @z{z0 + pres[pos]}: IoU {iouv:.2f} "
                        f"nbr-med {nm:.2f} drop {drop:.2f}",
                        flush=True,
                    )
        cuts = [pres[p] for p in accepted]
        if not cuts:
            continue
        cutset = set(cuts)
        segidx = np.zeros(sub.shape[0], np.int32)
        s = 0
        for zi in range(sub.shape[0]):
            if zi in cutset:
                s += 1
            segidx[zi] = s
        for ss in range(1, s + 1):
            newL = next_id
            next_id += 1
            for zi in np.where(segidx == ss)[0]:
                gy, gx = np.where(sub[zi])
                seg[z0 + zi, y0 + gy, x0 + gx] = newL
        n_cut += s
    return seg, n_cut


def _tubeness(seg, S, zr, cache):
    """Median consecutive-slice shifted-IoU along ``S``."""
    if S in cache:
        return cache[S]
    # Read inside S's own bbox, padded by MAX_SHIFT so the np.roll in
    # _shifted_iou still shifts into empty space rather than wrapping around a
    # tight window -- with that halo the result equals the full-slice value.
    z0, z1, y0, y1, x0, x1 = zr[S]
    pad = MAX_SHIFT + 1
    wy0, wy1 = max(y0 - pad, 0), min(y1 + 1 + pad, seg.shape[1])
    wx0, wx1 = max(x0 - pad, 0), min(x1 + 1 + pad, seg.shape[2])
    prev, ious = None, []
    for z in range(z0, z1 + 1):
        m = seg[z, wy0:wy1, wx0:wx1] == S
        if not m.any():
            continue
        if prev is not None:
            ious.append(_shifted_iou(prev, m, max_shift=MAX_SHIFT))
        prev = m
    v = float(np.median(ious)) if ious else 0.0
    cache[S] = v
    return v


def _vel(seg, S, z_end, d, zr, n=5):
    pts = []
    for k in range(n):
        z = z_end - d * k
        if zr[S][0] <= z <= zr[S][1] and (seg[z] == S).any():
            pts.append((z, *center_of_mass(seg[z] == S)))
    if len(pts) < 2:
        return np.array([0.0, 0.0])
    pts = np.array(pts)
    dz = pts[0, 0] - pts[-1, 0]
    return (pts[0, 1:] - pts[-1, 1:]) / (dz if dz else 1)


def detect_confident(seg, zr, sizes, verbose=False, host_both=False):
    Z = seg.shape[0]
    cands = [
        int(L)
        for L in np.unique(seg)
        if L > 0
        and zr[L][1] > zr[L][0]
        and (sizes[L] >= CAND_MIN or (zr[L][1] - zr[L][0] + 1) >= CAND_SPAN)
    ]
    ups, downs = defaultdict(list), defaultdict(list)
    tcache = {}
    for S in cands:
        for z_end, d in ((zr[S][1], +1), (zr[S][0], -1)):
            zn = z_end + d
            if not (0 <= zn < Z):
                continue
            sm = seg[z_end] == S
            sa = int(sm.sum())
            if sa < 50:
                continue
            u, c = np.unique(seg[zn][sm], return_counts=True)
            uc = [(int(l), int(n)) for l, n in zip(u, c) if l > 0 and l != S]  # noqa: E741
            if not uc:
                continue
            H, _ = max(uc, key=lambda t: t[1])
            if sizes[H] < HOST_MIN:
                continue
            iomin, iomax = _shifted_iomm(sm, seg[zn] == H, max_shift=MAX_SHIFT)
            if (
                iomin >= IOMIN_HI
                and iomax <= IOMAX_LO
                and _tubeness(seg, S, zr, tcache) >= TUBE_THR
            ):
                (ups if d == +1 else downs)[H].append(
                    (
                        S,
                        z_end,
                        np.array(center_of_mass(sm)),
                        sa,
                        _vel(seg, S, z_end, d, zr),
                    )
                )

    ha = lambda H, z: int((seg[z] == H).sum())  # noqa: E731
    found = []
    for H in set(ups) & set(downs):
        for S1, z1, c1, a1, v1 in ups[H]:
            for S2, z2, c2, a2, v2 in downs[H]:
                if S2 == S1 or not (1 <= z2 - z1 <= MAX_GAP):
                    continue
                if not (1 / CAL_RATIO <= a1 / max(a2, 1) <= CAL_RATIO):
                    continue
                zm = (z1 + z2) / 2
                p1 = c1 + v1 * (zm - z1)
                p2 = c2 + v2 * (zm - z2)
                off = float(np.hypot((p1[0] - p2[0]) * 9, (p1[1] - p2[1]) * 9))
                if off > COLLINEAR_NM:
                    continue
                below = zr[H][0] < z1 - 1
                above = zr[H][1] > z2 + 1
                need = (below and above) if host_both else (below or above)
                if not need:
                    continue
                cal = 0.5 * (a1 + a2)
                si = ha(H, z1 + 1) - ha(H, z1)
                so = ha(H, z2 - 1) - ha(H, z2)
                ok_lo = (STEP_LO <= si / cal <= STEP_HI) if below else True
                ok_hi = (STEP_LO <= so / cal <= STEP_HI) if above else True
                if not (ok_lo and ok_hi):
                    continue
                found.append(
                    dict(
                        H=H,
                        S1=S1,
                        S2=S2,
                        z1=z1,
                        z2=z2,
                        c1=c1,
                        c2=c2,
                        a1=a1,
                        a2=a2,
                        off=off,
                        cal=cal,
                    )
                )
    best = {}
    for f in found:
        k = (min(f["S1"], f["S2"]), max(f["S1"], f["S2"]))
        if k not in best or f["off"] < best[k]["off"]:
            best[k] = f
    return sorted(best.values(), key=lambda f: f["off"])


def gather_anchors(seg, f, zr, sizes_g):
    """Gather caliber-sized in-gap waypoints lying tightly on the S1-S2 line."""
    z1, z2, c1, c2, H, cal = (
        f["z1"],
        f["z2"],
        f["c1"],
        f["c2"],
        f["H"],
        f["cal"],
    )
    anchors = {z1: c1, z2: c2}
    seen = set()
    for z in range(z1 + 1, z2):
        best = None
        for L in np.unique(seg[z]).tolist():
            if L <= 0 or L == H or L in (f["S1"], f["S2"]):
                continue
            if sizes_g[L] > 4 * cal:
                continue
            m = seg[z] == L
            a = int(m.sum())
            if not (50 <= a <= 2 * cal):
                continue
            c = np.array(center_of_mass(m))
            t = (z - z1) / max(z2 - z1, 1)
            pl = c1 + (c2 - c1) * t
            d = float(np.hypot((c[0] - pl[0]) * 9, (c[1] - pl[1]) * 9))
            if d <= ANCHOR_NM and (best is None or d < best[0]):
                best = (d, c, L)
        if best:
            anchors[z] = best[1]
            seen.add(best[2])
    return dict(sorted(anchors.items())), seen


def _traj(anchors, z):
    zs = sorted(anchors)
    if z <= zs[0]:
        return anchors[zs[0]]
    if z >= zs[-1]:
        return anchors[zs[-1]]
    for i in range(1, len(zs)):
        if zs[i] >= z:
            z0, z1 = zs[i - 1], zs[i]
            t = (z - z0) / (z1 - z0)
            return anchors[z0] + (anchors[z1] - anchors[z0]) * t


def split_pair(seg, f, zr, sizes, verbose=False):
    """Carve the tube out of its host along the two-sided anchor trajectory."""
    Z, Y, X = seg.shape  # noqa: F841
    S1, S2, H, z1, z2, cal = (
        f["S1"],
        f["S2"],
        f["H"],
        f["z1"],
        f["z2"],
        f["cal"],
    )
    anchors, anchor_ids = gather_anchors(seg, f, zr, sizes)
    anchor_zs = set(anchors) - {z1, z2}
    tmpl = binary_erosion(seg[z1] == S1, iterations=1)
    if not tmpl.any():
        tmpl = seg[z1] == S1
    tc = np.array(center_of_mass(tmpl))
    carved = []
    abort = ""
    for z in range(z1 + 1, z2):
        if z in anchor_zs:
            continue
        pm = seg[z] == H
        if not pm.any():
            abort = f"z{z} host absent"
            break
        C = _traj(anchors, z)
        dy, dx = int(round(C[0] - tc[0])), int(round(C[1] - tc[1]))
        seed = np.roll(np.roll(tmpl, dy, 0), dx, 1) & pm
        if not seed.any():
            yy, xx = int(round(C[0])), int(round(C[1]))
            if 0 <= yy < Y and 0 <= xx < X and pm[yy, xx]:
                seed = np.zeros_like(pm)
                seed[yy, xx] = True
                seed = binary_dilation(seed, iterations=2) & pm
            if not seed.any():
                abort = f"z{z} seed empty (C={tuple(round(x) for x in C)})"
                break
        y0, y1e, x0, x1e = _bbox2d(pm)
        M = 3
        y0, y1e, x0, x1e = (
            max(y0 - M, 0),
            min(y1e + M, Y),
            max(x0 - M, 0),
            min(x1e + M, X),
        )
        pmc = pm[y0:y1e, x0:x1e]
        sc = seed[y0:y1e, x0:x1e]
        ma = pmc & binary_dilation(sc, iterations=1)
        mb = pmc & ~binary_dilation(sc, iterations=3)
        ma = ma & ~(ma & mb)
        if not (ma.any() and mb.any()):
            abort = (
                f"z{z} marker fail ma{int(ma.sum())} mb{int(mb.sum())} "
                f"pm{int(pmc.sum())} seed{int(sc.sum())}"
            )
            break
        mk = np.zeros(pmc.shape, np.int32)
        mk[mb] = 2
        mk[ma] = 1
        cc = watershed(-distance_transform_edt(pmc), mk, mask=pmc) == 1
        lab, ncc = cc_label(cc)
        if ncc > 1:
            cc = lab == (int(np.bincount(lab.ravel())[1:].argmax()) + 1)
        if not cc.any() or int(cc.sum()) > DRIFT_HI * cal:
            abort = f"z{z} cc {int(cc.sum())} > {DRIFT_HI * cal:.0f} (cal {cal:.0f})"
            break
        ccf = np.zeros((Y, X), bool)
        ccf[y0:y1e, x0:x1e] = cc
        carved.append((z, ccf))
    if abort or (not carved and not anchor_ids):
        if verbose:
            print(
                f"  ABORT {S1}<->{S2} through {H}: "
                f"{abort or 'no carve'} (anchors {sorted(anchor_ids)})",
                flush=True,
            )
        return 0
    for z, cc in carved:
        seg[z][cc] = S1
    for L in list(anchor_ids) + [S2]:
        seg[seg == L] = S1
    if verbose:
        print(
            f"  split {S1}<->{S2} through {H} z{z1}-{z2} "
            f"({len(carved)} carve, anchors {sorted(anchor_ids)})",
            flush=True,
        )
    return len(carved) or 1


def confident_parallel_split(
    seg,
    verbose=False,
    host_both=False,
    *,
    stats=None,
    inplace=False,
):
    """Run the validated relaxed close-ended tunnel/parallel carve."""
    if not inplace:
        seg = np.array(seg, dtype=np.uint32, copy=True)
    zr, sizes = stats[:2] if stats is not None else seg_stats(seg)[:2]
    pairs = detect_confident(seg, zr, sizes, verbose=verbose, host_both=host_both)
    used, n = set(), 0
    for f in pairs:
        if f["S1"] in used or f["S2"] in used or f["H"] in used:
            continue
        if split_pair(seg, f, zr, sizes, verbose=verbose):
            used.add(f["S1"])
            used.add(f["S2"])
            n += 1
    return seg, n, pairs


def branch_split(
    aff,
    seg,
    *,
    drop_thr=DROP_THR,
    w=W,
    min_size=MIN_SIZE,
    min_frag=MIN_FRAG,
    recover=1.1,
    host_both=False,
    verbose=False,
    trace=frozenset(),
    stats=None,
    inplace=False,
):
    """Run all validated branch-splitting stages in their fixed research order.

    ``aff`` is accepted for the graph-op ``[raw, seg]`` contract. The validated
    split stages are geometric and intentionally do not inspect affinity values.
    """
    del aff
    if not inplace:
        seg = np.array(seg, dtype=np.uint32, copy=True)
    if stats is None:
        stats = seg_stats(seg)
    seg, n_cut = link_cut_change(
        seg,
        drop_thr=drop_thr,
        w=w,
        min_size=min_size,
        min_frag=min_frag,
        recover=recover,
        verbose=verbose,
        trace=trace,
        stats=stats,
        inplace=True,
    )
    if n_cut:
        stats = seg_stats(seg)
    seg, _, _ = confident_parallel_split(
        seg,
        verbose=verbose,
        host_both=host_both,
        stats=stats,
        inplace=True,
    )
    return seg
