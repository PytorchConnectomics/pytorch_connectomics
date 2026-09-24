# review_v0 — raw planner notes (in-session; planner == claude == this session)

Method: validated the artifact, read the `merge_chunks.cpp` diff against 312bf54, then ran V6 --
which the coder correctly reported NOT RUN -- on SLURM. The coder's transcript is at
state/code_v0.stdout.log / state/code_v0.last_message.md.

## V6, run by me: FAILED

Arm `dev/zebrafinch/nuc_z1_y7_x6/wssnap`, param = worst3 with NUC_PATH and deliberately WITHOUT
NUC_TABLE, so the only delta versus the `w2ctl` control is the watershed snap+split and the sibling
run's global-table change stays inert. WORKER_HOME=work2/abiss, rebuilt with
ABISS_WS_NUC_GUARD=ON (CMakeCache confirmed) and all pipeline targets current.

SLURM 2821533, FAILED after 00:03:43, in `STAGE ws`, composite chunk 1_0_2_2:

    ...
    populate maps in 0.00302 seconds
    This should not happen in b
    72057869184326525 72128237928471919
    0.999918 0.999898
    scripts/init.sh: line 5: 471130 Aborted (core dumped) "$@"
    composite_chunk_ws.sh: cannot .../build/ws2 param.txt 0.99999 0.00001 200 200 1_0_2_2

So no crop numbers exist. Everything V6 was meant to measure is unknown.

## Locating the invariant

`src/ws/merge_chunks.cpp:262-267` (pre-existing, unmodified):

    } else {
        needs_an_edge = true;
        if (descent[vbi[idx]] != high_threshold && descent[vbi[idx]] != conn[idx]) {
            std::cerr << "This should not happen in b" << std::endl;
            ... std::abort();
        }
        descent[vbi[idx]] = conn[idx];
    }

and the mirrored check "in a" at 242-246.

## Cause

The descent union site as implemented:

    if ((descent[v1] == val || descent[v2] == val)
        && (!nucleus_snap
            || ws_nuc_tags_compatible(nucleus_tags[v1], nucleus_tags[v2]))) {
        const F merged_descent = std::max(descent[v1], descent[v2]);
        nuc_union(sizes, nucleus_tags, sets, v1, v2);
        const ID vr = sets.find_set(v1);
        descent[vr] = merged_descent;
        ...
    }

Baseline had no tag clause and always unioned when the descent condition held. When the tag clause
is false the union is skipped, `descent[v1]`/`descent[v2]` are never reconciled, and the next
boundary pair touching that supervoxel trips the invariant above. The same applies to the plateau
(`same`) site.

The printed affinities 0.999918 / 0.999898 fall inside the band measured earlier for this data
(pairwise bottleneck ~0.999 between nuclei 275/319/373, `conflict_bottleneck.py`), so this is the
target case arriving at the wrong mechanism, not a rare edge.

## Assessment

This is a defect in plan v2 section D, which I wrote: it treated all four union sites in
merge_chunks.cpp as interchangeable "check tags before uniting". `try_merge` and the MST are
policy; `same`/plateau and descent are structure and maintain `descent[]`. The veto belongs only at
the former.

The fixtures did not catch it because the hierarchy fixture never produces a cross-tag pair at
plateau or descent strength.

Working tree: HEAD == 312bf54 (no commits); 10 modified files, new src/ws/{nucleus_tags,
nucleus_snap}.hpp.
