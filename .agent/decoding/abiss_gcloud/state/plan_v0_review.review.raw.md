[major] The plan does not implement the required merge-criterion input. It assumes the existing chunked path uses `mean`, but if S1 selects `max` or another criterion, the plan only says to “restate” quality expectations. It must specify how the selected criterion is passed through both whole-volume and chunked decode, or explicitly define the required code/config changes for each S1 outcome.

[major] Multi-region fallback is missing. The task explicitly requires multi-zone and multi-region fallback, but the launcher plan only covers alternate zones, smaller shapes, and optional on-demand capacity. It does not specify bucket replication, cross-region input staging, cost handling, or region selection.

[major] Chunk-boundary semantics are underspecified. The plan does not define overlap/halo size, ownership of boundary voxels, cross-chunk label reconciliation, or how `CHUNKMAP_OUTPUT` is consumed to produce globally consistent labels. A work queue alone does not make independent chunk decodes equivalent.

[major] The worker-queue design is not executable yet: no queue backend, task schema, claim/lease mechanism, atomic completion marker, retry/idempotency rule, or missing-chunk validation is specified. These are essential for preemption recovery and interchangeable workers.

[major] The affinity writer is an unresolved architectural branch: “determine whether it exists; if not, add a writer” does not define the required output layout, chunk alignment, dtype, metadata, concurrent-write behavior, or validation. This is a substantial design decision and is explicitly identified as the largest uncertainty.

[major] GCS scratch behavior is left unresolved despite being central to the proposed design. The plan does not establish whether ABISS can safely and acceptably use GCS for scratch, nor provide the local-SSD fallback’s staging, synchronization, and output-upload workflow.

[minor] The equivalence criteria are partly ambiguous. “Boundary-crossing subset” and “whole-volume-interior subset” need precise object-selection and metric definitions, and “no worse ... by more than that tolerance” needs a formula. The tolerance should also specify whether VOI means split VOI, merge VOI, or total VOI.

[minor] Several verification checks lack tolerances or objective bounds: “label count in the expected order,” “max inside the `scale_sigmoid` range,” and sizing numbers “match the task’s table.” These need numeric ranges and allowed rounding/error.

[minor] The one-chunk smoke test is redundant with the whole-volume path and does not meaningfully validate chunk scheduling, boundary handling, or output assembly. It can remain, but it should be clearly labeled as a plumbing test rather than an S3 acceptance test.

[minor] Parallel correctness is underdefined. “Same segmentation” needs a comparison rule accounting for label renumbering and a specified tolerance, and should include interrupted/retried workers rather than only 1-worker versus 4-worker runs.

[minor] The plan does not specify how inference block coverage is proven complete before decode: expected block/chunk inventory, missing-block detection, duplicate-write handling, and affinity metadata consistency are absent.

[minor] The sizing report requirements are not fully executable. It names per-chunk RSS and cap headroom but does not define the RSS model or measurement method, selected chunk dimensions, worker concurrency, temporary storage, or whether GPU-hours include retries and preemption overhead.

[minor] The plan notes that meshing/publication will block the promised deliverable but leaves no explicit handoff artifact or acceptance boundary. At minimum, it should state whether S3/S4 deliver only decoded segmentation artifacts or a usable published volume.

READY: no