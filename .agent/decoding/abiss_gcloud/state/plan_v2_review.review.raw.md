Plan_v2 is improved, but it is not executable now.

[major] The `max` S1 branch remains an unresolved human decision. If S1 selects `max`, option (a) conflicts with §6’s requirement that both decoders use the same criterion: the existing whole-volume reference would be `max`, while chunked decode would be `mean`. The plan must specify whether to produce a new whole-volume `mean` reference or stop S3, and who makes that decision.

[major] The required uncompression precondition for `mean` is not executable. The plan specifies float16 affinity layout and a probability-space threshold, but does not specify where `scale_sigmoid` is inverted, which artifact is consumed by ABISS, or how that representation is validated.

[major] S4 inference distribution is underspecified. Decode has a proposed dispatcher, leases, retries, and barriers, but inference only has “one block writes one disjoint region” and a validator. There is no inference task manifest, claim/retry/resume protocol, missing-block detection during execution, or explicit barrier guaranteeing affinity completeness before decode.

[major] The dispatcher still has an implementation-blocking contract gap. The plan says its contract is inferred from one `$PARALLEL_CMD` call site and “should be re-read” against the pinned revision. That exact command/quoting/environment behavior must be fixed before replacing GNU parallel; otherwise the proposed seam is not yet executable.

[minor] The cross-region cost figures are not reproducible from the displayed decimal-GB sizes and the stated `$0.02/GiB` rate. Either label the sizes as GiB or show the conversion.

READY: no