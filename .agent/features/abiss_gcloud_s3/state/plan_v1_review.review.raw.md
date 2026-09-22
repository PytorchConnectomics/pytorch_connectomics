[major] The plan does not specify how the required whole-volume `mean` reference is produced. The existing volume is known to have a whole-volume decode, but that is the published `max` result and explicitly cannot be used as the reference. `run_volume.sh DECODE=whole|chunked` is named, but its whole-volume-`mean` command, inputs, output artifact, and invocation before equivalence testing are undefined. Coding this requires a workflow decision.

[major] The config/artifact paths remain placeholders (`<test_out>`, `<volume>`, `<run_prefix>`, `BBOX`) rather than concrete values or a stated source of those values. Consequently, the preflight cannot be implemented or run without deciding the exact test-volume path, run prefix, bounding box, and whole/chunked output locations.

[minor] The unsupported-criterion error is required to name the available chunked stages, but the plan only explicitly requires the requested value, accepted set, and (for `max`) the missing-binary explanation. The error should explicitly include `agglomerate_mean_edge` (and any other accepted stage representation).

[minor] The plan says alignment and the multi-chunk guard occur in `resolve`, but does not explicitly guarantee that these checks run before CloudVolume allocation or layer creation in the actual decode entry point. The invocation ordering needs to be stated in the implementation contract.

READY: no