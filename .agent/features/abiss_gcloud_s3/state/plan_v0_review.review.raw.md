[major] The acceptance run does not specify an `AGG_THRESHOLD` value. Since the config has no default, implementation cannot run the equivalence test without making an additional decision.

[major] The second alignment-valid `CHUNK_SIZE` is unspecified. “A second alignment-valid size” is not executable as written; it should name the exact size and its corresponding storage-chunk pairing.

[major] The completeness guard is contradictory: it must inspect existing CloudVolume objects, but verification says config resolution performs no filesystem or subprocess I/O. The plan needs a concrete preflight/resolution boundary and invocation order.

[major] Completeness is defined as an object count comparison only. Equal counts can still hide a missing expected key plus an unexpected extra key. The guard must compare the exact expected key set, including edge chunks and prefix/schema handling.

[major] The S1 integrity check leaves the mapped threshold unspecified. It must state the compressed-to-probability threshold formula and the exact threshold used for both max decodes.

[major] The plan does not fully specify the exact tail clipping rule: it references `EPS` in an imported function but gives neither its value nor the precise clipping-before-logit behavior. That is insufficient for an artifact contract.

[major] Criterion support is not concrete enough. “`rlme`, `cs` as ABISS names them” does not define the accepted values, their exact stage names, or their threshold semantics. The config-resolution guard must enumerate the canonical supported mapping explicitly.

[minor] The VOI boundary definition needs edge conventions: whether planes at volume boundaries count, whether bounding boxes are half-open, and whether background label `0` participates in `B`/`I`.

[minor] “Chunk-size invariance” does not define the comparison precisely—pairwise chunked-vs-chunked VOI, or each chunked result versus the whole-volume reference.

[minor] The claimed 8× larger-volume reuse is asserted but not given a concrete smoke-test/configuration path, especially since the converter is explicitly described as non-scaling.

READY: no