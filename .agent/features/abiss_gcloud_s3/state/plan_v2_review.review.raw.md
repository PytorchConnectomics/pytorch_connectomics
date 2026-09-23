[major] The S1 integrity check is still not executable from the plan. It requires comparing a whole-volume `max` decode from `aff_prob` against the compressed-artifact result, but the plan provides no command or independent result path for that compressed decode. The published `max` layer is explicitly “provenance only, never read.” This blocks starting code_v0 because the required validation has no defined comparison artifact or procedure.

[major] The HDF5 twin’s consumption schema is underspecified. The plan names its path, dtype, and value transform, but not the dataset name, axis order, or exact layout expected by `run_abiss_volume.py`. “Shape, dtype and checksum” does not establish that both ABISS paths consume the same tensor semantics. This blocks starting code_v0 because the converter and whole-volume decoder still require a design decision.

[minor] `resolve` is declared pure and without I/O, but it also “asserts [BBOX] equals the affinity’s own dimensions.” Checking the actual affinity dimensions requires reading the source or generated artifact. An implementer can reasonably move that assertion into `preflight`; it does not block coding.

[minor] The acceptance section says the metric uses “interior planes only,” while the task defines the boundary set from labels whose bounding boxes intersect chunk planes and the interior set as all remaining labels. The plan should state precisely how these sets become masked volumes, but an implementer can reasonably settle that from the stated set definitions; it does not block coding.

READY: no