Remaining findings:

- [minor] The cross-chunk-size comparison `VOI_total(C_A, C_B) ≤ 0.01` says “same masking” but does not define whether that means `W ≠ 0`, `M_B`, `M_I`, or separate masks. This is non-blocking; an implementer can reasonably use the common `W ≠ 0` mask.

- [minor] Non-degeneracy requires `|B| > 0` but not `|I| > 0`; the stated difference can otherwise be undefined or vacuous. This is non-blocking; the implementer can add an interior-label check or define empty-mask behavior.

No remaining major findings. Code v0 can start.

READY: yes