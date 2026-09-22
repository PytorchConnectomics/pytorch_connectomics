[major] The namespace fix is not propagated into Verification Plan 3(a), which still unconditionally requires “each enumerated node → one distinct nonzero id.” As a hard check, that contradicts S1 and would fail after any successful union. Explicitly scope 3(a) to the zero-union remap and state the root-wise linked-output assertions there.

[minor] S6’s certificate decision enum cannot represent the required `find(u)==find(v)` no-op. Add `same_root_noop`, or explicitly define how it is recorded without misclassifying it as an accepted union or rejection.

The best-other-partner ambiguity formula is correct. Canonicalization gives one union attempt per undirected pair, subject to the certificate-state omission above. No other blocker is apparent.

READY: no