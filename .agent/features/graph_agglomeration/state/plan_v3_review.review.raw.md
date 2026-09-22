The plan is close, but two contradictions still prevent a semantics-free code_v0.

- [major] The ambiguity competitor remains inconsistent. “Second-highest raw among other partners, excluding `b`” can skip the strongest alternative, contradicting the stated best-competitor rule. Pin it as `max({raw(a,x) | x != b}, default=0)`—equivalently, the runner-up overall when `e` is top. This materially changes defer decisions.

- [major] The structural assertion requires a distinct ID per node, but accepted unions intentionally give multiple nodes the same ID. Scope node-wise injectivity to the zero-union control. For linked output, assert total nonzero mapping and `R(u) == R(v)` iff `find(u) == find(v)`, with distinct IDs across roots. Otherwise a literal implementation fails after its first accepted union.

- [minor] “All candidates (both directions)” does not establish whether a physical edge is certified and processed once or twice. Canonicalize each undirected pair once, or explicitly separate directed certificate records from one union attempt and define same-root no-op handling.

The remaining revisions are resolved: raw ranking is otherwise non-circular, endpoint aggregation/guard/clip are specified, the caliber floor is removed, the marker gate is numeric with no fallback, all 600 crossing schemas are hard-checked, and all generated candidates are retained. The supplied paths, dtype, pipeline ordering, and `--remap` evaluation are otherwise coherent.

READY: no