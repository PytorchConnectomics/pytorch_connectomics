# Task

`tutorials/neuron_nisb/base_banis.yaml` reproduces the baseline result with NERL ~24% at 50k iterations and ~32% at 200k iterations.

What are ideas to further improve the model? Examples to consider:

- Bigger model (e.g., MedNeXt-L)
- Multi-head outputs
- Better decoding than `cc3d` (e.g., `waterz`)
- SDT (signed distance transform) head
- Erosion on ground-truth labels to avoid false merges

Read `tutorials/*` for additional brainstorming inputs (cremi, mitoEM, neuron_snemi, neuron_liconn_mit, etc.) and survey decoder/loss/architecture options already supported by the codebase.

The deliverable is a brainstorm + ranked plan, not necessarily a single chosen approach. Each idea should include: hypothesis, expected NERL impact magnitude, implementation cost (config-only vs code change), and risks.
