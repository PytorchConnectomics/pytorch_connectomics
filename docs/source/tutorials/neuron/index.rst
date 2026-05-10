Neuron Segmentation
=====================

Dense neuron segmentation in EM is an **instance segmentation** task. The
canonical pipeline first predicts an affinity map (the connectivity of each
voxel to its neighbors) with an encoder-decoder, then converts the affinity
map into a segmentation via watershed or a similar algorithm.

This section covers two benchmarks:

- **SNEMI3D** — the classic small isotropic-anisotropic benchmark, used
  for end-to-end affinity training and waterz post-processing. Evaluated
  with `Rand Index <https://en.wikipedia.org/wiki/Rand_index>`_ and
  `Variation of Information <https://en.wikipedia.org/wiki/Variation_of_information>`_.
- **NISB** — a larger, anisotropic neuron-segmentation benchmark
  evaluated with the **NERL** skeleton metric. Reproduction targets in
  ``tutorials/neuron_nisb/`` mirror the upstream BANIS pipeline.

.. toctree::
   :maxdepth: 1

   snemi3d
   nisb
