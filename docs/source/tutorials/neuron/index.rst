Neuron Segmentation
=====================

Dense neuron segmentation in EM is an **instance segmentation** task. The
canonical pipeline first predicts an affinity map (the connectivity of each
voxel to its neighbors) with an encoder-decoder, then converts the affinity
map into a segmentation via watershed or a similar algorithm. Evaluation
uses the
`Rand Index <https://en.wikipedia.org/wiki/Rand_index>`_ and
`Variation of Information <https://en.wikipedia.org/wiki/Variation_of_information>`_.

This section covers two benchmarks:

- **SNEMI3D** — the classic small isotropic-anisotropic benchmark, used for
  end-to-end affinity training and waterz post-processing.
- **NISB / BANIS** — a larger, anisotropic benchmark with reproduction
  targets in ``tutorials/neuron_nisb/``.

.. toctree::
   :maxdepth: 1

   snemi3d
   nisb
