## NucMM Dataset: 3D Neuronal Nuclei Instance Segmentation at Sub-Cubic Millimeter Scale

[[**Paper**](https://link.springer.com/chapter/10.1007/978-3-030-87193-2_16)] [[arXiv](https://arxiv.org/abs/2107.05840)] [[**Challenge Page**](https://nucmm.grand-challenge.org/)] [[Project Page](https://connectomics-bazaar.github.io/proj/nucMM/index.html)]

### Abstract

Segmenting 3D cell nuclei from microscopy image volumes is critical for biological and clinical analysis, enabling the study of cellular expression patterns and cell lineages. However, current datasets for neuronal nuclei usually contain volumes smaller than 10^-3 mm^3 with fewer than 500 instances per volume, unable to reveal the complexity in large brain regions and restrict the investigation of neuronal structures. In this paper, we have pushed the task forward to the sub-cubic millimeter scale and curated the **NucMM** dataset with two fully annotated volumes: one 0.1 mm^3 electron microscopy (EM) volume containing nearly the entire zebrafish brain with around 170,000 nuclei; and one 0.25 mm^3 micro-CT (uCT) volume containing part of a mouse visual cortex with about 7,000 nuclei. With two imaging modalities and significantly increased volume size and instance numbers, we discover a great diversity of neuronal nuclei in appearance and density, introducing new challenges to the field. We also perform a statistical analysis to illustrate those challenges quantitatively. To tackle the challenges, we propose a novel hybrid-representation learning model that combines the merits of foreground mask, contour map, and signed distance transform to produce high-quality 3D masks. The benchmark comparisons on the NucMM dataset show that our proposed method significantly outperforms state-of-the-art nuclei segmentation approaches.

### Notes

The config files in this folder can reproduce the results of both **U3D-B** (bianry mask and contour map) and our proposed **U3D-BCD** (bianry mask, contour map and distance transform) models on two NucMM volumes.

### Citation

This paper is published at the 24th International Conference on Medical Image Computing and Computer Assisted Intervention (**MICCAI 2021**).

```bibtex
@inproceedings{lin2021nucmm,
  title={NucMM Dataset: 3D Neuronal Nuclei Instance Segmentation at Sub-Cubic Millimeter Scale},
  author={Lin, Zudi and Wei, Donglai and Petkova, Mariela D and Wu, Yuelong and Ahmed, Zergham and Zou, Silin and Wendt, Nils and Boulanger-Weill, Jonathan and Wang, Xueying and Dhanyasi, Nagaraju and others},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={164--174},
  year={2021},
  organization={Springer}
}
```

### Acknowledgement

This work has been partially supported by NSF award IIS-1835231 and NIH award U19NS104653.
