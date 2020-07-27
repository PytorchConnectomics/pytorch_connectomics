## MitoEM Dataset: Large-scale 3D Mitochondria Instance Segmentation from EM Images

[[Project Page](https://donglaiw.github.io/page/mitoEM/index.html)] [[Tutorial](https://zudi-lin.github.io/pytorch_connectomics/build/html/tutorials/mitoem.html)]

### Introduction

Serial electron microscopy (EM) allows identification of intracellular organelles such as mitochondria, which provides novel insights for both clinical and scientific studies. However, the mitochondria reconstruction benchmark only contains around 100 instances that are well-separated and exhibit simple morphologies. Therefore, existing automatic methods that have achieved almost human-level performance on the small dataset usually fail to produce preferred results due to object diversity in appearance and morphologies.

To enable the development of robust models for large-scale biomedical analysis, we introduce **MitoEM**, a 3D mitochondria instance segmentation dataset consisting of two 30μm cubic volumes from human and rat cortices respectively, which are **3,600x** larger than the previous benchmark dataset. Our new dataset posts new challenges for existing state-of-the-art segmentation approaches as they consistently fail to generate object masks with quality on par with expert annotators. With approximately 40k mitochondria in our new dataset, we provide in-depth analysis of the dataset properties, as well as the performance of different combinations of deep learning models and post-processing methods. The MitoEM dataset and our comprehensive analysis will enable further researches in large-scale instance segmentation and a better understanding of mammalian brains.

The configuration files in this folder can be used to produce the 3D mitochondria instance segmentation with higher or comparable performance as reported in the paper.

### Citation

```bibtex
@inproceedings{wei2020mitoem,
  title={MitoEM Dataset: Large-scale 3D Mitochondria Instance Segmentation from EM Images},
  author={D. Wei, Z. Lin, D. Barranco, N. Wendt, X. Liu, W. Yin, X. Huang, A. Gupta,
  W. Jang, X. Wang,  I. Arganda-Carreras, J. Lichtman, H. Pfister},
  booktitle={International Conference on Medical Image Computing and Computer Assisted Intervention},
  year={2020}
}
```

### Acknowledgement

This work has been partially supported by NSF award IIS-1835231 and NIH award 5U54CA225088-03.