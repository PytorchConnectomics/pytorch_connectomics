## MitoEM Dataset: Large-scale 3D Mitochondria Instance Segmentation from EM Images

[[**Challenge Page**](https://mitoem.grand-challenge.org/)] [[Project Page](https://donglaiw.github.io/page/mitoEM/index.html)] [[Tutorial](https://zudi-lin.github.io/pytorch_connectomics/build/html/tutorials/mito.html#instance-segmentation)]

### Introduction

Serial electron microscopy (EM) allows identification of intracellular organelles such as mitochondria, which provides novel insights for both clinical and scientific studies. However, the mitochondria reconstruction benchmark only contains around 100 instances that are well-separated and exhibit simple morphologies. Therefore, existing automatic methods that have achieved almost human-level performance on the small dataset usually fail to produce preferred results due to object diversity in appearance and morphologies.

To enable the development of robust models for large-scale biomedical analysis, we introduce **MitoEM**, a 3D mitochondria instance segmentation dataset consisting of two 30μm cubic volumes from human and rat cortices respectively, which are **3,600x** larger than the previous benchmark dataset. Our new dataset posts new challenges for existing state-of-the-art segmentation approaches as they consistently fail to generate object masks with quality on par with expert annotators. With approximately 40k mitochondria in our new dataset, we provide in-depth analysis of the dataset properties, as well as the performance of different combinations of deep learning models and post-processing methods. The MitoEM dataset and our comprehensive analysis will enable further researches in large-scale instance segmentation and a better understanding of mammalian brains.

The configuration files in this folder can be used to produce the 3D mitochondria instance segmentation with higher or comparable performance as reported in the paper.

### Notes

We use [**TileDataset**](https://zudi-lin.github.io/pytorch_connectomics/build/html/_modules/connectomics/data/dataset/dataset_tile.html#TileDataset) for data loading because the training volumes (400x4096x4096) are too large to be directly loaded into memory. The **TileDataset** class reads a JSON file containing the path of the images. We provide examples for MitoEM-R training images ([```im_train.json```](https://github.com/zudi-lin/pytorch_connectomics/blob/master/configs/MitoEM/im_train.json)) and labels ([```mito_train.json```](https://github.com/zudi-lin/pytorch_connectomics/blob/master/configs/MitoEM/mito_train.json)). Please change the path to your own data directory.

A simple Python script for changing the MitoEM-R image paths in the JSON file:

```python
import json

js_path = 'im_train.json'
my_path = '/path/to/MitoEM-R/'
with open(js_path, 'r') as fp:
    data = json.load(fp)

for i in range(len(data['image'])):
    x = data['image'][i]
    x = x.strip().split('/')
    data['image'][i] = my_path+'/'.join(x[-2:])

with open(js_path, 'w') as fp:
    json.dump(data, fp)
```

The **TileDataset** generates prediction on small chunks sequentially, which produces
multiple ``*.h5`` files with the coordinate information. To merge the chunks into a single volume
and apply the segmentation algorithm:

```python
import glob
import numpy as np
from connectomics.data.utils import readvol
from connectomics.utils.processing import bc_watershed

output_files = 'outputs/MitoEM_R_BC/test/*.h5' # output folder with chunks
chunks = glob.glob(output_files)

vol_shape = (2, 500, 4096, 4096) # MitoEM test set
pred = np.ones(vol_shape, dtype=np.uint8)
for x in chunks:
    pos = x.strip().split("/")[-1]
    print("process chunk: ", pos)
    pos = pos.split("_")[-1].split(".")[0].split("-")
    pos = list(map(int, pos))
    chunk = readvol(x)
    pred[:, pos[0]:pos[1], pos[2]:pos[3], pos[4]:pos[5]] = chunk

# This function process the array in numpy.float64 format.
# Please allocate enough memory for processing.
segm = bc_watershed(pred, thres1=0.85, thres2=0.6, thres3=0.8, thres_small=1024)
```

### Citation

This paper is published at the 23rd International Conference on Medical Image Computing and Computer Assisted Intervention (**MICCAI 2020**).

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
