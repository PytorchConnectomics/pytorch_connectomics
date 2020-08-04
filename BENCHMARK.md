# Benchmark Results in Connectomics

## SNEMI 3D Neuron Segmentation

## CREMI Synaptic Cleft Segmentation

## Lucchi Mitochondria Segmentation

The model is trained using the [Lucchi-Mitochondria.yaml](https://github.com/zudi-lin/pytorch_connectomics/blob/master/configs/Lucchi-Mitochondria.yaml) configuration file. A detailed tutorial of this task can be found [here](https://zudi-lin.github.io/pytorch_connectomics/build/html/tutorials/mito.html#semantic-segmentation).

* **Dataset**: the test volume of 165x1024x768 voxels.
* **Evaluation Metric**: Foreground IoU and IoU, both are multiplied by 100.
* **Pre-trained Model**: please click this [[link](https://drive.google.com/uc?export=download&id=1BI05iDGUoDCgykv1giET7qEZVxUpy2sb)] to download the pre-trained weights.

| Filter/Threshold 	|      0.4     	|      0.5     	|      0.6     	|      0.7     	|      0.8     	|      0.9     	|     0.95     	|
|------------------	|:------------:	|:------------:	|:------------:	|:------------:	|:------------:	|:------------:	|:------------:	|
| **No Filtering**     	| 86.18, 92.65 	| 87.16, 93.18 	| 88.02, 93.64 	| 88.63, 93.97 	| 89.07, 94.21 	| 89.05, 94.20 	| 88.31, 93.81 	|
| **(7,7,7) Median**   	| 86.63, 92.90 	| 87.54, 93.39 	| 88.33, 93.80 	| 88.86, 94.10 	| **89.18, 94.27** 	| 88.96, 94.15 	| 88.01, 93.66 	|