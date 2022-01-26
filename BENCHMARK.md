# Benchmark Results in Connectomics

1. [Mitochondria (Lucchi)](#lucchi)
2. [Mitochondria (MitoEM)](#mitoem)
3. [Synaptic Clefts (CREMI)](#cremi)
4. [Neurons (SNEMI)](#snemi)

## Lucchi Mitochondria Semantic Segmentation <a name="lucchi"></a>

The model is trained using the [Lucchi-Mitochondria.yaml](https://github.com/zudi-lin/pytorch_connectomics/blob/master/configs/Lucchi-Mitochondria.yaml) configuration file. A detailed tutorial of this task can be found [here](https://connectomics.readthedocs.io/en/latest/tutorials/mito.html#semantic-segmentation). We show the performance comparison under different
thresholds, with or without post-processing (median filtering).

* **Dataset**: the test volume of size 165x1024x768 voxels.
* **Evaluation Metric**: Foreground IoU and IoU, both are multiplied by 100.

| Filter/Threshold 	|      0.4     	|      0.5     	|      0.6     	|      0.7     	|      0.8     	|      0.9     	|     0.95     	|
| :----------------     |:------------:	|:------------:	|:------------:	|:------------:	|:------------:	|:------------:	|:------------:	|
| **No Filtering**     	| 86.18, 92.65 	| 87.16, 93.18 	| 88.02, 93.64 	| 88.63, 93.97 	| 89.07, 94.21 	| 89.05, 94.20 	| 88.31, 93.81 	|
| **(7,7,7) Median**   	| 86.63, 92.90 	| 87.54, 93.39 	| 88.33, 93.80 	| 88.86, 94.10 	| **89.18, 94.27** 	| 88.96, 94.15 	| 88.01, 93.66 	|

***

## MitoEM Mitochondria Instance Segmentation <a name="mitoem"></a>

The model is trained using the [MitoEM-R-BC.yaml](https://github.com/zudi-lin/pytorch_connectomics/blob/master/configs/MitoEM/MitoEM-R-BC.yaml) configuration file which predicts both the foreground mask and instance contour map. A detailed tutorial of this task can be found [here](https://connectomics.readthedocs.io/en/latest/tutorials/mito.html#instance-segmentation). We show the performance comparison on the MitoEM-R validation set with different parameters in the [```bc_watershed```](https://zudi-lin.github.io/pytorch_connectomics/build/html/modules/utils.html#connectomics.utils.process.bc_watershed) function.

* **Dataset**: each of the two validation volumes has a shape of 100x4096x4096 voxels, and each of the two test volumes has a shape of 500x4096x4096 voxels. We only show the results on the MitoEM-Rat set for validation.
* **Evaluation Metric**: following the [**MitoEM Challenge**](https://mitoem.grand-challenge.org), we use the average precision (AP) metric with an IoU threshold of 0.75, which is denoted as **AP-75**.

**Validation Results**: for the three thresholds in the [```bc_watershed```](https://zudi-lin.github.io/pytorch_connectomics/build/html/modules/utils.html#connectomics.utils.processing.bc_watershed) function, we fix the foreground threshold *thres3=0.8*, and show the results (AP-75) of diffrent seed threshold (*thres1*) and contour threshold (*thres2*) on the MitoEM-Rat validation set:

| Thres1/Thres2 	|      0.6     	|      0.7     	|      0.75     	|      0.8     	|      0.85     	|      0.9     	|     0.95     	|
| :----------------     |:------------:	|:------------:	|:------------:	|:------------:	|:------------:	|:------------:	|:------------:	|
| **0.85**     	| **0.692** | 0.664 | 0.657 | 0.634 | 0.623 | 0.640 | 0.645 |
| **0.90**   	| 0.678 | 0.658 | 0.655 | 0.624 | 0.606 | 0.629 | 0.632 |
| **0.95**   	| 0.641 | 0.627 | 0.628 | 0.616 | 0.593 | 0.594 | 0.600 |

**Test Results**: we process the predictions on both MitoEM-R and MitoEM-H test sets with the best post-processing hyper-parameter on the MitoEM-R validation set, and submitted the 
results to the [**challenge website**](https://mitoem.grand-challenge.org/evaluation/challenge/leaderboard/) for evaluation. The test results are:

|     | MitoEM-R | MitoEM-H | Average |
| :-- | :--: | :--: | :--: |
| **AP-50** | 0.892 | 0.877 | 0.885 |
| **AP-75** | 0.816 | 0.804 | 0.810 |

The complete evaluation results of this submission can be found at this [page](https://mitoem.grand-challenge.org/evaluation/1a1757dd-c2d8-4aa4-9a01-8b8e504cde42/).

***

## CREMI Synaptic Cleft Detection <a name="cremi"></a>

TODO

***

## SNEMI Neuron Instance Segmentation <a name="snemi"></a>

TODO

***