# NCCNet

**GitHub:** https://github.com/seung-lab/NCCNet
**Language:** Python | **Stars:** 40

Weakly supervised deep metric learning for template matching using normalized cross correlation (NCC). Trains siamese convolutional networks to maximize contrast between NCC values of true and false matches, improving robustness without requiring true match locations during training.

## Key Features
- Weakly supervised siamese network training for NCC template matching
- Reduces false matches compared to parameter-tuned bandpass filters
- Designed for serial section EM image registration at petascale
- Docker-based workflow with TensorBoard logging

## API
```python
# Training
python src/train.py  # uses hparams.json config
# Inference
python infer.py A B  # A=first slice, B=second slice
```

## Relevance to Connectomics
Improves EM section-to-section template matching needed for volume assembly in connectomics reconstruction pipelines.
