## Run prediction using a pretrained model

The base configuration file `JWR15-Synapse-Base.yaml` is for large-scale inference using the JSON dataset. To run inference using a pretrained model on an arbitrary volume, using the following command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u scripts/main.py --inference \
--config-base configs/JWR15/synapse/JWR15-Synapse-Base.yaml \
--config-file configs/JWR15/synapse/JWR15-Synapse-BCE.yaml \
--checkpoint outputs/outputs/JWR15_Syn_BCE/checkpoint_100000.pth.tar \
INFERENCE.IMAGE_NAME <ABSOLUTE_DATA_PATH> INFERENCE.IS_ABSOLUTE_PATH True \
INFERENCE.DO_CHUNK_TITLE 0
```

If the model is trained using the semantic segmentation configuration with exclusive pre- and post-synaptic masks, just change 
the config file to `JWR15-Synapse-Semantic-CE`.
