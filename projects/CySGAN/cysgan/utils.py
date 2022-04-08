from connectomics.data.dataset.collate import TrainBatchRecon, TrainBatchReconOnly

def collate_fn_trainX(batch):
    return TrainBatchRecon(batch)

def collate_fn_trainY(batch):
    return TrainBatchReconOnly(batch)
