import numpy as np
import torch


from torch_connectomics.data.dataset.misc import rebalance_binary_class   


def test_rebalance():
    data = np.zeros([50,50],np.uint8)
    data[:30]=1
    data[:,30:]=1
    data_c = data.copy()
    data = torch.from_numpy(data)
    # check: from_numpy doesn't change value
    print((data.data.numpy()!=data_c).sum())

    data_t = data.clone()
    weight_factor, weight = rebalance_binary_class(data)
    print((data.data.numpy()!=data_t.data.numpy()).sum())


if __name__ == '__main__':
    test_rebalance()
