import unittest
import torch
import numpy as np

from connectomics.model.loss import Criterion


class TestCriterion(unittest.TestCase):

    def test_regularization(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        criterion = Criterion(device, target_opt=['0', '6'],
                              loss_opt=[['WeightedBCEWithLogitsLoss',
                                         'DiceLoss'], ['WeightedMSE']],
                              output_act=[['none', 'sigmoid'], ['tanh']],
                              loss_weight=[[1.0, 1.0], [1.0]],
                              regu_opt=['FgDT'],
                              regu_target=[[0, 1]],
                              regu_weight=[1.0])

        pred = torch.rand(2, 2, 4, 8, 8).to(device)
        target = [torch.ones(2, 1, 4, 8, 8), torch.rand(2, 1, 4, 8, 8)]
        weight = [[torch.ones(1), torch.ones(1)], [torch.ones(1)]]
        loss, _ = criterion(pred, target, weight)
        self.assertTrue(loss > 0.0)


if __name__ == '__main__':
    unittest.main()
