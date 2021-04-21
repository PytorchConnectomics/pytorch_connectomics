from __future__ import print_function, division
import numpy as np
import torch

__all__ = [
    'collate_fn_train',
    'collate_fn_test']

####################################################################
# Collate Functions
####################################################################


def collate_fn_train(batch):
    return TrainBatch(batch)


def collate_fn_test(batch):
    return TestBatch(batch)

####################################################################
# Custom Batch Class
####################################################################


class TrainBatch:
    def __init__(self, batch):
        self._handle_basics(*zip(*batch))

    def _handle_basics(self, pos, out_input, out_target, out_weight):
        self.pos = pos
        self.out_input = torch.from_numpy(np.stack(out_input, 0))

        out_target_l = [None]*len(out_target[0])
        out_weight_l = [[None]*len(out_weight[0][x])
                        for x in range(len(out_weight[0]))]

        for i in range(len(out_target[0])):
            out_target_l[i] = np.stack([out_target[x][i]
                                        for x in range(len(out_target))], 0)
            out_target_l[i] = torch.from_numpy(out_target_l[i])

        # each target can have multiple loss/weights
        for i in range(len(out_weight[0])):
            for j in range(len(out_weight[0][i])):
                out_weight_l[i][j] = np.stack(
                    [out_weight[x][i][j] for x in range(len(out_weight))], 0)
                out_weight_l[i][j] = torch.from_numpy(out_weight_l[i][j])

        self.out_target_l = out_target_l
        self.out_weight_l = out_weight_l

    # custom memory pinning method on custom type
    def pin_memory(self):
        self._pin_basics()
        return self

    def _pin_basics(self):
        self.out_input = self.out_input.pin_memory()
        for i in range(len(self.out_target_l)):
            self.out_target_l[i] = self.out_target_l[i].pin_memory()
        for i in range(len(self.out_weight_l)):
            for j in range(len(self.out_weight_l[i])):
                self.out_weight_l[i][j] = self.out_weight_l[i][j].pin_memory()


class TestBatch:
    def __init__(self, batch):
        pos, out_input = zip(*batch)
        self.pos = pos
        self.out_input = torch.from_numpy(np.stack(out_input, 0))

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.out_input = self.out_input.pin_memory()
        return self
