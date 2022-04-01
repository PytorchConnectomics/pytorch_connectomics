from __future__ import print_function, division
import numpy as np
import torch

__all__ = [
    'collate_fn_train',
    'collate_fn_test']

####################################################################
# Collate Functions
####################################################################

def collate_fn_test_cond(batch):
    return TestBatchCond(batch)

class TestBatchCond:
    def __init__(self, batch):
        self._handle_batch(*zip(*batch))

    def _handle_batch(self, pos, out_input, out_target):
        self.pos = pos
        self.out_input = torch.from_numpy(np.stack(out_input, 0))
        self.out_target_l = torch.from_numpy(np.stack(out_target, 0))

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.out_input = self.out_input.pin_memory()
        self.out_target_l = self.out_target_l.pin_memory()
        return self
