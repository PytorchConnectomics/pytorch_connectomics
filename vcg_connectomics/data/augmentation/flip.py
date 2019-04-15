import numpy as np
from .augmentor import DataAugment

class Flip(DataAugment):
    """
    Flip along z-, y- and x-axes as well as swap y- and x-axes
    """
    def __init__(self, p=0.5):
        super(Flip, self).__init__(p)

    def set_params(self):
        # No change in sample size
        pass

    def flip_and_swap(self, data, rule):
        assert data.ndim==3 or data.ndim==4
        if data.ndim == 3:
            # z reflection.
            if rule[0]:
                data = data[::-1, :, :]
            # y reflection.
            if rule[1]:
                data = data[:, ::-1, :]
            # x reflection.
            if rule[2]:
                data = data[:, :, ::-1]
            # Transpose in xy.
            if rule[3]:
                data = data.transpose(0, 2, 1)
        else:
            # z reflection.
            if rule[0]:
                data = data[:, ::-1, :, :]
            # y reflection.
            if rule[1]:
                data = data[:, :, ::-1, :]
            # x reflection.
            if rule[2]:
                data = data[:, :, :, ::-1]
            # Transpose in xy.
            if rule[3]:
                data = data.transpose(0, 1, 3, 2)
        return data
    
    def __call__(self, data, random_state):
        if random_state is None:
            random_state = np.random.RandomState(1234)
        
        output = {}
        rule = random_state.randint(2, size=4)
        augmented_image = self.flip_and_swap(data['image'], rule)
        augmented_label = self.flip_and_swap(data['label'], rule)
        output['image'] = augmented_image
        output['label'] = augmented_label

        return output