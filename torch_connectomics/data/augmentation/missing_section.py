import numpy as np
from .augmentor import DataAugment

class MissingSection(DataAugment):
    """Missing-section augmentation of image stacks
    
    Args:
        num_sections (int): number of missing sections.
        p (float): probability of applying the augmentation.
    """
    def __init__(self, num_sections=2, p=0.5):
        super(MissingSection, self).__init__(p=p)
        self.num_sections = 2
        self.set_params()

    def set_params(self):
        self.sample_params['add'] = [self.num_sections, 0, 0]

    def missing_section(self, data, random_state):
        images, labels = data['image'], data['label']
        new_images = images.copy()   
        new_labels = labels.copy()

        idx = random_state.choice(np.array(range(1, images.shape[0]-1)), self.num_sections, replace=False)

        new_images = np.delete(new_images, idx, 0)
        new_labels = np.delete(new_labels, idx, 0)

        return new_images, new_labels
    
    def __call__(self, data, random_state=None):
        if random_state is None:
            random_state = np.random.RandomState(1234)
        new_images, new_labels = self.missing_section(data, random_state)
        return {'image': new_images, 'label': new_labels}