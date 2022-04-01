from typing import Optional

from .augmentor import DataAugment
import numpy as np
import torch
import torchvision.transforms.functional as tf
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage.morphology import generate_binary_structure

class CopyPasteAugmentor(DataAugment):
    r"""Copy-paste augmentor (experimental).

    The input can be a `numpy.ndarray` or `torch.Tensor` of shape :math:`(C, Z, Y, X)` or :math:`(Z, Y, X)`.

    Args:
        aug_thres: Maximum fractional size of the object occupying the volume. If
                   the object is too large it is not augmented. Default: 0.7
    """
    def __init__(self,
                 aug_thres: float = 0.7,
                 p: float = 0.8,
                 additional_targets: Optional[dict] = {'label': 'mask'},
                 skip_targets: list = []):
        assert additional_targets is not None and 'label' in additional_targets.keys(), \
            "Copy paste augmentation needs segmentation labels to work"
        super().__init__(p, additional_targets, skip_targets)
        self.aug_thres = aug_thres
        self.dil_struct = generate_binary_structure(3,3)

    def set_params(self):
        '''Doesn't change sample size'''
        pass

    def __call__(self, sample, random_state=np.random.RandomState()):
        assert 'label' in sample.keys(), "Labels not found in sample"
        volume, label = sample['image'], sample['label']
        if not isinstance(volume, (torch.Tensor, np.ndarray)):
            raise TypeError("Type {} is not supported in CopyPasteAugmentor".format(type(volume)))

        is_np = isinstance(volume, np.ndarray)
        label = torch.from_numpy(label.copy()).bool() if isinstance(label, np.ndarray) else label.bool()
        if is_np: volume = torch.from_numpy(volume.copy())
        assert label.ndim == 3 and (volume.ndim == 4 or volume.ndim == 3), "CopyPaste doesn't work on batched data"
        label_flipped = label[torch.arange(label.shape[0]-1,-1,-1)] #flip on z-axis
        if label.float().mean() <= self.aug_thres:
            neuron_tensor = volume * label
            neuron_tensor, label = self.copy_paste_single(torch.stack([label, label_flipped]), neuron_tensor)
            volume = volume * (~label) + neuron_tensor * label

        return np.array(volume) if is_np else volume

    def rotate(self, tensor, angle):
        c,z,y,x = tensor.shape
        rotated = tf.rotate(tensor.reshape(1,c*z,y,x), angle)
        return rotated.reshape(c,z,y,x)

    def crop_overlap(self, rot_label, gt, border=5):
        y_any = gt.any(dim=2).any(dim=0)
        x_any = gt.any(dim=1).any(dim=0)
        x1,x2,y1,y2 = *torch.where(x_any)[0][[0,-1]], *torch.where(y_any)[0][[0,-1]]
        x1,x2,y1,y2 = torch.clamp(x1-border,min=0),x2+border,torch.clamp(y1-border,min=0),y2+border

        return_dict = {}
        return_dict[rot_label[...,:x1].int().sum()] = [slice(None), slice(None,None), slice(x1, None)]
        return_dict[rot_label[...,x2:].int().sum()] = [slice(None), slice(None,None), slice(None, x2)]
        return_dict[rot_label[...,:y1].int().sum()] = [slice(None), slice(y1, None), slice(None,None)]
        return_dict[rot_label[...,y2:].int().sum()] = [slice(None), slice(None, y2), slice(None,None)]
        return return_dict[max(return_dict.keys())]

    def crop_overlap_dil(self, rot_label, gt, border=3):
        gt = torch.tensor(binary_dilation(gt, structure = self.dil_struct, iterations=border))
        return torch.where(gt)

    def distance(self, rot_label, orig_label, shape=None):
        orig_center = torch.stack(torch.where(orig_label)).float().mean(dim=-1)
        rot_center = torch.stack(torch.where(rot_label)).float().mean(dim=-1)
        if shape is not None:
            orig_center, rot_center = orig_center/torch.tensor(shape), rot_center/torch.tensor(shape)
        return ((rot_center-orig_center)**2).mean()

    def copy_paste_single(self, rot_label, neuron_tensor):
        '''
        Find rotation with least overlap with GT and if there are
        multiple rotations with no overlap, find one with least
        distance from GT
        '''
        gt = rot_label[0]
        min_overlap = torch.logical_and(rot_label[1], gt).int().sum()
        min_dist = float('inf') if min_overlap else self.distance(rot_label[1], gt, gt.shape)
        rot_angle, crop, ind = 0, [slice(None,None), slice(None,None)], 1
        for angle in range(30, 360, 30):
            rotated = self.rotate(rot_label, angle)
            overlap0, overlap1 = torch.logical_and(rotated[0] , gt).int().sum(), \
                        torch.logical_and(rotated[1] , gt).int().sum()
            if min(min_overlap, overlap0, overlap1) == min_overlap:
                rot_dist0, rot_dist1 = self.distance(rotated[0], gt, gt.shape), \
                            self.distance(rotated[1], gt, gt.shape)
                if overlap0 == 0 and rot_dist0 < min_dist:
                    min_dist = rot_dist0
                    rot_angle, ind = angle, 0
                if overlap1 == 0 and rot_dist1 < min_dist:
                    min_dist = rot_dist1
                    rot_angle, ind = angle, 1
            elif min(min_overlap, overlap0, overlap1) == overlap0:
                min_overlap = overlap0
                rot_angle, ind = angle, 0
            else:
                min_overlap = overlap1
                rot_angle, ind = angle, 1

        rot_label = rot_label[ind].unsqueeze(0)
        if ind: # flip
            neuron_tensor = neuron_tensor[torch.arange(neuron_tensor.shape[0]-1,-1,-1)]
        rot_label, neuron_tensor = self.rotate(rot_label, rot_angle).squeeze(0), \
                                    self.rotate(neuron_tensor.unsqueeze(0), rot_angle).squeeze(0)
        crop = self.crop_overlap_dil(rot_label.squeeze(), gt)
        neuron_tensor[crop[0],crop[1],crop[2]], rot_label[crop[0],crop[1],crop[2]] = 0, False
        return neuron_tensor, rot_label
