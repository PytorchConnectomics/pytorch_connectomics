from typing import Optional, List
import numpy as np
import random

import torch
import torch.utils.data
from ..augmentation import Compose
from ..utils import *

TARGET_OPT_TYPE = List[str]
WEIGHT_OPT_TYPE = List[List[str]]
AUGMENTOR_TYPE = Optional[Compose]


class VolumeDataset(torch.utils.data.Dataset):
    """
    Dataset class for volumetric image datasets. At training time, subvolumes are randomly sampled from all the large
    input volumes with (optional) rejection sampling to increase the frequency of foreground regions in a batch. At inference
    time, subvolumes are yielded in a sliding-window manner with overlap to counter border artifacts.

    Args:
        volume (list): list of image volumes.
        label (list, optional): list of label volumes. Default: None
        valid_mask (list, optional): list of valid masks. Default: None
        valid_ratio (float): volume ratio threshold for valid samples. Default: 0.5
        sample_volume_size (tuple, int): model input size.
        sample_label_size (tuple, int): model output size.
        sample_stride (tuple, int): stride size for sampling.
        augmentor (connectomics.data.augmentation.composition.Compose, optional): data augmentor for training. Default: None
        target_opt (list): list the model targets generated from segmentation labels.
        weight_opt (list): list of options for generating pixel-wise weight masks.
        mode (str): ``'train'``, ``'val'`` or ``'test'``. Default: ``'train'``
        do_2d (bool): load 2d samples from 3d volumes. Default: False
        iter_num (int): total number of training iterations (-1 for inference). Default: -1
        do_relabel (bool): reduce the the mask indicies in a sampled label volume. This option be set to
            False for semantic segmentation, otherwise the classes can shift. Default: True
        reject_size_thres (int, optional): threshold to decide if a sampled volumes contains foreground objects. Default: 0
        reject_diversity (int, optional): threshold to decide if a sampled volumes contains multiple objects. Default: 0
        reject_p (float, optional): probability of rejecting non-foreground volumes. Default: 0.95
        data_mean (float): mean of pixels for images normalized to (0,1). Default: 0.5
        data_std (float): standard deviation of pixels for images normalized to (0,1). Default: 0.5
        data_match_act (str): the data is normalized to match the range of an activation. Default: ``'none'``

    Note:
        For relatively small volumes, the total number of possible subvolumes can be smaller than the total number
        of samples required in training (the product of total iterations and mini-natch size), which raises *StopIteration*.
        Therefore the dataset length is also decided by the training settings.
    """

    background: int = 0  # background label index

    def __init__(self,
                 volume: list,
                 label: Optional[list] = None,
                 valid_mask: Optional[list] = None,
                 valid_ratio: float = 0.5,
                 sample_volume_size: tuple = (8, 64, 64),
                 sample_label_size: tuple = (8, 64, 64),
                 sample_stride: tuple = (1, 1, 1),
                 augmentor: AUGMENTOR_TYPE = None,
                 target_opt: TARGET_OPT_TYPE = ['1'],
                 weight_opt: WEIGHT_OPT_TYPE = [['1']],
                 erosion_rates: Optional[List[int]] = None,
                 dilation_rates: Optional[List[int]] = None,
                 mode: str = 'train',
                 do_2d: bool = False,
                 iter_num: int = -1,
                 do_relabel: bool = True,
                 # rejection sampling
                 reject_size_thres: int = 0,
                 reject_diversity: int = 0,
                 reject_p: float = 0.95,
                 # normalization
                 data_mean: float = 0.5,
                 data_std: float = 0.5,
                 data_match_act: str = 'none'):

        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.do_2d = do_2d
        self.do_relabel = do_relabel

        # data format
        self.volume = volume
        self.label = label
        self.augmentor = augmentor

        # target and weight options
        self.target_opt = target_opt
        self.weight_opt = weight_opt
        # For 'all', users will create their own targets
        if self.target_opt[-1] == 'all':
            self.target_opt = self.target_opt[:-1]
            self.weight_opt = self.weight_opt[:-1]
        self.erosion_rates = erosion_rates
        self.dilation_rates = dilation_rates

        # rejection samping
        self.reject_size_thres = reject_size_thres
        self.reject_diversity = reject_diversity
        self.reject_p = reject_p

        # normalization
        self.data_mean = data_mean
        self.data_std = data_std
        self.data_match_act = data_match_act

        # dataset: channels, depths, rows, cols
        # volume size, could be multi-volume input
        self.volume_size = [np.array(x.shape) for x in self.volume]
        self.sample_volume_size = np.array(
            sample_volume_size).astype(int)  # model input size
        if self.label is not None:
            self.sample_label_size = np.array(
                sample_label_size).astype(int)  # model label size
            self.label_vol_ratio = self.sample_label_size / self.sample_volume_size
            if self.augmentor is not None:
                assert np.array_equal(
                    self.augmentor.sample_size, self.sample_label_size)
        self._assert_valid_shape()

        # compute number of samples for each dataset (multi-volume input)
        self.sample_stride = np.array(sample_stride).astype(int)
        self.sample_size = [count_volume(self.volume_size[x], self.sample_volume_size, self.sample_stride)
                            for x in range(len(self.volume_size))]

        # total number of possible inputs for each volume
        self.sample_num = np.array([np.prod(x) for x in self.sample_size])
        self.sample_num_a = np.sum(self.sample_num)
        self.sample_num_c = np.cumsum([0] + list(self.sample_num))

        # handle partially labeled volume
        self.valid_mask = valid_mask
        self.valid_ratio = valid_ratio

        if self.mode in ['val', 'test']:  # for validation and test
            self.sample_size_test = [
                np.array([np.prod(x[1:3]), x[2]]) for x in self.sample_size]

        # For relatively small volumes, the total number of samples can be generated is smaller
        # than the number of samples required for training (i.e., iteration * batch size). Thus
        # we let the __len__() of the dataset return the larger value among the two during training.
        self.iter_num = max(
            iter_num, self.sample_num_a) if self.mode == 'train' else self.sample_num_a
        print('Total number of samples to be generated: ', self.iter_num)

    def __len__(self):
        # total number of possible samples
        return self.iter_num

    def __getitem__(self, index):
        # orig input: keep uint/int format to save cpu memory
        # output sample: need np.float32

        vol_size = self.sample_volume_size
        if self.mode == 'train':
            sample = self._rejection_sampling(vol_size)
            return self._process_targets(sample)

        elif self.mode == 'val':
            pos = self._get_pos_test(index)
            sample = self._crop_with_pos(pos, vol_size)
            return self._process_targets(sample)

        elif self.mode == 'test':
            pos = self._get_pos_test(index)
            out_volume = (crop_volume(
                self.volume[pos[0]], vol_size, pos[1:])/255.0).astype(np.float32)
            if self.do_2d:
                out_volume = np.squeeze(out_volume)

            return pos, self._process_image(out_volume)

    def _process_targets(self, sample):
        pos, out_volume, out_label, out_valid = sample

        if self.do_2d:
            out_volume, out_label, out_valid = numpy_squeeze(
                out_volume, out_label, out_valid)

        out_volume = self._process_image(out_volume)
        if out_label is None: # unlabeled data, compatible with collate_fn_test
            return pos, out_volume

        # convert masks to different learning targets
        out_target = seg_to_targets(
            out_label, self.target_opt, self.erosion_rates, self.dilation_rates)
        out_weight = seg_to_weights(
            out_target, self.weight_opt, out_valid, out_label)
        return pos, out_volume, out_target, out_weight

    #######################################################
    # Position Calculator
    #######################################################

    def _index_to_dataset(self, index):
        return np.argmax(index < self.sample_num_c) - 1  # which dataset

    def _index_to_location(self, index, sz):
        # index -> z,y,x
        # sz: [y*x, x]
        pos = [0, 0, 0]
        pos[0] = np.floor(index/sz[0])
        pz_r = index % sz[0]
        pos[1] = int(np.floor(pz_r/sz[1]))
        pos[2] = pz_r % sz[1]
        return pos

    def _get_pos_test(self, index):
        pos = [0, 0, 0, 0]
        did = self._index_to_dataset(index)
        pos[0] = did
        index2 = index - self.sample_num_c[did]
        pos[1:] = self._index_to_location(index2, self.sample_size_test[did])
        # if out-of-bound, tuck in
        for i in range(1, 4):
            if pos[i] != self.sample_size[pos[0]][i-1]-1:
                pos[i] = int(pos[i] * self.sample_stride[i-1])
            else:
                pos[i] = int(self.volume_size[pos[0]][i-1] -
                             self.sample_volume_size[i-1])
        return pos

    def _get_pos_train(self, vol_size):
        # random: multithread
        # np.random: same seed
        pos = [0, 0, 0, 0]
        # pick a dataset
        did = self._index_to_dataset(random.randint(0, self.sample_num_a-1))
        pos[0] = did
        # pick a position
        tmp_size = count_volume(
            self.volume_size[did], vol_size, self.sample_stride)
        tmp_pos = [random.randint(0, tmp_size[x]-1) * self.sample_stride[x]
                   for x in range(len(tmp_size))]

        pos[1:] = tmp_pos
        return pos

    #######################################################
    # Volume Sampler
    #######################################################
    def _rejection_sampling(self, vol_size):
        """Rejection sampling to filter out samples without required number
        of foreground pixels or valid ratio.
        """
        sample_count = 0
        while True:
            sample = self._random_sampling(vol_size)
            pos, out_volume, out_label, out_valid = sample
            if self.augmentor is not None:

                if out_valid is not None:
                    assert 'valid_mask' in self.augmentor.additional_targets.keys(), \
                        "Need to specify the 'valid_mask' option in additional_targets " \
                        "of the data augmentor when training with partial annotation."

                data = {'image': out_volume,
                        'label': out_label,
                        'valid_mask': out_valid}

                augmented = self.augmentor(data)
                out_volume, out_label = augmented['image'], augmented['label']
                out_valid = augmented['valid_mask']

            if self._is_valid(out_valid) and self._is_fg(out_label):
                return pos, out_volume, out_label, out_valid

            sample_count += 1
            if sample_count > 100:
                err_msg = (
                    "Can not find any valid subvolume after sampling the "
                    "dataset for more than 100 times. Please adjust the "
                    "valid mask or rejection sampling configurations."
                )
                raise RuntimeError(err_msg)

    def _random_sampling(self, vol_size):
        """Randomly sample a subvolume from all the volumes.
        """
        pos = self._get_pos_train(vol_size)
        return self._crop_with_pos(pos, vol_size)

    def _crop_with_pos(self, pos, vol_size):
        out_volume = (crop_volume(
            self.volume[pos[0]], vol_size, pos[1:])/255.0).astype(np.float32)

        # position in the label and valid mask
        out_label, out_valid = None, None
        if self.label is not None:
            pos_l = np.round(pos[1:]*self.label_vol_ratio)
            out_label = crop_volume(self.label[pos[0]], self.sample_label_size, pos_l)
            # For warping: cv2.remap requires input to be float32.
            # Make labels index smaller. Otherwise uint32 and float32 are not
            # the same for some values.
            out_label = reduce_label(out_label.copy()) if self.do_relabel else out_label.copy()
            out_label = out_label.astype(np.float32)

        if self.valid_mask is not None:
            out_valid = crop_volume(self.valid_mask[pos[0]], self.sample_label_size, pos_l)
            out_valid = (out_valid != 0).astype(np.float32)

        return pos, out_volume, out_label, out_valid

    def _is_valid(self, out_valid: np.ndarray) -> bool:
        """Decide whether the sampled region is valid or not using
        the corresponding valid mask.
        """
        if self.valid_mask is None or out_valid is None:
            return True
        ratio = float(out_valid.sum()) / np.prod(np.array(out_valid.shape))
        return ratio > self.valid_ratio

    def _is_fg(self, out_label: np.ndarray) -> bool:
        """Decide whether the sample belongs to a foreground decided
        by the rejection sampling criterion.
        """
        if self.label is None or out_label is None:
            return True

        p = self.reject_p
        size_thres = self.reject_size_thres
        if size_thres > 0:
            temp = out_label.copy().astype(int)
            temp = (temp != self.background).astype(int).sum()
            if temp < size_thres and random.random() < p:
                return False

        num_thres = self.reject_diversity
        if num_thres > 0:
            temp = out_label.copy().astype(int)
            num_objects = len(np.unique(temp))
            if num_objects < num_thres and random.random() < p:
                return False

        return True

    #######################################################
    # Utils
    #######################################################
    def _process_image(self, x: np.array):
        x = np.expand_dims(x, 0) # (z,y,x) -> (c,z,y,x)
        x = normalize_image(x, self.data_mean, self.data_std,
                            match_act=self.data_match_act)
        return x

    def _assert_valid_shape(self):
        assert all(
            [(self.sample_volume_size <= x).all()
             for x in self.volume_size]
        ), "Input size should be smaller than volume size."

        if self.label is not None:
            assert all(
                [(self.sample_label_size <= x).all()
                 for x in self.volume_size]
            ), "Label size should be smaller than volume size."


class VolumeDatasetRecon(VolumeDataset):
    def _rejection_sampling(self, vol_size):
        while True:
            sample = self._random_sampling(vol_size)
            pos, out_volume, out_label, out_valid = sample
            if self.augmentor is not None:
                if out_valid is not None:
                    assert 'valid_mask' in target_dict.keys(), \
                        "Need to specify the 'valid_mask' option in additional_targets " \
                        "of the data augmentor when training with partial annotation."

                target_dict = self.augmentor.additional_targets
                assert 'recon_image' in target_dict.keys() and target_dict['recon_image'] == 'img'
                data = {'image': out_volume,
                        'label': out_label,
                        'valid_mask': out_valid,
                        'recon_image': out_volume.copy()}

                augmented = self.augmentor(data)
                out_volume, out_label = augmented['image'], augmented['label']
                out_valid = augmented['valid_mask']
                out_recon = augmented['recon_image']

            else: # no augmentation, out_recon is out_volume
                out_recon = out_volume.copy()

            if self._is_valid(out_valid) and self._is_fg(out_label):
                return pos, out_volume, out_label, out_valid, out_recon

    def _process_targets(self, sample):
        pos, out_volume, out_label, out_valid, out_recon = sample

        if self.do_2d:
            out_volume, out_label, out_valid, out_recon = numpy_squeeze(
                out_volume, out_label, out_valid, out_recon)

        out_volume = self._process_image(out_volume)
        out_recon = self._process_image(out_recon)

        # output list
        if out_label is None: # unlabeled data
            return pos, out_volume, out_recon

        # convert masks to different learning targets
        out_target = seg_to_targets(
            out_label, self.target_opt, self.erosion_rates, self.dilation_rates)
        out_weight = seg_to_weights(
            out_target, self.weight_opt, out_valid, out_label)
        return pos, out_volume, out_target, out_weight, out_recon


class VolumeDatasetMultiSeg(VolumeDataset):
    def __init__(self, multiseg_split: list = [1,2], **kwargs):
        self.multiseg_split = multiseg_split
        super().__init__(**kwargs)

        if self.mode == 'test': 
            return # no need for target_opt and multiseg_split in inference

        assert len(self.target_opt) == sum(multiseg_split)
        self.multiseg_cumsum = [0] + list(np.cumsum(multiseg_split))
        self.target_opt_multiseg = []
        for i in range(len(multiseg_split)):
            self.target_opt_multiseg.append(
                self.target_opt[self.multiseg_cumsum[i]:self.multiseg_cumsum[i+1]])
        print("Multiseg target options: ", self.target_opt_multiseg)

    def _rejection_sampling(self, vol_size):
        while True:
            sample = self._random_sampling(vol_size)
            pos, out_volume, out_label, out_valid = sample
            n_seg_maps =  out_label.shape[0]

            if self.augmentor is not None:
                if out_valid is not None:
                    assert 'valid_mask' in target_dict.keys(), \
                        "Need to specify the 'valid_mask' option in additional_targets " \
                        "of the data augmentor when training with partial annotation."

                data = {'image': out_volume}
                target_dict = self.augmentor.additional_targets
                for i in range(n_seg_maps):
                    assert 'label%d' % i in target_dict.keys() # each channel is augmented
                    data['label%d' % i] = out_label[i,:,:,:]

                augmented = self.augmentor(data)
                out_volume = augmented['image']
                out_label = [augmented['label%d' % i] for i in range(n_seg_maps)]

            else: # no augmentation, out_recon is out_volume
                out_label = [out_label[i,:,:,:] for i in range(n_seg_maps)]

            # the first segmentation map is used for rejection sampling
            if self._is_valid(out_valid) and self._is_fg(out_label[0]):
                return pos, out_volume, out_label, out_valid

    def _process_targets(self, sample):
        pos, out_volume, out_label, out_valid = sample

        if self.do_2d:
            out_volume, out_valid = numpy_squeeze(out_volume, out_valid)
            out_label = numpy_squeeze(*out_label)

        out_volume = self._process_image(out_volume)

        if out_label is None: # unlabeled data
            return pos, out_volume, None, None

        # convert masks to different learning targets
        out_target = self.multiseg_to_targets(out_label)
        out_weight = seg_to_weights(
            out_target, self.weight_opt, out_valid, out_label)
        return pos, out_volume, out_target, out_weight

    def multiseg_to_targets(self, out_label):
        assert len(out_label) == len(self.target_opt_multiseg)
        out_target = []
        for i in range(len(out_label)):
            out_target.extend(seg_to_targets(
                out_label[i], self.target_opt_multiseg[i], 
                self.erosion_rates, self.dilation_rates))
        
        return out_target
