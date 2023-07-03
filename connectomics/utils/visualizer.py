from __future__ import print_function, division
from typing import Optional, List, Union, Tuple

import torch
import torchvision.utils as vutils
import numpy as np
from ..data.utils import decode_quantize, dx_to_circ
from connectomics.model.utils import SplitActivation

__all__ = [
    'Visualizer'
]


class Visualizer(object):
    """TensorboardX visualizer for displaying loss, learning rate and predictions
    at training time.
    """

    def __init__(self, cfg, vis_opt=0, N=16):
        self.cfg = cfg
        self.act = SplitActivation.build_from_cfg(cfg, do_cat=False)
        self.vis_opt = vis_opt
        self.N = N  # default maximum number of sections to show
        self.N_ind = None

        self.semantic_colors = {}
        for topt in self.cfg.MODEL.TARGET_OPT:
            if topt[0] == '9':
                channels = int(topt.split('-')[1])
                colors = [torch.rand(3) for _ in range(channels)]
                colors[0] = torch.zeros(3)  # make background black
                self.semantic_colors[topt] = torch.stack(colors, dim=0)
                
            if topt[0] == '1' and len(topt) != 1: # synaptic cleft (exclusive)
                _, exclusive = topt.split('-')
                assert int(exclusive), f"Option {topt} is not expected!"    
                colors = torch.stack([
                    torch.tensor([0.0, 0.0, 0.0]),
                    torch.tensor([1.0, 0.0, 1.0]),
                    torch.tensor([0.0, 1.0, 1.0])], dim=0)      
                self.semantic_colors[topt] = colors

    def visualize(self, volume, label, output, weight, iter_total, writer,
                  suffix: Optional[str] = None, additional_image_groups: Optional[dict] = None):
        self.visualize_image_groups(writer, iter_total, additional_image_groups)
        
        volume = self._denormalize(volume)
        # split the prediction into chunks along the channel dimension
        output = self.act(output)
        assert len(output) == len(label)

        for idx in range(len(self.cfg.MODEL.TARGET_OPT)):
            topt = self.cfg.MODEL.TARGET_OPT[idx]
            if topt[0] == '9': # semantic segmentation
                output[idx] = self.get_semantic_map(output[idx], topt)
                label[idx] = self.get_semantic_map(label[idx], topt, argmax=False)

            if topt[0] == '1' and len(topt) != 1: # synaptic cleft segmentation
                _, exclusive = topt.split('-')
                assert int(exclusive), f"Option {topt} is not expected!"
                output[idx] = self.get_semantic_map(output[idx], topt)
                label[idx] = self.get_semantic_map(label[idx], topt, argmax=False)

            if topt[0] == '5': # distance transform
                if len(topt) == 1:
                    topt = topt + '-2d-0-0-5.0' # default
                _, mode, padding, quant, z_res = topt.split('-')
                if bool(int(quant)): # only the quantized version needs decoding
                    output[idx] = decode_quantize(
                        output[idx], mode='max').unsqueeze(1)
                    temp_label = label[idx].clone().float()[
                        :, np.newaxis]
                    label[idx] = temp_label / temp_label.max() + 1e-6

            if topt[0]=='7': # diffusion gradient
                output[idx] = dx_to_circ(output[idx])
                label[idx] = dx_to_circ(label[idx])

            RGB = (topt[0] in ['1', '2', '7', '9'])
            vis_name = self.cfg.MODEL.TARGET_OPT[idx] + '_' + str(idx)
            if suffix is not None:
                vis_name = vis_name + '_' + suffix
            if isinstance(label[idx], (np.ndarray, np.generic)):
                label[idx] = torch.from_numpy(label[idx])

            weight_maps = {}
            for j, wopt in enumerate(self.cfg.MODEL.WEIGHT_OPT[idx]):
                if wopt != '0':
                    w_name = vis_name + '_' + wopt
                    weight_maps[w_name] = weight[idx][j]
                else:  # The weight map can be the binary valid mask.
                    if weight[idx][j].shape[-1] != 1:
                        weight_maps['valid_mask'] = weight[idx][j]

            self.visualize_consecutive(volume, label[idx], output[idx], weight_maps,
                                       iter_total, writer, RGB=RGB, vis_name=vis_name)

    def visualize_image_groups(self, writer, iteration, image_groups: Optional[dict] = None,
                               is_3d: bool = True) -> None:
        if image_groups is None:
            return

        for name in image_groups.keys():
            image_list = image_groups[name]
            image_list = [self._denormalize(x) for x in image_list]
            image_list = [self.permute_truncate(x, is_3d=is_3d) for x in image_list]
            sz = image_list[0].size()
            canvas = [x.detach().cpu().expand(sz[0], 3, sz[2], sz[3]) for x in image_list]
            canvas_merge = torch.cat(canvas, 0)
            canvas_show = vutils.make_grid(
                canvas_merge, nrow=8, normalize=True, scale_each=True)

            writer.add_image('Image_Group_%s' % name, canvas_show, iteration)

    def visualize_consecutive(self, volume, label, output, weight_maps, iteration,
                              writer, RGB=False, vis_name='0_0'):
        volume, label, output, weight_maps = self.prepare_data(
            volume, label, output, weight_maps)
        sz = volume.size()  # z,c,y,x
        canvas = []
        volume_visual = volume.detach().cpu().expand(sz[0], 3, sz[2], sz[3])
        canvas.append(volume_visual)

        def maybe2rgb(temp):
            if temp.shape[1] == 2: # 2d affinity map has two channels
                temp = torch.cat([temp, torch.zeros(
                    sz[0], 1, sz[2], sz[3]).type(temp.dtype)], dim=1)
            return temp

        if RGB:
            output_visual = [maybe2rgb(output.detach().cpu())]
            label_visual = [maybe2rgb(label.detach().cpu())]
        else:
            output_visual = [self.vol_reshape(
                output[:, i], sz) for i in range(sz[1])]
            label_visual = [self.vol_reshape(
                label[:, i], sz) for i in range(sz[1])]

        weight_visual = []
        for key in weight_maps.keys():
            weight_visual.append(maybe2rgb(weight_maps[key]).detach().cpu().expand(
                                 sz[0], 3, sz[2], sz[3]))

        canvas = canvas + output_visual + label_visual + weight_visual
        canvas_merge = torch.cat(canvas, 0)
        canvas_show = vutils.make_grid(
            canvas_merge, nrow=8, normalize=True, scale_each=True)

        writer.add_image('Consecutive_%s' % vis_name, canvas_show, iteration)

    def prepare_data(self, volume, label, output, weight_maps):
        ndim = volume.ndim
        assert ndim in [4, 5]
        is_3d = (ndim == 5)

        volume = self.permute_truncate(volume, is_3d)
        label = self.permute_truncate(label, is_3d)
        output = self.permute_truncate(output, is_3d)
        for key in weight_maps.keys():
            weight_maps[key] = self.permute_truncate(weight_maps[key], is_3d)

        return volume, label, output, weight_maps

    def permute_truncate(self, data, is_3d=False):
        if is_3d: # show consecutive slices of a 3d volume
            data = data[0].permute(1, 0, 2, 3)
        high = min(data.size()[0], self.N)
        return data[:high]

    def get_semantic_map(self, output, topt, argmax=True):
        if isinstance(output, (np.ndarray, np.generic)):
            output = torch.from_numpy(output)
        # output shape: BCDHW or BCHW
        if argmax:
            output = torch.argmax(output, 1)
        pred = self.semantic_colors[topt][output.cpu()]
        if len(pred.size()) == 4:   # 2D Inputs
            pred = pred.permute(0, 3, 1, 2)
        elif len(pred.size()) == 5:  # 3D Inputs
            pred = pred.permute(0, 4, 1, 2, 3)

        return pred

    def vol_reshape(self, vol, sz):
        vol = vol.detach().cpu().unsqueeze(1)
        return vol.expand(sz[0], 3, sz[2], sz[3])

    def _denormalize(self, volume):
        match_act = self.cfg.DATASET.MATCH_ACT
        if match_act == 'none':
            volume = (volume * self.cfg.DATASET.STD) + self.cfg.DATASET.MEAN
        elif match_act == 'tanh':
            volume = (volume + 1.0) * 0.5
        return volume
