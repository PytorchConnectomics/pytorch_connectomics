import torch
import torchvision.utils as vutils
import numpy as np
from ..data.utils import decode_quantize
from ..model.utils import get_functional_act

__all__ = [
    'Visualizer'
]

class Visualizer(object):
    """TensorboardX visualizer for displaying loss, learning rate and predictions
    at training time.
    """
    # number of channels of different target options
    num_channels_dict = {
        '0': 1,
        '1': 3,
        '2': 3,  
        '3': 1,
        '4': 1,
        '5': 11,
    }
    def __init__(self, cfg, vis_opt=0, N=16, do_2d=False):
        self.cfg = cfg
        self.vis_opt = vis_opt
        self.N = N # default maximum number of sections to show
        self.N_ind = None
        if do_2d:
            self.num_channels_dict['2'] = 2

        self.split_channels = []
        self.semantic_colors = {}
        for topt in self.cfg.MODEL.TARGET_OPT:
            if topt[0] == '9':
                channels = int(topt.split('-')[1])
                self.split_channels.append(channels)
                colors = [torch.rand(3) for _ in range(channels)]
                colors[0] = torch.zeros(3) # make background black
                self.semantic_colors[topt] = torch.stack(colors, 0)
            else:
                self.split_channels.append(
                    self.num_channels_dict[topt[0]])

        self.act = self.get_act(cfg.INFERENCE.OUTPUT_ACT)

    def prepare_data(self, volume, label, output):
        if len(volume.size()) == 4:   # 2D Inputs
            if volume.size()[0] > self.N:
                return volume[:self.N], label[:self.N], output[:self.N]
            return volume, label, output

        elif len(volume.size()) == 5: # 3D Inputs
            volume = volume[0].permute(1,0,2,3)
            label, output = label[0].permute(1,0,2,3), output[0].permute(1,0,2,3)
            if volume.size()[0] > self.N:
                return volume[:self.N], label[:self.N], output[:self.N]
            return volume, label, output

    def visualize(self, volume, label, output, iter_total, writer):
        # split the prediction into chunks along the channel dimension
        output = torch.split(output, self.split_channels, dim=1)
        output = list(output) # torch.split returns a tuple
        output = [self.act[i](output[i]) for i in range(len(output))]
        assert len(output) == len(label)

        for idx in range(len(self.cfg.MODEL.TARGET_OPT)):
            topt = self.cfg.MODEL.TARGET_OPT[idx]
            if topt[0] == '9':
                output[idx] = self.get_semantic_map(output[idx], topt)
                label[idx] = self.get_semantic_map(label[idx], topt, argmax=False)
            if topt[0] == '5':
                output[idx] = decode_quantize(output[idx], mode='max').unsqueeze(1)
                temp_label = label[idx].copy().astype(np.float32)[:, np.newaxis]
                label[idx] = temp_label / temp_label.max() + 1e-6

            RGB = (topt[0] in ['1', '2', '9'])
            vis_name = self.cfg.MODEL.TARGET_OPT[idx] + '_' + str(idx)
            if isinstance(label[idx], (np.ndarray, np.generic)):
                label[idx] = torch.from_numpy(label[idx])
            self.visualize_consecutive(volume, label[idx], output[idx], iter_total, writer, RGB, vis_name)

    def visualize_consecutive(self, volume, label, output, iteration, writer, RGB=False, vis_name='0_0'):
        volume, label, output = self.prepare_data(volume, label, output)
        sz = volume.size() # z,c,y,x
        canvas = []
        volume_visual = volume.detach().cpu().expand(sz[0],3,sz[2],sz[3])
        canvas.append(volume_visual)

        if RGB:
            output_visual = [output.detach().cpu()]
            label_visual = [label.detach().cpu()]
        else:
            output_visual = [output[:,i].detach().cpu().unsqueeze(1).expand(sz[0],3,sz[2],sz[3]) for i in range(sz[1])]
            label_visual = [label[:,i].detach().cpu().unsqueeze(1).expand(sz[0],3,sz[2],sz[3]) for i in range(sz[1])]

        canvas = canvas + output_visual
        canvas = canvas + label_visual
        canvas_merge = torch.cat(canvas, 0)
        canvas_show = vutils.make_grid(canvas_merge, nrow=8, normalize=True, scale_each=True)

        writer.add_image('Consecutive_%s' % vis_name, canvas_show, iteration)

    def get_semantic_map(self, output, topt, argmax=True):
        if isinstance(output, (np.ndarray, np.generic)):
            output = torch.from_numpy(output)
        # output shape: BCDHW or BCHW
        if argmax:
            output = torch.argmax(output, 1)
        pred = self.semantic_colors[topt][output] 
        if len(pred.size()) == 4:   # 2D Inputs
            pred = pred.permute(0,3,1,2)
        elif len(pred.size()) == 5: # 3D Inputs
            pred = pred.permute(0,4,1,2,3)
        
        return pred

    def get_act(self, output_act):
        num_target = len(output_act)
        out = [None]*num_target
        for i, act in enumerate(output_act):
            out[i] = get_functional_act(act)
        return out
