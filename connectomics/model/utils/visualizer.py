import torch
import torchvision.utils as vutils
import numpy as np

class Visualizer(object):
    def __init__(self, vis_opt=0, N=16):
        self.vis_opt = vis_opt
        self.N = N # default maximum number of sections to show
        self.N_ind = None
        # number of channels of different target options
        self.num_channels_dict = {
            '0': 1,
            '1': 3,
            '2': 3,  
            '3': 1,
            '4': 1,
        }

    def prepare_data(self, volume, label, output):
        if len(volume.size()) == 4:   # 2D Inputs
            if volume.size()[0] > self.N:
                return volume[:self.N], label[:self.N], output[:self.N]
            else:
                return volume, label, output
        elif len(volume.size()) == 5: # 3D Inputs
            volume, label, output = volume[0].permute(1,0,2,3), label[0].permute(1,0,2,3), output[0].permute(1,0,2,3)
            if volume.size()[0] > self.N:
                return volume[:self.N], label[:self.N], output[:self.N]
            else:
                return volume, label, output

    def visualize(self, cfg, volume, label, output, iter_total, writer):
        # split the prediction into chunks along the channel dimension
        split_channels = [self.num_channels_dict[x.split('-')[0]] for x in cfg.MODEL.TARGET_OPT]
        output = torch.split(output, split_channels, dim=1)
        assert len(output) == len(label)

        for idx in range(len(cfg.MODEL.TARGET_OPT)):
            RGB = (cfg.MODEL.TARGET_OPT[idx] == '1')
            vis_name = cfg.MODEL.TARGET_OPT[idx] + '_' + str(idx)
            self.visualize_consecutive(volume, torch.from_numpy(label[idx]), output[idx], iter_total, writer, RGB, vis_name)

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

    # def visualize_composite(self, volume, label, output, iteration, writer, RGB=False, vis_name='0_0'):
    #     volume, label, output = self.prepare_data(volume, label, output)
    #     sz = volume.size() # z,c,y,x
    #     canvas = []
    #     volume_visual = volume.detach().cpu().expand(sz[0],3,sz[2],sz[3])
    #     canvas.append(volume_visual)

    #     if RGB:
    #         output_visual = output.detach().cpu()
    #         label_visual = label.detach().cpu()

    #         label_visual = [torch.max(volume_visual, label_visual)]
    #         output_visual = [torch.max(volume_visual, output_visual)]
    #     else:
    #         output_visual = [output[:,i].detach().cpu().unsqueeze(1).expand(sz[0],3,sz[2],sz[3]) for i in range(sz[1])]
    #         label_visual = [label[:,i].detach().cpu().unsqueeze(1).expand(sz[0],3,sz[2],sz[3]) for i in range(sz[1])]

    #     canvas = canvas + label_visual
    #     canvas = canvas + output_visual

    #     canvas_merge = torch.cat(canvas, 0)
    #     canvas_show = vutils.make_grid(canvas_merge, nrow=8, normalize=True, scale_each=True)

    #     writer.add_image('Composite_%s' % vis_name, canvas_show, iteration)