import torch
import torchvision.utils as vutils
import numpy as np

class Visualizer(object):
    def __init__(self, vis_opt=0, N=8):
        self.vis_opt = vis_opt
        self.N = N # default maximum number of sections to show
        self.N_ind = None

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

    def visualize(self, volume, label, output, iter_total, writer):
        if self.vis_opt == 0:
            self.visualize_consecutive(volume, label, output, iter_total, writer)
        elif self.vis_opt == 1:
            self.visualize_individual(volume, label, output, iter_total, writer)

    def visualize_individual(self, volume, label, output, iteration, writer):
        volume, label, output = self.prepare_data(volume, label, output)

        sz = volume.size() # z,c,y,x
        volume_visual = volume.detach().cpu().expand(sz[0],3,sz[2],sz[3])
        output_visual = output.detach().cpu().expand(sz[0],3,sz[2],sz[3])
        label_visual = label.detach().cpu().expand(sz[0],3,sz[2],sz[3])

        volume_show = vutils.make_grid(volume_visual, nrow=8, normalize=True, scale_each=True)
        output_show = vutils.make_grid(output_visual, nrow=8, normalize=True, scale_each=True)
        label_show = vutils.make_grid(label_visual, nrow=8, normalize=True, scale_each=True)

        writer.add_image('Input', volume_show, iteration)
        writer.add_image('Label', label_show, iteration)
        writer.add_image('Output', output_show, iteration)

    def visualize_consecutive(self, volume, label, output, iteration, writer):
        volume, label, output = self.prepare_data(volume, label, output)
        sz = volume.size() # z,c,y,x
        canvas = []
        volume_visual = volume.detach().cpu().expand(sz[0],3,sz[2],sz[3])
        canvas.append(volume_visual)

        sz = output.size() # z,c,y,x
        output_visual = [output[:,i].detach().cpu().unsqueeze(1).expand(sz[0],3,sz[2],sz[3]) for i in range(sz[1])]

        sz = label.size() # z,c,y,x
        label_visual = [label[:,i].detach().cpu().unsqueeze(1).expand(sz[0],3,sz[2],sz[3]) for i in range(sz[1])]
        canvas = canvas + output_visual
        canvas = canvas + label_visual
        canvas_merge = torch.cat(canvas, 0)
        canvas_show = vutils.make_grid(canvas_merge, nrow=8, normalize=True, scale_each=True)

        writer.add_image('Consecutive', canvas_show, iteration)