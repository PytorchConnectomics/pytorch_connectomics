import torch
import torchvision.utils as vutils
import numpy as np
import pudb

class Visualizer(object):
    def __init__(self, vis_opt=0, N=8):
        self.vis_opt = vis_opt
        self.N = N # default maximum number of sections to show
        self.N_ind = None

    def _prepare_data(self, volume, label, output):
        if self.N_ind is None and volume.size()[0]*volume.size()[2] > self.N:
            self.N_ind = (np.linspace(0,1,self.N)*(volume.size()[0]*volume.size()[2]-1)).astype(int) 
        if len(volume.size()) == 4:   # 2D Inputs
            if self.N_ind is not None:
                return volume[self.N_ind], label[self.N_ind], output[self.N_ind]
            else:
                return volume, label, output
        elif len(volume.size()) == 5: # 3D Inputs
            # first instance
            # volume, label, output = volume[0].permute(1,0,2,3), label[0].permute(1,0,2,3), output[0].permute(1,0,2,3)
            # stack instance
            # bczyx -> (bz)cyx
            volume, label, output = volume.permute(0,2,1,3,4), label.permute(0,2,1,3,4), output.permute(0,2,1,3,4)
            volume, label, output = volume.reshape([-1]+list(volume.shape[2:])), label.reshape([-1]+list(label.shape[2:])), output.reshape([-1]+list(output.shape[2:]))
            if self.N_ind is not None:
                return volume[self.N_ind], label[self.N_ind], output[self.N_ind]
            else:
                return volume, label, output

    def visualize(self, volume, label, output, iter_total, writer):
        if self.vis_opt == 0:
            self.visualize_combine(volume, label, output, iter_total, writer)
        elif self.vis_opt == 1:
            self.visualize_individual(volume, label, output, iter_total, writer)
        elif self.vis_opt == 2:
            self.visualize_individual(volume, label, output, iter_total, writer, composite=True)

    def visualize_individual(self, volume, label, output, iteration, writer, composite=False):
        volume, label, output = self._prepare_data(volume, label, output)

        sz0 = volume.size() # z,c,y,x
        volume_visual = volume.detach().cpu().expand(sz0[0],3,sz0[2],sz0[3])
        sz = output.size() # z,c,y,x
        output_visual = output.detach().cpu().expand(sz[0],3,sz[2],sz[3])
        label_visual = label.detach().cpu().expand(sz[0],3,sz[2],sz[3])

        volume_show = vutils.make_grid(volume_visual, nrow=8, normalize=True, scale_each=True)
        output_show = vutils.make_grid(output_visual, nrow=8, normalize=True, scale_each=True)
        label_show = vutils.make_grid(label_visual, nrow=8, normalize=True, scale_each=True)

        writer.add_image('Input', volume_show, iteration)
        writer.add_image('Label', label_show, iteration)
        writer.add_image('Output', output_show, iteration)

        if composite:
            composite_1 = torch.max(volume_show, label_show) 
            composite_2 = torch.max(volume_show, output_show)
            writer.add_image('Composite_GT', composite_1, iteration)
            writer.add_image('Composite_PD', composite_2, iteration)

    def visualize_combine(self, volume, label, output, iteration, writer):
        volume, label, output = self._prepare_data(volume, label, output)
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

        writer.add_image('Combine', canvas_show, iteration)
