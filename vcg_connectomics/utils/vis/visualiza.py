import torch
import torchvision.utils as vutils

def visualize(volume, label, output, iteration, writer):

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

def visualize_aff(volume, label, output, iteration, writer):

    sz = volume.size() # z,c,y,x
    canvas = []
    volume_visual = volume.detach().cpu().expand(sz[0],3,sz[2],sz[3])
    canvas.append(volume_visual)
    output_visual = [output[:,i].detach().cpu().unsqueeze(1).expand(sz[0],3,sz[2],sz[3]) for i in range(3)]
    label_visual = [label[:,i].detach().cpu().unsqueeze(1).expand(sz[0],3,sz[2],sz[3]) for i in range(3)]
    canvas = canvas + output_visual
    canvas = canvas + label_visual
    canvas_merge = torch.cat(canvas, 0)
    canvas_show = vutils.make_grid(canvas_merge, nrow=8, normalize=True, scale_each=True)

    writer.add_image('Affinity', canvas_show, iteration)