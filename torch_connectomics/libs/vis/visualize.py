import torch
import torchvision.utils as vutils

N = 8 # default maximum number of sections to show

def prepare_data(volume, label, output):
    if len(volume.size()) == 4:   # 2D Inputs
        if volume.size()[0] > N:
            return volume[:N], label[:N], output[:N]
        else:
            return volume, label, output
    elif len(volume.size()) == 5: # 3D Inputs
        volume, label, output = volume[0].permute(1,0,2,3), label[0].permute(1,0,2,3), output[0].permute(1,0,2,3)
        if volume.size()[0] > N:
            return volume[:N], label[:N], output[:N]
        else:
            return volume, label, output

def visualize(volume, label, output, iteration, writer, composite=False):
    volume, label, output = prepare_data(volume, label, output)

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

    if composite:
        composite_1 = torch.max(volume_show, label_show) 
        composite_2 = torch.max(volume_show, output_show)
        writer.add_image('Composite_GT', composite_1, iteration)
        writer.add_image('Composite_PD', composite_2, iteration)

def visualize_aff(volume, label, output, iteration, writer):
    volume, label, output = prepare_data(volume, label, output)
    #print('ho:',volume.shape,label.shape)

    sz = volume.size() # z,c,y,x
    canvas = []
    volume_visual = volume.detach().cpu().expand(sz[0],3,sz[2],sz[3])
    canvas.append(volume_visual)

    output_visual = [output[:,i].detach().cpu().unsqueeze(1).expand(sz[0],3,sz[2],sz[3]) for i in range(sz[1])]
    label_visual = [label[:,i].detach().cpu().unsqueeze(1).expand(sz[0],3,sz[2],sz[3]) for i in range(sz[1])]
    canvas = canvas + output_visual
    canvas = canvas + label_visual
    canvas_merge = torch.cat(canvas, 0)
    canvas_show = vutils.make_grid(canvas_merge, nrow=8, normalize=True, scale_each=True)

    writer.add_image('Affinity', canvas_show, iteration)
