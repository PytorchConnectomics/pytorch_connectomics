import os,time
import numpy as np
from ..data.utils import blend_gaussian
from ..io import writeh5
import torch
import itertools
import pudb

def test(args, test_loader, model, do_eval=True, do_3d=True, model_output_id=None, output_name='result.h5'):
    if do_eval:
        model.eval()
    else:
        model.train()
    volume_id = 0
#     pudb.set_trace()
    ww = blend_gaussian(args.model_output_size)#!MB
    NUM_OUT = args.model_out_channel
    pad_size = args.pad_size
    if len(args.pad_size)==3:
        pad_size = [args.pad_size[0],args.pad_size[0],
                    args.pad_size[1],args.pad_size[1],
                    args.pad_size[2],args.pad_size[2]]
    
    if(args.architecture == "super"):
        output_size = (np.array(test_loader.dataset.input_size)*np.array(args.scale_factor)).tolist()
        result = [np.stack([np.zeros(x, dtype=np.float32) for _ in range(NUM_OUT)]) for x in output_size]
    else:
        result = [np.stack([np.zeros(x, dtype=np.float32) for _ in range(NUM_OUT)]) for x in test_loader.dataset.input_size]
        weight = [np.zeros(x, dtype=np.float32) for x in test_loader.dataset.input_size]

    # print(result[0].shape, weight[0].shape)

    start = time.time()

    sz = tuple([NUM_OUT] + list(args.model_output_size))
    with torch.no_grad():
        for _, (pos, volume) in enumerate(test_loader):
            volume_id += args.batch_size
            print('volume_id:', volume_id)

            # for gpu computing
            volume = torch.from_numpy(volume).to(args.device)
            if not do_3d:
                volume = volume.squeeze(1)

            if args.test_aug_num!=0:
                output = inference_aug16(model, volume,args.test_aug_mode, args.test_aug_num)
            else:
                output = model(volume).cpu().detach().numpy()

            if model_output_id is not None: # select channel
                output = output[model_output_id]
            if args.architecture != "super":
                for idx in range(output.shape[0]):
                    st = pos[idx]
                    result[st[0]][:, st[1]:st[1]+sz[1], st[2]:st[2]+sz[2], \
                    st[3]:st[3]+sz[3]] += output[idx] * np.expand_dims(ww, axis=0)
                    weight[st[0]][st[1]:st[1]+sz[1], st[2]:st[2]+sz[2], \
                    st[3]:st[3]+sz[3]] += ww
            else:
                for idx in range(output.shape[0]):
                    st = pos[idx]
                    result[st[0]][:, st[1]:st[1]+sz[1], st[2]:st[2]+sz[2], \
                    st[3]:st[3]+sz[3]] += output[idx]

    end = time.time()
    print("prediction time:", (end-start))

    if args.architecture != "super":
        for vol_id in range(len(result)):
            if result[vol_id].ndim > weight[vol_id].ndim:
                weight[vol_id] = np.expand_dims(weight[vol_id], axis=0)
            result[vol_id] = (result[vol_id]/weight[vol_id]*255).astype(np.uint8)
            sz = result[vol_id].shape
            result[vol_id] = result[vol_id][:,
                        pad_size[0]:sz[1]-pad_size[1],
                        pad_size[2]:sz[2]-pad_size[3],
                        pad_size[4]:sz[3]-pad_size[5]]

    if args.output_path is None:
        return result
    else:
        print('save h5')
        writeh5(os.path.join(args.output_path, output_name), result,['vol%d'%(x) for x in range(len(result))])

def inference_aug16(model, data, mode='min', num_aug=4):
    out = None
    cc = 0
    if num_aug ==4:
        opts = itertools.product((False, ), (False, True), (False, True), (False, ))
    else:
        opts = itertools.product((False, True), (False, True), (False, True), (False, True))

    for xflip, yflip, zflip, transpose in opts:
        extension = ""
        if transpose:
            extension += "t"
        if zflip:
            extension += "z"
        if yflip:
            extension += "y"
        if xflip:
            extension += "x"
        volume = data.clone()
        # batch_size,channel,z,y,x 

        if xflip:
            volume = torch.flip(volume, [4])
        if yflip:
            volume = torch.flip(volume, [3])
        if zflip:
            volume = torch.flip(volume, [2])
        if transpose:
            volume = torch.transpose(volume, 3, 4)
        # aff: 3*z*y*x 
        vout = model(volume).cpu().detach().numpy()

        if transpose: # swap x-/y-affinity
            vout = vout.transpose(0, 1, 2, 4, 3)
            vout[:,[1,2]] = vout[:,[2,1]]
        if zflip:
            vout = vout[:,:,::-1]
        if yflip:
            vout = vout[:,:, :, ::-1]
        if xflip:
            vout = vout[:,:, :, :, ::-1]
        if out is None:
            if mode == 'min':
                out = np.ones(vout.shape,dtype=np.float32)
            elif mode == 'mean':
                out = np.zeros(vout.shape,dtype=np.float32)
        if mode == 'min':
            out = np.minimum(out,vout)
        elif mode == 'mean':
            out += vout
        cc+=1
    if mode == 'mean':
        out = out/cc

    return out
