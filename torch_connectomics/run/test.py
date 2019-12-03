import numpy as np
from torch_connectomics.utils.net import *

def test(args, test_loader, model, device, model_io_size, pad_size, do_eval=True, do_3d=True, model_output_id=None):
    if do_eval:
        model.eval()
    else:
        model.train()
    volume_id = 0
    ww = blend(model_io_size)
    NUM_OUT = args.out_channel
    if len(pad_size)==3:
        pad_size = [pad_size[0],pad_size[0],
                    pad_size[1],pad_size[1],
                    pad_size[2],pad_size[2]]

    result = [np.stack([np.zeros(x, dtype=np.float32) for _ in range(NUM_OUT)]) for x in test_loader.dataset.input_size]
    weight = [np.zeros(x, dtype=np.float32) for x in test_loader.dataset.input_size]
    # print(result[0].shape, weight[0].shape)

    start = time.time()

    sz = tuple([NUM_OUT] + list(model_io_size))
    with torch.no_grad():
        for _, (pos, volume) in enumerate(test_loader):
            volume_id += args.batch_size
            print('volume_id:', volume_id)

            # for gpu computing
            volume = volume.to(device)
            if not do_3d:
                volume = volume.squeeze(1)

            if args.test_augmentation:
                output = inference_aug16(model, volume, mode= 'mean')
            else:
                output = model(volume).cpu().detach().numpy()

            if model_output_id is not None: # select channel
                output = output[model_output_id]

            for idx in range(output.shape[0]):
                ii = (volume_id - args.batch_size) + idx
                save_each(args, volume[idx], output[idx], ii)
            
            for idx in range(output.shape[0]):
                st = pos[idx]
                result[st[0]][:, st[1]:st[1]+sz[1], st[2]:st[2]+sz[2], \
                st[3]:st[3]+sz[3]] += output[idx] * np.expand_dims(ww, axis=0)
                weight[st[0]][st[1]:st[1]+sz[1], st[2]:st[2]+sz[2], \
                st[3]:st[3]+sz[3]] += ww

    end = time.time()
    print("prediction time:", (end-start))

    for vol_id in range(len(result)):
        if result[vol_id].ndim > weight[vol_id].ndim:
            weight[vol_id] = np.expand_dims(weight[vol_id], axis=0)
        result[vol_id] = (result[vol_id]/weight[vol_id]*255).astype(np.uint8)
        sz = result[vol_id].shape
        result[vol_id] = result[vol_id][:,
                    pad_size[0]:sz[1]-pad_size[1],
                    pad_size[2]:sz[2]-pad_size[3],
                    pad_size[4]:sz[3]-pad_size[5]]

    if args.output is None:
        return result
    else:
        print('save h5')
        hf = h5py.File(args.output + '/result.h5','w')
        for vol_id in range(len(result)): 
            hf.create_dataset('vol%d'%(vol_id), data=result[vol_id], compression='gzip')
        hf.close()

def inference_aug16(model, data, mode= 'min'):
    out = None
    cc = 0
    for xflip, yflip, zflip, transpose in itertools.product(
                    (False, ), (False, True), (False, True), (False, )):
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
            vout = vout[:,:, :, :, :, ::-1]
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

# -----------------------
#    utils
# -----------------------

def save_each(args, volume, output, idx):
    # volume: (C,Z,Y,X)
    volume = volume.cpu().detach().numpy()
    volume = np.concatenate([volume, volume, volume], 0).transpose((1, 2, 3, 0))
    output = output.transpose((1, 2, 3, 0))
    composite = np.maximum(volume, output)
    composite = (composite*255).astype(np.uint8)

    hf = h5py.File(args.output + '/composite_%d.h5' % (idx), 'w')
    hf.create_dataset('main', data=composite)
    hf.close()