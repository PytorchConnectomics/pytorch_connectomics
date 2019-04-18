import os,sys
import numpy as np
import torch
import h5py, time, itertools, datetime
from torch_connectomics.utils.net import *

def test(args, test_loader, model, device, model_io_size, volume_shape, pad_size):
    # switch to eval mode
    model.eval()
    volume_id = 0
    ww = blend(model_io_size)
    NUM_OUT = 3

    result = [np.stack([np.zeros(x, dtype=np.float32) for _ in range(NUM_OUT)]) for x in volume_shape]
    weight = [np.zeros(x, dtype=np.float32) for x in volume_shape]
    print(result[0].shape, weight[0].shape)

    start = time.time()
    with torch.no_grad():
        for i, (pos, volume) in enumerate(test_loader):
            volume_id += args.batch_size
            print('volume_id:', volume_id)

            # for gpu computing
            volume = volume.to(device)
            output = model(volume)

            sz = tuple([NUM_OUT]+list(model_io_size))
            for idx in range(output.size()[0]):
                st = pos[idx]
                result[st[0]][:, st[1]:st[1]+sz[1], st[2]:st[2]+sz[2], \
                st[3]:st[3]+sz[3]] += output[idx].cpu().detach().numpy().reshape(sz) * np.expand_dims(ww, axis=0)
                weight[st[0]][st[1]:st[1]+sz[1], st[2]:st[2]+sz[2], \
                st[3]:st[3]+sz[3]] += ww

    end = time.time()
    print("prediction time:", (end-start))

    for vol_id in range(len(result)):
        result[vol_id] = result[vol_id] / weight[vol_id]
        data = (result[vol_id]*255).astype(np.uint8)
        data = data[:,
                    pad_size[0]:-pad_size[0],
                    pad_size[1]:-pad_size[1],
                    pad_size[2]:-pad_size[2]]

        print('Output shape: ', data.shape)
        hf = h5py.File(args.output+'/volume_'+str(vol_id)+'.h5','w')
        hf.create_dataset('main', data=data, compression='gzip')
        hf.close()

def main():
    args = get_args(mode='test')

    print('0. initial setup')
    model_io_size, device = init(args)
    print('model I/O size:', model_io_size) 

    print('1. setup data')
    test_loader, volume_shape, pad_size = get_input(args, model_io_size, 'test')

    print('2. setup model')
    model = setup_model(args, device, exact=True)

    print('3. start testing')
    test(args, test_loader, model, device, model_io_size, volume_shape, pad_size)
  
    print('4. finish testing')

if __name__ == "__main__":
    main()