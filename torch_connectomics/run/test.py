import numpy as np
from torch_connectomics.utils.net import *

def test(args, test_loader, model, device, model_io_size, pad_size, do_eval=True, do_3d=True, model_output_id=None):
    if do_eval:
        # switch to eval mode
        model.eval()
    else:
        model.train()
    volume_id = 0
    ww = blend(model_io_size)
    NUM_OUT = args.out_channel

    result = [np.stack([np.zeros(x, dtype=np.float32) for _ in range(NUM_OUT)]) for x in test_loader.dataset.input_size]
    weight = [np.zeros(x, dtype=np.float32) for x in test_loader.dataset.input_size]
    print(result[0].shape, weight[0].shape)

    start = time.time()
    with torch.no_grad():
        for i, (pos, volume) in enumerate(test_loader):
            volume_id += args.batch_size
            print('volume_id:', volume_id)

            # for gpu computing
            volume = volume.to(device)
            if do_3d:
                output = model(volume)
            else:
                output = model(volume.squeeze(1))

            if model_output_id is not None:
                output = output[model_output_id]

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
        sz = data.shape
        data = data[:,
                    pad_size[0]:sz[1]-pad_size[0],
                    pad_size[1]:sz[2]-pad_size[1],
                    pad_size[2]:sz[3]-pad_size[2]]
        print('Output shape: ', data.shape)
        hf = h5py.File(args.output+'volume_'+str(vol_id)+'.h5','w')
        hf.create_dataset('main', data=data, compression='gzip')
        hf.close()
