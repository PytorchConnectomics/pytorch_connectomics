from __future__ import division, print_function
import os,sys
import numpy as np
import h5py, json
from scipy.misc import imread
from skimage.transform import resize

# Functions
#-------------------------------
def get_dataset_image(dataset_dict, x0p, x1p, y0p, y1p, z0p, z1p):
    # calculate padding at the boundary
    #print('original: ', z0p, z1p, y0p, y1p, x0p, x1p)
    boundary = [z0p<z0, z1p>z1, y0p<y0, y1p>y1, x0p<x0, x1p>x1]
    #print('touch boundary? ', boundary)
    pad_size = [z0p-z0, z1p-z1, 
                y0p-y0, y1p-y1,
                x0p-x0, x1p-x1]
    pad_need = list(np.array(boundary).astype(int)*np.abs(np.array(pad_size)))
    #print(pad_need)
    z0p = z0p + pad_need[0]
    z1p = z1p - pad_need[1]
    y0p = y0p + pad_need[2]
    y1p = y1p - pad_need[3]
    x0p = x0p + pad_need[4]
    x1p = x1p - pad_need[5]

    #print('adjusted: ', z0p, z1p, y0p, y1p, x0p, x1p)
    result = np.zeros((z1p-z0p, y1p-y0p, x1p-x0p), np.uint8)
    c0 = x0p // tile_sz # floor
    c1 = (x1p + tile_sz-1) // tile_sz # ceil
    r0 = y0p // tile_sz
    r1 = (y1p + tile_sz-1) // tile_sz
    for z in range(z0p, z1p):
        for row in range(r0, r1):
            for column in range(c0, c1):
                pattern = dataset_dict["sections"][z]
                path = pattern.format(row=row+1, column=column+1)
                if not os.path.exists(path):
                    print('no file: ', path)
                    patch = np.zeros((3750, 3750), dtype=np.uint8)
                else:    
                    patch = imread(path, 0)
                    if down_ratio != 1: # float -> fraction
                        patch = resize(patch, np.array(patch.shape) // down_ratio, mode='reflect',
                                order=1, anti_aliasing=True, anti_aliasing_sigma=(0.25, 0.25))
                        # [0,1] --> [0, 255] 
                        patch = (patch * 255).astype(np.uint8)
                xp0 = column * tile_sz
                xp1 = (column+1) * tile_sz
                yp0 = row * tile_sz
                yp1 = (row + 1) * tile_sz
                if patch is not None:
                    x0a = max(x0p, xp0)
                    x1a = min(x1p, xp1)
                    y0a = max(y0p, yp0)
                    y1a = min(y1p, yp1)
                    result[z-z0p, y0a-y0p:y1a-y0p, x0a-x0p:x1a-x0p] = patch[y0a-yp0:y1a-yp0, x0a-xp0:x1a-xp0]

    if True in boundary:
        result = np.pad(result, 
                ((pad_need[0], pad_need[1]),
                 (pad_need[2], pad_need[3]),
                 (pad_need[4], pad_need[5])),'reflect')
    #print('input shape: ', result.shape)
    return result

def get_dataset_label(path, x0p, x1p, y0p, y1p, z0p, z1p):
    # calculate padding at the boundary
    #print('original: ', z0p, z1p, y0p, y1p, x0p, x1p)
    boundary = [z0p<z0, z1p>z1, y0p<y0, y1p>y1, x0p<x0, x1p>x1]
    #print('touch boundary? ', boundary)
    pad_size = [z0p-z0, z1p-z1, 
                y0p-y0, y1p-y1,
                x0p-x0, x1p-x1]
    pad_need = list(np.array(boundary).astype(int)*np.abs(np.array(pad_size)))
    #print(pad_need)
    z0p = z0p + pad_need[0]
    z1p = z1p - pad_need[1]
    y0p = y0p + pad_need[2]
    y1p = y1p - pad_need[3]
    x0p = x0p + pad_need[4]
    x1p = x1p - pad_need[5]

    #print('adjusted: ', z0p, z1p, y0p, y1p, x0p, x1p)

    # need to get label from multiple volumes
    sz = (3, z1p-z0p, y1p-y0p, x1p-x0p)
    result = np.zeros(sz, dtype=np.uint8)
    # floor
    zf = (z0p-z0) // volume_size[0]
    yf = (y0p-y0) // volume_size[1]
    xf = (x0p-x0) // volume_size[2]
    # ceiling
    zc = min((z1p-z0) // volume_size[0] + 1, (z1-z0) // volume_size[0])
    yc = min((y1p-y0) // volume_size[1] + 1, (y1-y0) // volume_size[1])
    xc = min((x1p-x0) // volume_size[2] + 1, (x1-x0) // volume_size[2])
    # zc = min(z1p // volume_size[0] + 1, (z1-z0-1) // volume_size[0])
    # yc = min(y1p // volume_size[1] + 1, (y1-y0-1) // volume_size[1])
    # xc = min(x1p // volume_size[2] + 1, (x1-x0-1) // volume_size[2])

    #print(zf,yf,xf)
    #print(zc,yc,xc)
    for z in range(zf, zc):
        for y in range(yf, yc):
            for x in range(xf, xc):
                zp0 = z * volume_size[0] + z0
                zp1 = (z+1) * volume_size[0] + z0
                yp0 = y * volume_size[1] + y0
                yp1 = (y+1) * volume_size[1] + y0
                xp0 = x * volume_size[2] + x0
                xp1 = (x+1) * volume_size[2] + x0
                fn = path+'/%d/%d/%d/mask.h5'%(xp0,yp0,zp0)
                if os.path.exists(fn):
                    fl = h5py.File(fn, 'r')
                    data = np.array(fl['main'])

                    z0a = max(z0p, zp0)
                    z1a = min(z1p, zp1)
                    y0a = max(y0p, yp0)
                    y1a = min(y1p, yp1)
                    x0a = max(x0p, xp0)
                    x1a = min(x1p, xp1)

                    # print('z0p, z1p, y0p, y1p, x0p, x1p')
                    # print(z0p, z1p, y0p, y1p, x0p, x1p)
                    # print('z0a, z1a, y0a, y1a, x0a, x1a')
                    # print(z0a, z1a, y0a, y1a, x0a, x1a)
                    # print('zp0, zp1, yp0, yp1, xp0, xp1')
                    # print(zp0, zp1, yp0, yp1, xp0, xp1)

                    # print('========')

                    # print(z0a-z0p, z1a-z0p, y0a-y0p, y1a-y0p, x0a-x0p, x1a-x0p)
                    # print(z0a-zp0, z1a-zp0, y0a-yp0, y1a-yp0, x0a-xp0, x1a-xp0)

                    result[:, z0a-z0p:z1a-z0p, y0a-y0p:y1a-y0p, x0a-x0p:x1a-x0p] = data[:, z0a-zp0:z1a-zp0, y0a-yp0:y1a-yp0, x0a-xp0:x1a-xp0]
                    fl.close()
                else:
                    print('No file: ', fn)
                    exit(0)
                  
    if True in boundary:
        # print(result.shape)
        # print([(0,0),
        #        (pad_need[0], pad_need[1]),
        #        (pad_need[2], pad_need[3]),
        #        (pad_need[4], pad_need[5])])
        result = np.pad(result, 
                [(0,0),
                 (pad_need[0], pad_need[1]),
                 (pad_need[2], pad_need[3]),
                 (pad_need[4], pad_need[5])], 'reflect')
    #print('input shape: ', result.shape)
    return result  

def random_window(w0, w1, sz, shift=False):
    assert (w1 >= w0)
    diff = np.abs((w1-w0)-sz)
    if (w1-w0) <= sz:
        if shift==True: # shift augmentation
            low = np.random.randint(w0-diff//2-5, w0-diff//2+5)
        else:
            low = w0 - diff//2 
    else:
        if shift==True: # shift augmentation
            low = np.random.randint(w0, w1-sz)
        else:
            low = w0 + diff//2 
    high = low+sz    
    return low, high

def process_bbox(bfly_db,bbox, gt=True, use_gt=True):
    #print(bbox)
    z0p, z1p, y0p, y1p, x0p, x1p = bbox
    #print('original: ', z0p, z1p, y0p, y1p, x0p, x1p)
    # cerebellum dataset
    x0p = x0p + x0
    x1p = x1p + x0
    y0p = y0p + y0
    y1p = y1p + y0
    #print('original: ', z0p, z1p, y0p, y1p, x0p, x1p)

    # start
    z0p, z1p = random_window(z0p, z1p, model_io_size[0], shift=False)
    y0p, y1p = random_window(y0p, y1p, model_io_size[1], shift=False)
    x0p, x1p = random_window(x0p, x1p, model_io_size[2], shift=False)
    image = get_dataset_image(bfly_db, x0p, x1p, y0p, y1p, z0p, z1p)
    image = image.astype(np.float32) / 255.0 #normalize
    # if gt==True: # if it is a synapse
    #     label = get_dataset_label(mask_path, x0p, x1p, y0p, y1p, z0p, z1p)
    #     label = (label > 128).astype(np.float32)
    #     #label = (label/255.0).astype(np.float32)
    # else:
    #     label = np.zeros(label_io_size, dtype=np.float32)    

    if use_gt==True:
        label = get_dataset_label(mask_path, x0p, x1p, y0p, y1p, z0p, z1p)
        label = (label > 128).astype(np.float32)
        if gt==True: # if it is a synapse
            pseudo_label = label
        else: # remove things in bbox
            pseudo_label = label.copy()
            lz = np.max([bbox[0]-z0p, 0])
            hz = np.min([bbox[1]-z0p, model_io_size[0]])
            ly = np.max([bbox[2]-y0p, 0])
            hy = np.min([bbox[3]-y0p, model_io_size[1]])
            lx = np.max([bbox[4]-x0p, 0])
            hx = np.min([bbox[5]-x0p, model_io_size[2]])
            pseudo_label[:, lz:hz, ly:hy, lx:hx] = 0

    else:
        pseudo_label = np.zeros(label_io_size, dtype=np.float32)                

    # hk = h5py.File('visual.h5','w')
    # hk.create_dataset('main',data=image)
    # hk.close()

    # hk = h5py.File('label.h5','w')
    # hk.create_dataset('main',data=label)
    # hk.close()
    #print(z0p, z1p, y0p, y1p, x0p, x1p)
    return image, pseudo_label

if __name__== "__main__":
    print('T_bbox')
    # Parameters
    #-------------------------------
    #1. JWR Dataset
    D0 = 'path/to/data/'
    mask_path = D0 + 'synapse'
    #bbox_path = D0 + 'bbox'
    #pfrd_path = D0 + 'proofread'
    bfly_path = D0 + 'bfly_v2-2_add8.json'
    bfly_db = json.load(open(bfly_path))

    tile_sz = 2560
    z0=0; z1=3072
    y0=0; y1=12320
    x0=0; x1=12320
    model_io_size = (4, 128, 128)
    label_io_size = tuple([3]+list(model_io_size))
    volume_size = (96, 1120, 1120)
    down_ratio = 1


    # bbox_path = '/path/to/bbox/'
    # bbox_text = bbox_path + 'bb.txt'
    # g_text = bbox_path + 'pseudo_label/pseudo_label_1.0_g.txt' 
    # b_text = bbox_path + 'pseudo_label/pseudo_label_1.0_b.txt' 
    # print(bbox_path)
    # print(g_text)
    # print(b_text)

    # bbox = np.loadtxt(bbox_text, delimiter=',', dtype=int)
    # bbox = bbox.astype(int)
    # g_list = np.loadtxt(g_text, dtype=int)
    # b_list = np.loadtxt(b_text, dtype=int)
    # print('positive: ', len(g_list))
    # print('negative: ', len(b_list))

    # MASK = []
    
    # input_bbox = bbox[g_list[0]]
    # print(input_bbox)
    # gt = True
    # image, pseudo_label = process_bbox(input_bbox, gt, use_gt=True)
