import os,sys
import h5py
import numpy as np
import time
import imageio
import argparse

from connectomics.utils.seg.seg_dist import DilateData
from connectomics.utils.seg.seg_util import relabel
from connectomics.utils.seg.io_util import writeh5, readh5
from connectomics.utils.seg.seg_eval import adapted_rand

from skimage.morphology import dilation,erosion

def get_args():
    parser = argparse.ArgumentParser(description='Specifications for segmentation.')
    parser.add_argument('-gt',  default='/home/gt_seg/', help='path to groundtruth segmentation')
    parser.add_argument('-pd',  default='/home/pd_aff/', help='path to predicted affinity graph')
    parser.add_argument('--mode', type=int, default=1, help='segmentation method')
    parser.add_argument('--save', type=bool, default=False, help='save segmentation')
    args = parser.parse_args()
    return args

args = get_args()

# add affinity location
D_aff = args.pd
aff = np.array(h5py.File(D_aff)['main'])
print('dtype of affinity graph:', aff.dtype)
assert aff.dtype in [np.uint8, np.float32]
if aff.dtype == np.uint8: aff = (aff/255.0).astype(np.float32)

print('shape of affinity graph:', aff.shape)
# ground truth

#D0='/n/coxfs01/zudilin/research/mitoNet/data/file/snemi/label/'
D_seg = args.gt
suffix = D_seg.strip().split('.')[-1]
assert suffix in ['tif', 'h5']
if suffix == 'tif':
    seg = imageio.volread(D_seg).astype(np.uint32)
else:
    seg = readh5(D_seg).astype(np.uint32)

print('shape of gt segmenation:', seg.shape)

if args.mode == 0: 
    # 3D zwatershed
    import zwatershed as zwatershed
    #print('zwatershed:', zwatershed.__version__)
    st = time.time()
    T_aff=[0.05,0.995,0.2]
    T_thres = [800]
    T_dust=600
    T_merge=0.9
    T_aff_rel=1
    out = zwatershed.zwatershed(aff, T_thres, T_aff=T_aff, \
                                T_dust=T_dust, T_merge=T_merge,T_aff_relative=T_aff_rel)[0][0]
    et = time.time()
    out = relabel(out)
    sn = '%s_%f_%f_%d_%f_%d_%f_%d.h5'%(args.mode,T_aff[0],T_aff[1],T_thres[0],T_aff[2],T_dust,T_merge,T_aff_rel) 

elif args.mode == 1:
    # waterz
    import waterz
    print('waterz:', waterz.__version__)
    st = time.time()
    low=0.05; high=0.995
    mf = 'aff85_his256'; T_thres = [0.3]
    out = waterz.waterz(aff, T_thres, merge_function=mf, gt_border=0,
                        fragments=None, aff_threshold=[low, high], return_seg=True, gt=seg)[0]
    et = time.time()
    out = relabel(out)
    print(out.shape)
    sn = '%s_%f_%f_%f_%s.h5'%(args.mode,low,high,T_thres[0],mf) 
    
elif args.mode == 2:
    # 2D zwatershed + waterz
    import waterz
    import zwatershed
    print('waterz:', waterz.__version__)
    print('zwatershed:', zwatershed.__version__)
    st = time.time()
    T_thres = [150]
    T_aff=[0.05,0.8,0.2]
    T_dust=150
    T_merge=0.9
    T_aff_rel=1
    sz = np.array(aff.shape)
    out = np.zeros(sz[1:],dtype=np.uint64)
    id_st = np.uint64(0)
    # need to relabel the 2D seg, o/w out of bound
    for z in range(sz[1]):
        out[z] = relabel(zwatershed.zwatershed(aff[:,z:z+1], T_thres, T_aff=T_aff, \
                                T_dust=T_dust, T_merge=T_merge,T_aff_relative=T_aff_rel)[0][0])

        out[z][np.where(out[z]>0)] += id_st
        id_st = out[z].max()
    
    mf = 'aff50_his256';T_thres2 = [0.5]
    out = waterz.waterz(affs=aff, thresholds=T_thres2, fragments=out, merge_function=mf)[0]
    et = time.time()
    sn = '%s_%f_%f_%d_%f_%d_%f_%d_%f_%s.h5'%(args.mode,T_aff[0],T_aff[1],T_thres[0],T_aff[2],T_dust,T_merge,T_aff_rel,T_thres2[0],mf) 

else:
    print('The segmentation method is not implemented yet!')
    raise NotImplementedError

print('time: %.1f s'%((et-st)))
# do evaluation

score = adapted_rand(out.astype(np.uint32), seg)
print('Adaptive rand: ', score) 
    # 0: 0.22
    # 1: 0.098
    # 2: 0.137
# do save
if args.save:
    result_dir = os.path.dirname(args.pd) + '/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    writeh5( result_dir + sn, 'main', out)
