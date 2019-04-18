from __future__ import division
import os,sys
import numpy as np
import waterz

sys.path.append('../')
def readh5(filename, datasetname='main'):
    import h5py
    return np.array(h5py.File(filename,'r')[datasetname])

opt = sys.argv[1]
Dv = '/n/coxfs01/vcg_connectomics/snemi/' 
gt = None
f2 = './tmp/'
if opt[0]=='0':
    nn=''
    if opt =='0': # snemi
        f1=Dv+'seg_zwaterz/zwatershed2d/test2d.h5'
        seg2d = readh5(f1) 
        aff = readh5(Dv+'affs/tmquan_0524/model_snemi_dice_mls._min.h5')
    elif opt =='0.01': # snemi-train: quan aff
        seg2d = None
        aff = readh5(Dv+'affs/tmquan_0524/model_snemi_dice_mls._train_min.h5')
        gt = readh5(Dv+'label/train-labels.h5').astype(np.uint32)
        f2='tmp/qua/'
    elif opt =='0.02': # snemi-train: zudi aff
        seg2d = None
        aff = readh5('/n/coxfs01/zudilin/data/result20k/volume_0.h5').astype(np.float32)/255.0
        gt = readh5(Dv+'label/train-labels.h5').astype(np.uint32)
        f2='tmp/zudi/'

    elif opt =='0.1': # fiber
        D0='/n/coxfs01/vcg_connectomics/cerebellum_P7/gt-seg_pf1/felix/z0-104/'
        f1=D0+'zwatershed.h5' 
        f2=D0+'zwaterz/'
        seg2d = readh5(f1).astype(np.uint64) 
        aff = np.stack([readh5(D0+'aff_z.h5'),readh5(D0+'aff_y.h5'),readh5(D0+'aff_x.h5')],axis=0).astype(np.float32)/255 
    elif opt =='0.2': # fibsem
        D0='/n/coxfs01/vcg_connectomics/Fib25/FibSEM/result_unet3D_felix/'
        nn='fibsem_test_data' 
        #nn='500-distal' 
        seg2d = readh5(D0+'zwatershed/'+nn+'_v1.h5').astype(np.uint64) 
        aff = readh5(D0+'affinity/'+nn+'-pred.h5') 
        f2=D0+'zwaterz/'

    if not os.path.exists(f2):
        os.mkdir(f2)

    if seg2d is not None:
        # make unique ids:
        next_id=np.uint64(0)
        # change value by reference
        for zid in range(seg2d.shape[0]):
            tile = seg2d[zid]
            tile[np.where(tile>0)] += next_id
            next_id = tile.max()

    low=0.05; high=0.995
    mf = 'aff85_his256';T_thres = [0.2, 0.3, 0.4,0.5, 0.6]
    #mf = 'aff50_his256';T_thres = [0.4,0.5,0.6,0.7,0.8]
    waterz.waterz(aff, T_thres, merge_function=mf, gt_border=0, output_prefix=f2+nn+'_'+mf,
                        fragments=seg2d, aff_threshold=[low,high],return_seg=False, gt=gt)
