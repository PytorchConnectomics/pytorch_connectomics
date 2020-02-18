import numpy as np
from scipy.sparse import coo_matrix
from scipy.ndimage.morphology import binary_erosion, binary_dilation
from skimage.morphology import erosion, dilation
from skimage.measure import label as label_cc # avoid namespace conflict

from .data_affinity import seg_to_aff

# reduce the labeling
def getSegType(mid):
    m_type = np.uint64
    if mid<2**8:
        m_type = np.uint8
    elif mid<2**16:
        m_type = np.uint16
    elif mid<2**32:
        m_type = np.uint32
    return m_type

def relabel(seg, do_type=False):
    # get the unique labels
    uid = np.unique(seg)
    # get the maximum label for the segment
    mid = int(uid.max()) + 1

    # create an array from original segment id to reduced id
    m_type = seg.dtype
    if do_type:
        m_type = getSegType(mid)
    mapping = np.zeros(mid, dtype=m_type)
    mapping[uid] = np.arange(len(uid), dtype=m_type)
    return mapping[seg]

def remove_small(seg, thres=100):                                                                    
    sz = seg.shape                                                                                   
    seg = seg.reshape(-1)                                                                            
    uid, uc = np.unique(seg, return_counts=True)                                                     
    seg[np.in1d(seg,uid[uc<thres])] = 0                                                              
    return seg.reshape(sz)

def im2col(A, BSZ, stepsize=1):
    # Parameters
    M,N = A.shape
    # Get Starting block indices
    start_idx = np.arange(0,M-BSZ[0]+1,stepsize)[:,None]*N + np.arange(0,N-BSZ[1]+1,stepsize)
    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(BSZ[0])[:,None]*N + np.arange(BSZ[1])
    # Get all actual indices & index into input array for final output
    return np.take(A,start_idx.ravel()[:,None] + offset_idx.ravel())

def seg_widen_border(seg, tsz_h=1):
    # Kisuk Lee's thesis (A.1.4) 
    # we preprocessed the ground truth seg such that any voxel centered on a 3 × 3 × 1 window containing more than one positive segment ID (zero is reserved for background) is marked as background
    # seg=0: background
    tsz = 2*tsz_h+1
    sz = seg.shape
    if len(sz)==3:
        for z in range(sz[0]):
            mm = seg[z].max()
            patch = im2col(np.pad(seg[z],((tsz_h,tsz_h),(tsz_h,tsz_h)),'reflect'),[tsz,tsz])
            p0=patch.max(axis=1)
            patch[patch==0] = mm+1
            p1=patch.min(axis=1)
            seg[z] =seg[z]*((p0==p1).reshape(sz[1:]))
    else:
        mm = seg.max()
        patch = im2col(np.pad(seg,((tsz_h,tsz_h),(tsz_h,tsz_h)),'reflect'),[tsz,tsz])
        p0=patch.max(axis=1)
        patch[patch==0] = mm+1
        p1=patch.min(axis=1)
        seg =seg*((p0==p1).reshape(sz[1:]))
    return seg

def seg_to_small_seg(seg,thres=25,rr=2):
    # rr: z/x-y resolution ratio
    sz = seg.shape
    mask = np.zeros(sz,np.uint8)
    for z in range(sz[0]):
        tmp = label_cc(seg[z])
        ui,uc = np.unique(tmp,return_counts=True)
        rl = np.zeros(ui[-1]+1,np.uint8)
        rl[ui[uc<thres]]=1;rl[0]=0
        mask[z] += rl[tmp]
    for y in range(sz[1]):
        tmp = label_cc(seg[:,y])
        ui,uc = np.unique(tmp,return_counts=True)
        rl = np.zeros(ui[-1]+1,np.uint8)
        rl[ui[uc<thres//rr]]=1;rl[0]=0
        mask[:,y] += rl[tmp]
    for x in range(sz[2]):
        tmp = label_cc(seg[:,:,x])
        ui,uc = np.unique(tmp,return_counts=True)
        rl = np.zeros(ui[-1]+1,np.uint8)
        rl[ui[uc<thres//rr]]=1;rl[0]=0
        mask[:,:,x] += rl[tmp]
    return mask

def seg_to_instance_bd(seg, tsz_h=7, do_bg=False):
    tsz = tsz_h*2+1
    mm = seg.max()
    sz = seg.shape
    bd = np.zeros(sz, np.uint8)
    for z in range(sz[0]):
        patch = im2col(np.pad(seg[z], ((tsz_h,tsz_h),(tsz_h,tsz_h)),'reflect'),[tsz,tsz])
        p0 = patch.max(axis=1)
        if do_bg: # at least one non-zero seg
            p1 = patch.min(axis=1)
            bd[z] = ((p0>0)*(p0!=p1)).reshape(sz[1:])
        else: # between two non-zero seg
            patch[patch==0] = mm+1
            p1 = patch.min(axis=1)
            bd[z] = ((p0!=0)*(p1!=0)*(p0!=p1)).reshape(sz[1:])
    return bd

def markInvalid(seg, iter_num=2, do_2d=True):
    # find invalid 
    # if do erosion(seg==0), then miss the border
    if do_2d:
        stel=np.array([[1,1,1], [1,1,1]]).astype(bool)
        if len(seg.shape)==2:
            out = binary_dilation(seg>0, structure=stel, iterations=iter_num)
            seg[out==0] = -1
        else: # save memory
            for z in range(seg.shape[0]):
                tmp = seg[z] # by reference
                out = binary_dilation(tmp>0, structure=stel, iterations=iter_num)
                tmp[out==0] = -1
    else:
        stel=np.array([[1,1,1], [1,1,1], [1,1,1]]).astype(bool)
        out = binary_dilation(seg>0, structure=stel, iterations=iter_num)
        seg[out==0] = -1
    return seg

def label_to_weight(lopt, label, mask=None):
    out = None
    if lopt in ['0','1']: 
        out = [rebalance_binary_class(label, mask)]
    elif lopt[0] in ['2','3']:
        if lopt!='2'and lopt!='3':
            out = [rebalance_binary_class(label, mask), None]
    return out

def label_to_target(topt, label):
    # with batch_size
    # mito/synapse cleft binary: topt = 0 
    # synapse polarity: topt = 1.2,0 
    sz = list(label.shape)
    if topt == '0': # binary
        out = (label>0).astype(np.float32)
    elif topt[0] == '1': # multi-channel, e.g. 1.2
        num_channel = int(topt[2:])
        tmp = [None]*num_channel 
        for j in range(num_channel):
            tmp[j] = label==(j+1)
        # concatenate at channel
        out = np.concatenate(tmp,1).astype(np.float32)
    elif topt[0] == '2': # affinity
        out = np.zeros([sz[0],3]+sz[2:], np.float32)
        for i in range(sz[0]):
            out[i] = seg_to_aff(label[i,0])
    elif topt[0] == '3': # small object mask
        # size_thres: 2d threshold for small size
        # zratio: resolution ration between z and x/y
        # border_size: size of the border
        _, size_thres, zratio, border_size = [int(x) for x in topt.split('-')]
        out = np.zeros(sz, np.float32)
        for i in range(sz[0]):
            out[i,0] = seg_to_small_seg(label[i,0], size_thres, zratio)
    elif topt[0] == '4': # instance boundary mask
        _, bd_thres,do_bg = [int(x) for x in topt.split('-')]
        out = np.zeros(sz, np.float32)
        for i in range(sz[0]):
            out[i,0] = seg_to_instance_bd(label[i,0], bd_thres, do_bg)
    return out

def rebalance_binary_class(label, mask=None, alpha=1.0, return_factor=False):
    """Binary-class rebalancing."""
    # input: numpy tensor
    # weight for smaller class is 1, the bigger one is at most 100*alpha
    if mask is None:
        weight_factor = float(label.sum()) / np.prod(label.shape)
    else:
        weight_factor = float((label*mask).sum()) / mask.sum()
    weight_factor = np.clip(weight_factor, a_min=1e-2, a_max=0.99)

    if weight_factor>0.5:
        weight = label + alpha*weight_factor/(1-weight_factor)*(1-label)
    else:
        weight = alpha*(1-weight_factor)/weight_factor*label + (1-label)

    if mask is not None:
        weight = weight*mask

    if return_factor: 
        return weight_factor, weight
    else:
        return weight


def rebalance_binary_class_torch(label, mask=None, alpha=1.0, return_factor=False):
    """Binary-class rebalancing."""
    # input: torch tensor
    # weight for smaller class is 1
    if mask is None:
        weight_factor = label.float().sum() / torch.prod(torch.tensor(label.size()).float())
    else:
        weight_factor = (label*mask).float().sum() / mask.float().sum()
    weight_factor = torch.clamp(weight_factor, min=1e-2, max=0.99)
    if weight_factor>0.5:
        weight = label + alpha*weight_factor/(1-weight_factor)*(1-label)
    else:
        weight = alpha*(1-weight_factor)/weight_factor*label + (1-label)

    if mask is not None:
        weight = weight*mask

    if return_factor: 
        return weight_factor, weight
    else:
        return weight
