import numpy as np

def flip(data, rule, modality='img', rot_pad=False, rot_st=None):
    # Kisuk's version: minor bug at the boundary
    """Flip data according to a specified rule.
    Args:
        data:   4D numpy array to be transformed.
        rule:   Transform rule, specified as a Boolean array.
                [z reflection, y reflection, x reflection, xy transpose]
    Returns:
        Transformed data.
    """
    if max(rule)>0:
        assert np.size(rule)==4 and data.ndim == 4
        # z reflection.
        if rule[0]:
            data = data[:,::-1,:,:]
        # y reflection.
        if rule[1]:
            data = data[:,:,::-1,:]
        # x reflection.
        if rule[2]:
            data = data[:,:,:,::-1]

        if rot_pad: # pad 1 to top-left during volume sampling 
            if modality=='aff':
                # dw: fix the image-aff alignment bug
                if rot_st is None: 
                    rot_st = np.ones((3,3), dtype=int)
                    for i in range(3):
                        rot_st[i,i] = 0 if rule[i]==1 else 1 
                data = crop(data, np.array(data.shape[1:])-np.array(rule[:3]), rot_st)
            else: # img/seg: crop the top-left corner
                # print np.array(data.shape[1:]),np.array(rule[:3])
                data = crop(data, np.array(data.shape[1:])-np.array(rule[:3]))

        # Transpose in xy.
        if rule[3]:
            data = data.transpose(0,1,3,2)
            if modality=='aff': # need to swap x,y affinity
                data[[2,1],...] = data[[1,2],...]
    return data

def revert_flip(data, rule, modality='seg'):
    assert np.size(rule)==4 and data.ndim == 4

    # Special treat for affinity.
    is_affinity = False if dst is None else True
    if modality=='aff':
        (dz,dy,dx) = dst
        assert data.shape[-4]==3
        assert dx and abs(dx) < data.shape[-1]
        assert dy and abs(dy) < data.shape[-2]
        assert dz and abs(dz) < data.shape[-3]

    # Transpose in xy.
    if rule[3]:
        data = data.transpose(0,1,3,2)
        # Swap x/y-affinity maps.
        if modality=='aff':
            data[[0,1],...] = data[[1,2],...]
    # x reflection
    if rule[2]:
        data = data[:,:,:,::-1]
        # Special treatment for x-affinity.
        if modality == 'aff':
            if dx > 0:
                data[2,:,:,dx:] = data[0,:,:,:-dx]
                data[2,:,:,:dx].fill(0)
            else:
                dx = abs(dx)
                data[2,:,:,:-dx] = data[0,:,:,dx:]
                data[2,:,:,-dx:].fill(0)
    # y reflection
    if rule[1]:
        data = data[:,:,::-1,:]
        # Special treatment for y-affinity.
        if modality == 'aff':
            if dy > 0:
                data[1,:,dy:,:] = data[1,:,:-dy,:]
                data[1,:,:dy,:].fill(0)
            else:
                dy = abs(dy)
                data[1,:,:-dy,:] = data[1,:,dy:,:]
                data[1,:,-dy:,:].fill(0)
    # z reflection
    if rule[0]:
        data = data[:,::-1,:,:]
        # Special treatment for z-affinity.
        if is_affinity:
            if dz > 0:
                data[0,dz:,:,:] = data[2,:-dz,:,:]
                data[0,:dz,:,:].fill(0)
            else:
                dz = abs(dz)
                data[0,:-dz,:,:] = data[2,dz:,:,:]
                data[0,-dz:,:,:].fill(0)
    return data

def crop(data, sz, st=np.array([0,0,0])): # C*D*W*H                                            
    if np.array(st).ndim == 1: # pointer to sub-tensor
        return data[:,st[0]:st[0]+sz[0], st[1]:st[1]+sz[1], \
                st[2]:st[2]+sz[2]]
    else: # need to create new tensor for channel wise crop
        out = np.zeros([st.shape[0]]+list(sz)).astype(data.dtype)
        #print('cc,',out.shape,data.shape,st,sz)
        for i in range(st.shape[0]):
            out[i] = data[i, st[i,0]:st[i,0]+sz[0],st[i,1]:st[i,1]+sz[1],\
                          st[i,2]:st[i,2]+sz[2]]
        return out
                                                                                                     
def cropPad(data, sz, st=np.zeros(3)): # C*D*W*H
    # within the range
    dsz = np.array(data.shape[1:])
    if st.min()>=0 and (st+sz-dsz)<=0:
        return data[:,st[0]:st[0]+sz[0], st[1]:st[1]+sz[1], \
                st[2]:st[2]+sz[2]]
    else: # out of the range
        ran = [None]*3
        for i in range(3):
            ran[i] = np.abs(np.arange(st[i],st[i]+sz[i])) # reflect negative
            bid = np.where(ran[i]>=dsz[i])[0]
            ran[i][bid] = 2*(dsz[i]-1)-ran[i][bid]
        return data[:,ran[0], ran[1], ran[2]]

def cropCentralN(img, label, offset=np.array([0,0,0])):
    # multiple datasets
    for i in range(len(img)):
        img[i], label[i] = cropCentral(img[i], label[i], offset)
    return img, label

def cropCentral(img, label, offset=np.array([0,0,0])):                                               
    # input size: img >= label+offset*2
    # output size: same for warp augmentation
    # data format: CxDxWxH
    img_sz = np.array(img.shape[1:])
    label_sz = np.array(label.shape[1:])
    sz_diff = img_sz-label_sz-2*offset
    sz_offset = offset + abs(sz_diff) // 2 # floor
    if any(sz_offset!=0):
        # z axis
        if sz_diff[1] < 0: # label is bigger
            label = label[:,sz_offset[0]:sz_offset[0]+img.shape[1]]
        else: # data is bigger
            img = img[:,sz_offset[0]:sz_offset[0]+img.shape[1]]
        # y axis
        if sz_diff[2] < 0:
            label=label[:,:,sz_offset[1]:sz_offset[1]+img.shape[2]]
        else:
            img = img[:,:,sz_offset[1]:sz_offset[1]+label.shape[2]]
        # x axis
        if sz_diff[3] < 0:
            label=label[:,:,:,sz_offset[2]:sz_offset[2]+img.shape[3]]
        else:
            img = img[:,:,:,sz_offset[2]:sz_offset[2]+label.shape[3]]
    return img, label
