import numpy as np
import sys,os
# !import code; code.interact(local=vars())

def label_chunk(seg0, fns, numC, rr=1, rm_sz=0, m_type=np.uint64):
    # label chunks or slices
    from skimage.measure import label
    sz = fns(seg0, 0).shape
    numD = len(sz)
    
    mid = 0
    seg = [None]*numC
    for zi in range(numC):
        print('%d/%d [%d], '%(zi,numC,mid)),
        sys.stdout.flush()
        seg_c = label(fns(seg0, zi)>0).astype(m_type)
        if numD==2:
            seg_c = seg_c[np.newaxis]
        if rm_sz>0:
            seg_c = remove_small(seg_c, rm_sz)
            seg_c = seg_c[:,::rr,::rr]
            # preserve continuous id
            seg_c = relabel(seg_c).astype(m_type)

        if zi == 0: # first seg, relabel seg index        
            seg[zi] = seg_c
            mid += seg[zi].max()
            rlA = np.arange(mid+1,dtype=m_type)
        else: # link to previous slice
            slice_b = seg[zi-1][-1]
            slice_t = seg_c[0]            
            slices = label(np.stack([slice_b>0, slice_t>0],axis=0)).astype(m_type)

            # create mapping for seg cur
            lc = np.unique(seg_c);lc=lc[lc>0]
            rl_c = np.zeros(lc.max()+1, dtype=int)
            # merge curr seg
            # for 1 pre seg id -> slices id -> cur seg ids
            l0_p = np.unique(slice_b*(slices[0]>0))
            for l in l0_p:
                sid = np.unique(slices[0]*(slice_b==l))
                sid = sid[sid>0]
                cid = np.unique(slice_t*np.in1d(slices[1].reshape(-1),sid).reshape(sz[-2:]))
                rl_c[cid[cid>0]] = l
            
            # new id
            new_num = np.where(rl_c==0)[0][1:] # except the first one
            new_id = np.arange(mid+1,mid+1+len(new_num),dtype=m_type)
            rl_c[new_num] = new_id            
            seg[zi] = rl_c[seg_c]
            mid += len(new_num)
            
            # update global id
            rlA = np.hstack([rlA,new_id])
            # merge prev seg
            # for 1 cur seg id -> slices id -> prev seg ids
            l1_c = np.unique(slice_t*(slices[1]>0))
            for l in l1_c:
                sid = np.unique(slices[1]*(slice_t==l))
                sid = sid[sid>0]
                pid = np.unique(slice_b*np.in1d(slices[0].reshape(-1),sid).reshape(sz[-2:]))
                pid = pid[pid>0]
                # get all previous m-to-1 labels
                pid_p = np.where(np.in1d(rlA,rlA[pid]))[0]
                if len(pid_p)>1:
                    rlA[pid_p] = pid.max()
    return rlA[np.vstack(seg)]

def label_large(seg, chunk=[1,1,1]):
    # order: zyx
    from skimage.measure import label
    # for large chunk
    sz = seg.shape
    tsz = [sz[t]//chunk[t] for t in range(len(chunk))]
    # initial result
    mid = 0
    out = np.zeros(seg.shape, dtype=np.uint64)
    for zid in range(chunk[0]):
        for yid in range(chunk[1]):
            for xid in range(chunk[2]):
                print('label: ',zid,yid,xid,mid)
                tmp = label(seg[zid*tsz[0]:(zid+1)*tsz[0], \
                                yid*tsz[1]:(yid+1)*tsz[1], \
                                xid*tsz[2]:(xid+1)*tsz[2]]).astype(np.uint64)
                tmp[tmp>0] = tmp[tmp>0] + mid
                out[zid*tsz[0]:(zid+1)*tsz[0], \
                    yid*tsz[1]:(yid+1)*tsz[1], \
                    xid*tsz[2]:(xid+1)*tsz[2]] = tmp
                # assign big to small
                if xid>0:
                    print('\t check x-overlap')
                    tmp2 = label(seg[zid*tsz[0]:(zid+1)*tsz[0], \
                                yid*tsz[1]:(yid+1)*tsz[1], \
                                xid*tsz[2]-1:xid*tsz[2]+1]).astype(np.uint64)
                    bb = get_bb_label(tmp2)
                    import pdb; pdb.set_trace()
                    bb = bb[bb[:,0]>0] # remove 0
                    bid = bb[bb[:,6]-bb[:,5]==1,0]
                    if len(bid)>0:
                        print('\t',len(bid))
                        out=out.reshape(-1)
                        for bb in bid:
                            ii = np.unique(s1[tmp2==bb])
                            out[np.in1d(out,ii)] = ii.min()
                        out=out.reshape(seg.shape)
                if yid>0:
                    print('\t check y-overlap')
                    tmp2 = label(seg[zid*tsz[0]:(zid+1)*tsz[0], \
                                yid*tsz[1]-1:yid*tsz[1]+1, \
                                xid*tsz[2]:(xid+1)*tsz[2]]).astype(np.uint64)
                    bb = get_bb_label(tmp2)
                    bb = bb[bb[:,0]>0] # remove 0
                    bid = bb[bb[:,4]-bb[:,3]==1,0]
                    if len(bid)>0:
                        print('\t',len(bid))
                        out=out.reshape(-1)
                        for bb in bid:
                            ii = np.unique(s1[tmp2==bb])
                            out[np.in1d(out,ii)] = ii.min()
                        out=out.reshape(seg.shape)
                if zid>0:
                    print('\t check z-overlap')
                    tmp2 = label(seg[zid*tsz[0]-1:zid*tsz[0]+1, \
                                yid*tsz[1]:(yid+1)*tsz[1], \
                                xid*tsz[2]:(xid+1)*tsz[2]]).astype(np.uint64)
                    bb = get_bb_label(tmp2)
                    bid = bb[bb[:,2]-bb[:,1]==1,0]
                    if len(bid)>0:
                        print('\t',len(bid))
                        out=out.reshape(-1)
                        for bb in bid:
                            ii = np.unique(s1[tmp2==bb])
                            out[np.in1d(out,ii)] = ii.min()
                        out=out.reshape(seg.shape)
                mid = out[zid*tsz[0]:(zid+1)*tsz[0], \
                          yid*tsz[1]:(yid+1)*tsz[1], \
                          xid*tsz[2]:(xid+1)*tsz[2]].max()
    # check overlap
    return relabel(out, do_type=True)

def removeSeg(seg, did):
    sz = seg.shape
    seg = seg.reshape(-1)
    seg[np.in1d(seg,did)] = 0
    seg = seg.reshape(sz)

def listDiff(l1, l2):
    return sorted(list(set(l1)-set(l2)))

def remove_small(seg, thres=100,bid=None):
    if bid is None:
        uid, uc = np.unique(seg, return_counts=True)
        bid = uid[uc<thres]
    if len(bid)>0:
        sz = seg.shape
        seg = seg.reshape(-1)
        seg[np.in1d(seg,bid)] = 0
        seg = seg.reshape(sz)
    return seg

def seg2Vast(seg):
    # convert to 24 bits
    return np.stack([seg//65536, seg//256, seg%256],axis=2).astype(np.uint8)

def vast2Seg(seg):
    # convert to 24 bits
    return seg[:,:,0].astype(np.uint16)*65536+seg[:,:,1].astype(np.uint16)*256+seg[:,:,2].astype(np.uint16)

def labelSeg(seg): # do label for each seg-id
    from skimage.measure import label
    sid = np.unique(seg)
    sid = sid[sid>0]
    out = np.zeros(seg.shape, np.uint32)
    mid = 1
    for si in sid:
        tmp = label(seg==si) 
        out[tmp>0] = tmp[tmp>0]+mid
        mid += tmp.max()
    # convert to 24 bits
    return out.astype(getSegType(out.max()))

def getSegType(mid):
    m_type = np.uint64
    if mid<2**8:
        m_type = np.uint8
    elif mid<2**16:
        m_type = np.uint16
    elif mid<2**32:
        m_type = np.uint32
    return m_type

def relabel(seg, uid=None,nid=None,do_sort=False,do_type=False):
    if do_sort:
        uid,_ = seg2Count(seg,do_sort=True)
    else:
        # get the unique labels
        if uid is None:
            uid = np.unique(seg)
    uid = uid[uid>0] # leave 0 as 0, the background seg-id
    # get the maximum label for the segment
    mid = int(uid.max()) + 1

    # create an array from original segment id to reduced id
    # format opt
    m_type = seg.dtype
    if do_type:
        mid2 = len(uid) if nid is None else nid.max()+1
        m_type = getSegType(mid2)

    mapping = np.zeros(mid, dtype=m_type)
    if nid is None:
        mapping[uid] = np.arange(1,1+len(uid), dtype=m_type)
    else:
        mapping[uid] = nid.astype(m_type)
    # if uid is given, need to remove bigger seg id 
    seg[seg>=mid] = 0
    return mapping[seg]

def seg2Count(seg,do_sort=True,rm_zero=False):
    sm = seg.max()
    if sm==0:
        return None,None
    if sm>1:
        segIds,segCounts = np.unique(seg,return_counts=True)
        if rm_zero:
            segCounts = segCounts[segIds>0]
            segIds = segIds[segIds>0]
        if do_sort:
            sort_id = np.argsort(-segCounts)
            segIds=segIds[sort_id]
            segCounts=segCounts[sort_id]
    else:
        segIds=np.array([1])
        segCounts=np.array([np.count_nonzero(seg)])
    return segIds, segCounts

def seg2Zavg(seg):
    segIds, segCounts = seg2Count(seg)
    zCount = np.zeros(max(segIds)+1)
    for z in range(seg.shape[0]): 
        zCount[np.unique(seg[z])] += 1
    segAvg = segCounts/zCount[segIds]
    return segIds, segAvg

def seg2largest(seg):
    from skimage.measure import label
    seg = label(seg)
    if seg.max()>1:
        segIds,segCounts = np.unique(seg,return_counts=True)
        segIds=segIds[1:];segCounts=segCounts[1:]
        seg = (seg==segIds[np.argmax(segCounts)]).astype(np.uint8)
    return seg

def folderV2Seg(Do,dt=np.uint16):
    from scipy.misc import imread
    import glob
    fns = sorted(glob.glob(Do+'*.png'))
    sz = imread(fns[0]).shape
    seg = np.zeros((len(fns),sz[0],sz[1]), dtype=dt)
    for zi in range(len(fns)):
        seg[zi] = vast2Seg(imread(fns[zi]))
    return seg

# Columns: Nr  flags  red1 green1 blue1 pattern1  red2 green2 blue2 pattern2  anchorx anchory anchorz  parentnr childnr prevnr nextnr   collapsednr   bboxx1 bboxy1 bboxz1 bboxx2 bboxy2 bboxz2   "name"
def readVastSeg(fn):
    a= open(fn).readlines()
    # remove comments
    st_id = 0
    while a[st_id][0] in ['%','\\']:
        st_id+=1
    # remove segment name
    out = np.zeros((len(a)-st_id-1,24), dtype=int)
    name = [None]*(len(a)-st_id-1)
    for i in range(st_id+1,len(a)):
        out[i-st_id-1] = np.array([int(x) for x in a[i][:a[i].find('"')].split(' ') if len(x)>0])
        name[i-st_id-1] = a[i][a[i].find('"')+1:a[i].rfind('"')]
    return out, name

def writeVastAnchor(fn,bb):
    # plain structure
    # x0,y0,z0,x1,y1,z1
    oo = open(fn,'w')
    vast_str0='0   0   0 0 0 0   0 0 0 0   -1 -1 -1  0 0 0 1   0   -1 -1 -1 -1 -1 -1   "Background"\n'
    oo.write(vast_str0)

    vast_str='%d   0   255 0 0 0   255 0 0 0   %d %d %d  0 0 %d %d %d   %d %d %d %d %d %d "seg%d"\n'
    for i in range(bb.shape[0]):
        nn = i+2 if i!=bb.shape[0]-1 else 0
        oo.write(vast_str % (i+1, (bb[i,0]+bb[i,3])//2, (bb[i,1]+bb[i,4])//2, (bb[i,2]+bb[i,5])//2, \
                             i,nn,i+1,\
                             bb[i,0],bb[i,1],bb[i,2],
                             bb[i,3],bb[i,4],bb[i,5],
                             i+1))
    oo.close()

def get_spaced_colors(n):
    max_value = 16581375 #255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]
    
    return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]

def writeVastAnchorTree(fn,bbs,nn=['good','bad'], pref='seg'):
    # x0,y0,z0,x1,y1,z1
    oo = open(fn,'w')
    vast_str0='0   0   0 0 0 0   0 0 0 0   -1 -1 -1  0 0 0 1   0   -1 -1 -1 -1 -1 -1   "Background"\n'
    oo.write(vast_str0)

    vast_str='%d   0   255 0 0 0   255 0 0 0   %d %d %d  %d 0 %d %d %d   %d %d %d %d %d %d "%s%d"\n'
    sid = len(nn)+1
    cid = [None] * len(bbs)
    for bid in range(len(bbs)):
        bb = bbs[bid]
        if bb is not None:
            for i in range(bb.shape[0]):
                prevn = sid-1 if i!=0 else 0
                nextn = sid+1 if i!=bb.shape[0]-1 else 0
                oo.write(vast_str % (sid, (bb[i,0]+bb[i,3])//2, (bb[i,1]+bb[i,4])//2, (bb[i,2]+bb[i,5])//2, \
                                     bid+1,prevn,nextn,sid,\
                                     bb[i,0],bb[i,1],bb[i,2],
                                     bb[i,3],bb[i,4],bb[i,5],
                                     pref, sid))
                if i == 0:
                    cid[bid] = sid
                sid += 1

    ccs = get_spaced_colors(len(nn))
    for nid,n in enumerate(nn):
        prevn = nid-1 if nid!=0 else 0
        nextn = nid+1 if nid!=len(nn)-1 else 0
        vast_strF = '%d   0   %d %d %d %d   %d %d %d %d   -1 -1 -1  0 %d %d %d   %d   -1 -1 -1 -1 -1 -1   "%s"\n'\
                %(nid+1,\
                  ccs[nid][0],ccs[nid][1],ccs[nid][2],nid+1,\
                  ccs[nid][0],ccs[nid][1],ccs[nid][2],nid+1,\
                  cid[nid],prevn,nextn,nid+1,n)
        oo.write(vast_strF)

    oo.close()

def bfly_cv(bfly_db, x0, x1, y0, y1, z0, z1, tile_sz, dt=np.uint8,st=1, tile_ratio=1, resize_order=1):
    import cv2
    # x: column
    # y: row
    # no padding at the boundary
    # st: starting index 0 or 1
    result = np.zeros((z1-z0, y1-y0, x1-x0), dt)
    c0 = x0 // tile_sz # floor
    c1 = (x1 + tile_sz-1) // tile_sz # ceil
    r0 = y0 // tile_sz
    r1 = (y1 + tile_sz-1) // tile_sz
    for z in range(z0, z1):
        pattern = bfly_db["sections"][z]
        for row in range(r0, r1):
            for column in range(c0, c1):
                path = pattern.format(row=row+st, column=column+st)
                if not os.path.exists(path): 
                    #return None
                    patch = 128*np.ones((tile_sz,tile_sz),dtype=np.uint8)
                else:
                    if path[-3:]=='tif':
                        import tifffile
                        patch = tifffile.imread(path)
                    else:
                        patch = cv2.imread(path, 0)
                if tile_ratio != 1:
                    # scipy.misc.imresize: only do uint8
                    from scipy.ndimage import zoom
                    patch = zoom(patch, tile_ratio, order=resize_order)

                xp0 = column * tile_sz
                xp1 = (column+1) * tile_sz
                yp0 = row * tile_sz
                yp1 = (row + 1) * tile_sz
                if patch is not None:
                    x0a = max(x0, xp0)
                    x1a = min(x1, xp1)
                    y0a = max(y0, yp0)
                    y1a = min(y1, yp1)
                    result[z-z0, y0a-y0:y1a-y0, x0a-x0:x1a-x0] = patch[y0a-yp0:y1a-yp0, x0a-xp0:x1a-xp0]
    return result
def bfly(bfly_db, x0, x1, y0, y1, z0, z1, tile_sz, dt=np.uint8,st=1, tile_ratio=1, resize_order=1):
    # x: column
    # y: row
    # no padding at the boundary
    # st: starting index 0 or 1
    result = np.zeros((z1-z0, y1-y0, x1-x0), dt)
    c0 = x0 // tile_sz # floor
    c1 = (x1 + tile_sz-1) // tile_sz # ceil
    r0 = y0 // tile_sz
    r1 = (y1 + tile_sz-1) // tile_sz
    for z in range(z0, z1):
        pattern = bfly_db["sections"][z]
        for row in range(r0, r1):
            for column in range(c0, c1):
                path = pattern.format(row=row+st, column=column+st)
                if not os.path.exists(path): 
                    #return None
                    patch = 128*np.ones((tile_sz,tile_sz),dtype=np.uint8)
                else:
                    if path[-3:]=='tif':
                        import tifffile
                        patch = tifffile.imread(path)
                    else:
                        import scipy.misc
                        patch = scipy.misc.imread(path, 'L')
                if tile_ratio != 1:
                    # scipy.misc.imresize: only do uint8
                    from scipy.ndimage import zoom
                    patch = zoom(patch, tile_ratio, order=resize_order)

                xp0 = column * tile_sz
                xp1 = (column+1) * tile_sz
                yp0 = row * tile_sz
                yp1 = (row + 1) * tile_sz
                if patch is not None:
                    x0a = max(x0, xp0)
                    x1a = min(x1, xp1)
                    y0a = max(y0, yp0)
                    y1a = min(y1, yp1)
                    result[z-z0, y0a-y0:y1a-y0, x0a-x0:x1a-x0] = patch[y0a-yp0:y1a-yp0, x0a-xp0:x1a-xp0]
    return result
def U_mkdir(fn):
    if not os.path.exists(fn):
        os.mkdir(fn)

# get one slice
def bfly_z(imZ, x0, x1, y0, y1, z0, z1, tile_sz, dt=np.uint8,st=1):
    import tifffile
    # no padding at the boundary
    result = np.zeros((z1-z0, y1-y0, x1-x0), dt)
    c0 = x0 // tile_sz # floor
    c1 = (x1 + tile_sz-1) // tile_sz # ceil
    r0 = y0 // tile_sz
    r1 = (y1 + tile_sz-1) // tile_sz
    for z in range(z0, z1):
        pattern = imZ % z
        for row in range(r0, r1):
            for column in range(c0, c1):
                path = pattern.format(row=row+st, column=column+st)
                if not os.path.exists(path): 
                    return None
                else:
                    if path[-3:]=='tif':
                        patch = tifffile.imread(path)
                    else:
                        patch = scipy.misc.imread(path, 0)
                xp0 = column * tile_sz
                xp1 = (column+1) * tile_sz
                yp0 = row * tile_sz
                yp1 = (row + 1) * tile_sz
                if patch is not None:
                    x0a = max(x0, xp0)
                    x1a = min(x1, xp1)
                    y0a = max(y0, yp0)
                    y1a = min(y1, yp1)
                    result[z-z0, y0a-y0:y1a-y0, x0a-x0:x1a-x0] = patch[y0a-yp0:y1a-yp0, x0a-xp0:x1a-xp0]
    return result

# get one bbox
def bfly_bbox(ff, x0, x1, y0, y1, z0, z1, tile_sz, dim4=-1,dt=np.uint8):
    result = np.zeros((z1-z0, y1-y0, x1-x0), dt)
    c0 = x0 // tile_sz[2] # floor
    c1 = (x1 + tile_sz[2]-1) // tile_sz[2] # ceil
    r0 = y0 // tile_sz[1]
    r1 = (y1 + tile_sz[1]-1) // tile_sz[1]
    d0 = z0 // tile_sz[0]
    d1 = (z1 + tile_sz[0]-1) // tile_sz[0]
    #print 'bfly: ',d0,d1,r0,r1,c0,c1
    for depth in range(d0, d1):
        for row in range(r0, r1):
            for column in range(c0, c1):
                patch = ff[depth][row][column]
                xp0 = column * tile_sz[2]
                xp1 = (column+1) * tile_sz[2]
                yp0 = row * tile_sz[1]
                yp1 = (row + 1) * tile_sz[1]
                zp0 = depth * tile_sz[0]
                zp1 = (depth + 1) * tile_sz[0]
                if patch is not None:
                    x0a = max(x0, xp0)
                    x1a = min(x1, xp1)
                    y0a = max(y0, yp0)
                    y1a = min(y1, yp1)
                    z0a = max(z0, zp0)
                    z1a = min(z1, zp1)
                    if dim4==-1:
                        result[z0a-z0:z1a-z0, y0a-y0:y1a-y0, x0a-x0:x1a-x0] = \
                                np.array(patch[z0a-zp0:z1a-zp0, y0a-yp0:y1a-yp0, x0a-xp0:x1a-xp0])
                    else:
                        result[z0a-z0:z1a-z0, y0a-y0:y1a-y0, x0a-x0:x1a-x0] = \
                                np.array(patch[dim4,z0a-zp0:z1a-zp0, y0a-yp0:y1a-yp0, x0a-xp0:x1a-xp0])
    return result


def readtxt(filename):
    a= open(filename)
    content = a.readlines()
    a.close()
    return content

def writetxt(filename, content):
    a= open(filename,'w')
    if isinstance(content, (list,)):
        for ll in content:
            a.write(ll)
            if '\n' not in ll:
                a.write('\n')
    else:
        a.write(content)
    a.close()

def write_bfly(sz, numT, imN, tsz=1024, zPad=[0,0], im_id=None, outName=None,st=1):
    # one tile for each section
    dim={'depth':sz[0]+sum(zPad), 'height':sz[1], 'width':sz[2],
         'dtype':'uint8', 'n_columns':numT[1], 'n_rows':numT[0], "tile_size":tsz}
    # st: starting index
    if im_id is None:
        im_id = range(zPad[0]+st,st,-1)+range(st,sz[0]+st)+range(sz[0]-2+st,sz[0]-zPad[1]-2+st,-1)
    else: # st=0
        if zPad[0]>0:
            im_id = [im_id[x] for x in range(zPad[0],0,-1)]+im_id
        if zPad[1]>0:
            im_id += [im_id[x] for x in range(sz[0]-2,sz[0]-zPad[1]-2,-1)]
    sec=[imN(x) for x in im_id]
    out={'sections':sec, 'dimensions':dim}
    if outName is None:
        return out
    else:
        import json
        with open(outName,'w') as fid:
            json.dump(out, fid)

def doCLAHE(im, clahe=None, clip_limit=2.0, tileGridSize=(8,8)):
    import cv2
    # create a CLAHE object (Arguments are optional).
    if clahe is None:
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(im)

def readh5(filename, datasetname='main'):
    import h5py
    return np.array(h5py.File(filename,'r')[datasetname])

def writeh5(filename, dtarray, datasetname='main'):
    import h5py
    fid=h5py.File(filename,'w')
    if isinstance(datasetname, (list,)):
        for i,dd in enumerate(datasetname):
            ds = fid.create_dataset(dd, dtarray[i].shape, compression="gzip", dtype=dtarray[i].dtype)
            ds[:] = dtarray[i]
    else:
        ds = fid.create_dataset(datasetname, dtarray.shape, compression="gzip", dtype=dtarray.dtype)
        ds[:] = dtarray
    fid.close()

def get_angle2D(v1,v2):
    # two np array
    dot = np.sum(v1*v2)
    det = v1[0]*v2[1]-v2[0]*v1[1]
    return np.arctan2(det,dot)

def rotateIm(image, angle, center=None, scale=1.0):
    import cv2
    # grab the dimensions of the image
    (h, w) = image.shape[:2]

    # if the center is None, initialize it as the center of
    # the image
    if center is None:
        center = (w // 2, h // 2)

    # perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # return the rotated image
    return rotated
def arr2seg(ind):
    # ind: unique slice id
    dd = np.where(ind[1:]-ind[:-1] != 1)[0]
    out = np.zeros((len(dd)+1, 2),dtype=int)
    out[0] = [ind[0], ind[dd[0]]]
    for i in range(len(dd)-1):
        out[i+1] = [ind[dd[i]+1], ind[dd[i+1]]]
    out[len(dd)] = [ind[np.where(ind>ind[dd[-1]])[0][0]], ind[-1]]
    return out


# bbox utility
def get_bbs(data, thres, chunk=[1,1,1]):
    rr=[range(0,data.shape[x],np.ceil(data.shape[x]/float(chunk[x]))) for x in range(3)]
    for x in range(3):
        if rr[x][-1] != data.shape[x]:
            rr[x] += [data.shape[x]]

    # bbox 
    bb = [[[None for z in range(chunk[0])] for y in range(chunk[1])] for x in range(chunk[2])]
    # slice 
    sxy = [[[None for z in range(chunk[0])] for y in range(chunk[1]-1)] for x in range(chunk[2]-1)]
    # initial bbox
    for zi,zz in enumerate(rr[0][:-1]):
        for yi,yy in enumerate(rr[1][:-1]):
            for xi,xx in enumerate(rr[2][:-1]):
                mm = label(np.array(data[rr[0][zi]:rr[0][zi+1],rr[1][yi]:rr[1][yi+1],rr[2][xi]:rr[2][xi+1]])>thres)
                uid = np.unique(mm)
                uid = uid[uid>0]
                # check bbox
                bb[zi][yi][xi] = np.zeros((len(uid),7), np.uint16)
                bb[zi][yi][xi] = np.zeros((len(uid),7), np.uint16)
                for i in range(len(uid)):
                    bb[zi][yi][xi][i] = get_bb(mm==uid[i], True)

def list_create(chunk):
    if len(chunk)==1:
        out = [None for zi in range(chunk[0])]
    elif len(chunk)==2:
        out = [[None for yi in range(chunk[1])] for zi in range(chunk[0])]
    elif len(chunk)==3:
        out = [[[None for xi in range(chunk[2])] for yi in range(chunk[1])] for zi in range(chunk[0])]
    return out

def bbox_load(fn,delim=' ',dtype=int):
    bb = np.loadtxt(fn,delimiter=delim)
    if bb.ndim==1:
        bb=bb.reshape((1,len(bb)))
    if len(bb)>0:
        bb=bb.astype(dtype)
    return bb


def bbox_loadM(chunk,rr,fn, bbN=None, delim=' ',dtype=int, do_xy=True):
    numC = len(chunk)
    if not isinstance(chunk[0], (list,)): 
        # create chunk by number
        bb = list_create(chunk)
        chunk = [range(chunk[x]) for x in range(numC)]
    else:
        bb = list_create([chunk[x][-1]+1 for x in range(numC)])

    if numC == 3:
        # load 3D
        for xi in chunk[2]:
            for yi in chunk[1]:
                for zi in chunk[0]:
                    if do_xy:
                        if bbN is None:
                            tmp = bbox_load(fn%(zi,xi,yi),delim,dtype) 
                        else:
                            tmp = bbox_load(fn%(bbN[0][zi],bbN[2][xi],bbN[1][yi]),delim,dtype) 
                    else:
                        if bbN is None:
                            tmp = bbox_load(fn%(zi,yi,xi),delim,dtype) 
                        else:
                            tmp = bbox_load(fn%(bbN[0][zi],bbN[1][yi],bbN[0][xi]),delim,dtype) 
                    if len(tmp)==0:
                        continue
                    if rr is not None:
                        zo = rr[0][zi]
                        yo = rr[1][yi]
                        xo = rr[2][xi]
                        tmp += np.array([zo,zo,yo,yo,xo,xo]+[0]*(tmp.shape[1]-6))
                    bb[zi][yi][xi] = tmp
    elif numC == 2:
        # load 2D
        for xi in chunk[1]:
            for yi in chunk[0]:
                #print(yi,xi,len(bb),len(bb[0]))
                if do_xy:
                    if bbN is None:
                        tmp = bbox_load(fn%(xi,yi),delim,dtype) 
                    else:
                        tmp = bbox_load(fn%(bbN[1][xi],bbN[0][yi]),delim,dtype) 
                else:
                    if bbN is None:
                        tmp = bbox_load(fn%(yi,xi),delim,dtype) 
                    else:
                        tmp = bbox_load(fn%(bbN[0][yi],bbN[1][xi]),delim,dtype) 
                if len(tmp)==0:
                    continue
                if rr is not None:
                    yo = rr[0][yi]
                    xo = rr[1][xi]
                    if len(tmp.reshape(-1))>0:
                        tmp += np.array([0,0,yo,yo,xo,xo]+[0]*(tmp.shape[1]-6))
                bb[yi][xi] = tmp
    return bb

def bbox_concate(bb):
    if not isinstance(bb[0], (list,)):
        # 1D list
        out=np.zeros((0,bb[0].shape[1]),dtype=bb[0].dtype)
        for xx in bb:
            if xx.shape[1]>0:
                out=np.vstack([out,xx])
    else:
        if not isinstance(bb[0][0], (list,)):
            # 2D list
            out=np.zeros((0,bb[0][0].shape[1]),dtype=bb[0][0].dtype)
            for xx in bb:
                for yy in xx:
                    if yy.shape[1]>0:
                        out=np.vstack([out,yy])
    return out

def bbox_link(bb_l,bb_r,ax_l,ax_r,ax_m,tt_l,tt_r):
    # bbox in the same coord
    # link bb_l/bb_r by ax_l/ax_r dim with threshold value t1/t2
    if min(len(bb_l),len(bb_r))==0:
        return bb_l,bb_r
    b1 = np.where(bb_l[:,ax_l]==tt_l)[0]
    b2 = np.where(bb_r[:,ax_r]==tt_r)[0] 
    if min(len(b1),len(b2))==0:
        return bb_l,bb_r

    # coord
    ax_u = np.array(sorted([ax_l,ax_r]+list(ax_m)))
    # val
    ax_v = np.array(list(set(range(bb_l.shape[1]))-set(ax_u)),dtype=ax_u.dtype)

    for j in b1:
        sc = get_area(bb_l[j,ax_m],bb_r[b2][:,ax_m])
        if sc.max()>0: # there's a merge
            sid = b2[np.argmax(sc)]
            #print "in:",bb_l[j], bb_r[sid]
            bb_l[j,ax_u] = get_union(bb_l[j,ax_u], bb_r[sid,ax_u])
            bb_l[j,ax_v] = bb_l[j,ax_v]+bb_r[sid,ax_v]
            #print "out:",bb_l[j]
            #import pdb; pdb.set_trace()
            bb_r[sid,:] = -1
    return bb_l, bb_r[np.where(bb_r[:,0]>=0)[0]]

def get_bb_label(seg, do_count=False, uid=None):
    dim = len(seg.shape)
    if uid is None:
        uid = np.unique(seg)
        uid = uid[uid>0]
    out = np.zeros((len(uid),dim*2+1+do_count),dtype=np.uint32)
    #print('#bbox: ',len(uid))
    for i,j in enumerate(uid):
        out[i,0] = j
        a=np.where(seg==j)
        if len(a[0])>0:
            for k in range(dim):
                out[i,1+k*2:3+k*2] = [a[k].min(), a[k].max()]
            if do_count:
                out[i,-1] = len(a[0])
    return out

def get_bb(seg, do_count=False):
    dim = len(seg.shape)
    a=np.where(seg>0)
    if len(a[0])==0:
        return [-1]*dim*2
    out=[]
    for i in range(dim):
        out+=[a[i].min(), a[i].max()]
    if do_count:
        out+=[len(a[0])]
    return out

def get_area(a,b):
    # n*6
    # a: one box
    # b: multiple box
    #[xmin,xmax,ymin,ymax]
    if b.ndim==1:
        b=b.reshape(1,b.shape[0])
    dd = np.ones(b.shape[0])
    for i in range(len(a)//2):
        dd = dd*np.maximum(0,np.minimum(a[i*2+1], b[:,i*2+1]) - np.maximum(a[i*2], b[:,i*2]))
    return dd

def get_union(a,b):
    #[xmin,xmax,ymin,ymax]
    ll=len(a)
    out=[None]*ll
    for i in range(0,ll,2):
        out[i] = min(a[i],b[i])
    for i in range(1,ll,2):
        out[i] = max(a[i],b[i])
    return out

def get_intersect(a,b):
    #[xmin,xmax,ymin,ymax]
    ll=len(a)
    out=[None]*ll
    for i in range(0,ll,2):
        out[i] = max(a[i],b[i])
    for i in range(1,ll,2):
        out[i] = min(a[i],b[i])
    return out

def postprocess_mito(pred, sig=1.0, thres=64):
    # param for 8x8x30 nm
    from skimage.filters import gaussian
    if sig>0:
        pred = gaussian(pred, sigma=(sig,sig,sig), mode='reflect', preserve_range=True).astype(np.uint8)
    out = (pred > thres).astype(np.uint8)
    return out

def get_voc(pred, gt, thres=[0.5]):
    sc = [None for i in range(len(thres))]
    for tid,t in enumerate(thres):
        TP,FP,TN,FN = confusion_matrix(pred, gt, t)
        jaccard_foreground = float(TP)/(TP+FP+FN)
        jaccard_background = float(TN)/(TN+FP+FN)
        sc[tid] = (jaccard_foreground+jaccard_background)/2.
    return sc

def confusion_matrix(pred, gt, thres=0.5):
    TP = np.sum((gt==1) & (pred>thres))
    FP = np.sum((gt==0) & (pred>thres))
    TN = np.sum((gt==0) & (pred<=thres))
    FN = np.sum((gt==1) & (pred<=thres))
    return (TP, FP, TN, FN)

def get_iou(pred_b,gt_b,topk=-1):
    from skimage.measure import label
    seg = label(pred_b)
    gt = label(gt_b)
    ind,uc = seg2Count(gt,do_sort=True)
    if topk>=0:
        ind=ind[:topk]
    num = len(ind)
    iou = np.zeros(num) 
    msg = [None for i in range(num)]
    print(num)
    for i in range(num):
        tmp_id, tmp_cc = np.unique(seg[gt==ind[i]], return_counts=True)
        tmp_sid = np.argsort(-tmp_cc) # descend
        iou[i] = float(max(tmp_cc))/len(np.union1d(np.ravel_multi_index(np.where(gt==ind[i]),gt.shape), np.ravel_multi_index(np.where(seg==tmp_id[np.argmax(tmp_cc)]),seg.shape)))
        numV = float(sum(tmp_cc))
        msg[i] = '%d,%d,%.2f' %(ind[i],numV,iou[i])
        for j in range(min(5,len(tmp_cc))):
            msg[i]+=',%d,%.2f'%(tmp_id[tmp_sid[j]],tmp_cc[tmp_sid[j]]/numV)
    for j in np.argsort(iou):
        print(msg[j]) 

def do_filter(pred,ftype,fparam): # z-filter
    from skimage.filters import gaussian
    from scipy.ndimage import median_filter
    from scipy.ndimage import maximum_filter
    if ftype=='gaussian':
        out = gaussian(pred, sigma=fparam, mode='reflect', preserve_range=True).astype(np.uint8)
    elif ftype=='median': # fix in-z
        out = median_filter(pred, size=fparam, mode='reflect')                                       
    elif ftype=='max': # grow region
        out = maximum_filter(pred, size=fparam, mode='reflect')
    return out
