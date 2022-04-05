import torch
import scipy
import numpy as np

def seg2diffgrads(label: np.ndarray) -> np.array:
    # input: (y, x) for 2D data or (z, y, x) with z>1 for 3D data
    # output: (3, y, x) for 2D data & (3, z, y, x) for 3D data (channel first)
    # 1st channel is Y flow, 2nd channel is X flow, 3rd channel is foreground map
    masks = label.squeeze().astype(np.int32)
    label = np.expand_dims(label, axis=0)

    if masks.ndim==3:
        z, y, x = masks.shape
        mu = np.zeros((2, z, y, x), np.float32)
        for z in range(z):
            mu0 = masks2flows(masks[z])[0]
            mu[:, z] = mu0 
        flows = np.concatenate([mu, label>0.5], axis=0).astype(np.float32)
    elif masks.ndim==2:
        mu, _, _ = masks2flows(masks)
        flows = np.concatenate([mu, label>0.5], axis=0).astype(np.float32)
    else:
        raise ValueError('expecting 2D or 3D labels but received %dD input!' % masks.ndim)

    return flows


def masks2flows(masks: np.ndarray):
    """Convert masks to flows using diffusion from center pixel. Center of masks is defined to be the 
    closest pixel to the median of all pixels that is inside the mask. Result of diffusion is converted 
    into flows by computing the gradients of the diffusion density map. This function is adapted from
    https://github.com/MouseLand/cellpose.
    """
    h, w = masks.shape
    masks_padded = np.pad(masks, 1, mode='constant', constant_values=0).astype(np.int64)

    # get mask pixel neighbors
    y, x = np.nonzero(masks_padded)
    neighborsY = np.stack((y, y-1, y+1, 
                           y, y, y-1, 
                           y-1, y+1, y+1), axis=0)
    neighborsX = np.stack((x, x, x, 
                           x-1, x+1, x-1, 
                           x+1, x-1, x+1), axis=0)
    neighbors = np.stack((neighborsY, neighborsX), axis=-1)

    # get mask centers
    slices = scipy.ndimage.find_objects(masks)
    centers = np.zeros((masks.max(), 2), 'int')
    for i, si in enumerate(slices):
        if si is None: # the object index does not exist
            continue

        sr, sc = si
        yi, xi = np.nonzero(masks[sr, sc] == (i+1))
        yi = yi.astype(np.int32) + 1 # add padding
        xi = xi.astype(np.int32) + 1 # add padding
        ymed = np.median(yi)
        xmed = np.median(xi)
        imin = np.argmin((xi-xmed)**2 + (yi-ymed)**2)
        xmed = xi[imin]
        ymed = yi[imin]
        centers[i,0] = ymed + sr.start 
        centers[i,1] = xmed + sc.start

    # get neighbor validator (not all neighbors are in same mask)
    neighbor_masks = masks_padded[neighbors[:,:,0], neighbors[:,:,1]]
    isneighbor = neighbor_masks == neighbor_masks[0]
    ext = []
    for slice_data in slices:
        if slice_data is not None:
            sr, sc = slice_data
            ext.append([sr.stop - sr.start + 1, sc.stop - sc.start + 1])
    ext = np.array(ext)
    n_iter = 2 * (ext.sum(axis=1)).max()

    # run diffusion
    mu = extend_centers(neighbors, centers, isneighbor, h+2, w+2, n_iter=n_iter)

    # normalize
    mu /= (1e-20 + (mu**2).sum(axis=0)**0.5)

    # put into original image
    mu0 = np.zeros((2, h, w))
    mu0[:, y-1, x-1] = mu
    mu_c = np.zeros_like(mu0)
    return mu0, mu_c, centers


def extend_centers(neighbors, centers, isneighbor, h, w, n_iter: int = 200):
    """Run diffusion to generate flows for label images. This function is 
    adapted from: https://github.com/MouseLand/cellpose.

    Args: 
        neighbors : 9 x pixels in masks 
        centers : mask centers
        isneighbor : valid neighbor boolean 9 x pixels
    """
    nimg = neighbors.shape[0] // 9
    pt = torch.from_numpy(neighbors)
    
    T = torch.zeros((nimg, h, w), dtype=torch.double)
    meds = torch.from_numpy(centers.astype(int)).long()
    isneigh = torch.from_numpy(isneighbor)

    with torch.no_grad():
        for _ in range(n_iter):
            T[:, meds[:,0], meds[:,1]] +=1
            Tneigh = T[:, pt[:,:,0], pt[:,:,1]]
            Tneigh *= isneigh
            T[:, pt[0,:,0], pt[0,:,1]] = Tneigh.mean(axis=1)
        
    T = torch.log(1.+ T)
    # gradient positions
    grads = T[:, pt[[2,1,4,3],:,0], pt[[2,1,4,3],:,1]]
    dy = grads[:,0] - grads[:,1]
    dx = grads[:,2] - grads[:,3]

    mu = np.stack((dy.cpu().squeeze(), dx.cpu().squeeze()), axis=-2)
    return mu


def normalize_to_range(X,lower=0.01,upper=99.99):
    """ normalize image so 0.0 is 0.01st percentile and 1.0 is 99.99th percentile """
    x01,x99 = np.percentile(X, 1),np.percentile(X, 99)
    return (X - x01) / (x99 - x01)


# sinebow color
def dx_to_circ(flows, alpha=False, mask=None):
    """ Y & X flows to 'optic' flow representation. This function adapted from
    https://github.com/MouseLand/cellpose.
    Args: 
        flows : n x 2 x Ly x Lx array of flow field components [dy,dx]
        alpha: bool, magnitude of flow controls opacity, not lightness (clear background)
        mask: 2D array multiplied to each RGB component to suppress noise
    """
    flows = np.array(flows)
    if flows.ndim == 3 and flows[0]==2:
        flows = np.expand_dims(flows,0)
    
    assert flows.ndim == 4, "Expected flows to be of shape (n,2,y,x)"

    imgs = []
    for flow in flows:
        magnitude = np.clip(normalize_to_range(np.sqrt(np.sum(flow**2,axis=0))), 0, 1.)
        angles = np.arctan2(flow[1], flow[0])+np.pi
        a = 2
        r = ((np.cos(angles)+1)/a)
        g = ((np.cos(angles+2*np.pi/3)+1)/a)
        b = ((np.cos(angles+4*np.pi/3)+1)/a)

        if alpha:
            img = np.stack((r,g,b,magnitude),axis=-1)
        else:
            img = np.stack((r*magnitude,g*magnitude,b*magnitude),axis=-1)

        if mask is not None and alpha and flow.shape[0]<3:
            img[:,:,-1] *= mask

        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        imgs.append(img)

    return np.transpose(np.array(imgs),(3,0,1,2))