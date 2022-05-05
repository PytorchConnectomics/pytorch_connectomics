import torch
import scipy
import numpy as np

def seg2diffgrads(label: np.ndarray) -> np.array:
    # input: (y, x) for 2D data or (z, y, x) with z>1 for 3D data
    # output: (2, y, x) for 2D data & (2, z, y, x) for 3D data (channel first)
    masks = label.squeeze().astype(np.int32)

    if masks.ndim==3:
        z, y, x = masks.shape
        mu = np.zeros((2, z, y, x), np.float32)
        for z in range(z):
            mu0 = masks2flows(masks[z])[0]
            mu[:, z] = mu0 
        flows = mu.astype(np.float32)
    elif masks.ndim==2:
        mu, _, _ = masks2flows(masks)
        flows = mu.astype(np.float32)
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
    mu0 = np.zeros((2, h, w))
    mu_c = np.zeros_like(mu0)
    centers = np.zeros((masks.max(), 2), 'int')

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
    if(len(ext)==0):
        return mu0, mu_c, centers

    n_iter = 2 * (ext.sum(axis=1)).max()

    # run diffusion
    mu = extend_centers(neighbors, centers, isneighbor, h+2, w+2, n_iter=n_iter)

    # normalize
    mu /= (1e-20 + (mu**2).sum(axis=0)**0.5)

    # put into original image
    mu0[:, y-1, x-1] = mu
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


def normalize_to_range(X, lower=0.01, upper=0.9999):
    """ normalize image so 0.0 is 0.01st percentile and 1.0 is 99.99th percentile """
    x01, x99 = torch.quantile(X, 0.01),torch.quantile(X, 0.99)
    return (X - x01) / (x99 - x01)


# sinebow color
def dx_to_circ(flows, alpha=False, mask=None, return_nparr = False):
    """ Y & X flows to 'optic' flow representation. This function adapted from
    https://github.com/MouseLand/cellpose.
    Args: 
        flows : n x 2 x Ly x Lx array of flow field components [dy,dx]
        alpha: bool, magnitude of flow controls opacity, not lightness (clear background)
        mask: 2D array multiplied to each RGB component to suppress noise
    """
    if isinstance(flows,(np.ndarray,np.generic)):
        flows = torch.from_numpy(flows)
        return_nparr = True
    
    if flows.ndim == 3 and flows.shape[0] == 2:
        flows = torch.unsqueeze(flows, 0)
    
    assert flows.ndim == 4, "Expected flows to be of shape (n,2,y,x)"

    imgs = []
    for flow in flows:
        magnitude = torch.clip(normalize_to_range(torch.sqrt(torch.sum(flow**2,axis=0))), 0, 1.)
        angles = torch.atan2(flow[1], flow[0])+torch.pi
        a = 2
        r = ((torch.cos(angles)+1)/a)
        g = ((torch.cos(angles+2*torch.pi/3)+1)/a)
        b = ((torch.cos(angles+4*torch.pi/3)+1)/a)

        if alpha:
            img = torch.stack((r,g,b,magnitude),axis=-1)
        else:
            img = torch.stack((r*magnitude,g*magnitude,b*magnitude),axis=-1)

        if mask is not None and alpha and flow.shape[0]<3:
            img[:,:,-1] *= mask

        img = (torch.clip(img, 0, 1) * 255).to(torch.uint8)
        imgs.append(img)

    vis = torch.permute(torch.stack(imgs),(0,3,1,2))
    return vis.numpy() if return_nparr else vis
