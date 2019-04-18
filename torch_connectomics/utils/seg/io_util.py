import h5py
import numpy as np

def writeh5(filename, datasetname, dtarray):
    fid=h5py.File(filename,'w')
    ds = fid.create_dataset(datasetname, dtarray.shape, compression="gzip", dtype=dtarray.dtype)
    ds[:] = dtarray
    fid.close()

                                                                               
def readh5(filename, datasetname='main'):
    return np.array(h5py.File(filename,'r')[datasetname])
