import h5py
import numpy as np
fl = h5py.File('test.h5','w')
fl.create_dataset('part1', data=np.zeros((100))