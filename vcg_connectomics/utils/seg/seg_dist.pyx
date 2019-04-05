cimport cython
cimport numpy as np

import ctypes
import numpy as np

cdef extern from 'cpp/seg_dist/cpp-distance.h':
    float *CppTwoDimensionalDistanceTransform(long *segmentation, long resolution[3])
    long *CppDilateData(long *data, long resolution[3], float distance)


# get the two dimensional distance transform
def TwoDimensionalDistanceTransform(segmentation):
    zres, yres, xres = segmentation.shape

    # convert numpy array to c++
    cdef np.ndarray[long, ndim=3, mode='c'] cpp_segmentation
    cpp_segmentation = np.ascontiguousarray(segmentation, dtype=ctypes.c_int64)

    cdef float *cpp_distances = CppTwoDimensionalDistanceTransform(&(cpp_segmentation[0,0,0]), [zres, yres, xres])
    cdef float[:] tmp_distances = <float[:segmentation.size]> cpp_distances

    return np.reshape(np.asarray(tmp_distances), (zres, yres, xres))


# dilate segments from boundaries
def DilateData(data, distance):
    zres, yres, xres = data.shape

    # convert numpy array to c++
    cdef np.ndarray[long, ndim=3, mode='c'] cpp_data
    cpp_data = np.ascontiguousarray(data, dtype=ctypes.c_int64)

    cdef long *cpp_dilated = CppDilateData(&(cpp_data[0,0,0]), [zres, yres, xres], float(distance))
    cdef long[:] tmp_dilated = <long[:data.size]> cpp_dilated

    return np.reshape(np.asarray(tmp_dilated), (zres, yres, xres))
