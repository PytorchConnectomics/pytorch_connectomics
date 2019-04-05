cimport cython
cimport numpy as np

import ctypes
from libc.stdint cimport uint64_t,int32_t
import numpy as np
import scipy.ndimage

from em_segLib.aff_util import affgraph_to_edgelist
from em_segLib.seg_util import mknhood3d

cdef extern from 'cpp/seg_core/cpp-seg2seg.h':
    long *CppMapLabels(long *segmentation, long *mapping, unsigned long nentries)
    long *CppRemoveSmallConnectedComponents(long *segmentation, int threshold, unsigned long nentries)
    long *CppForceConnectivity(long *segmentation, long zres, long yres, long xres)

cdef extern from 'cpp/seg_core/cpp-seg_core.h':
    void connected_components_cpp(const int nVert,
                   const int nEdge, const uint64_t* node1, const uint64_t* node2, const int* edgeWeight,
                   uint64_t* seg);
    void marker_watershed_cpp(const int nVert, const uint64_t* marker,
                   const int nEdge, const uint64_t* node1, const uint64_t* node2, const float* edgeWeight,
                   uint64_t* seg);

cdef extern from 'cpp/seg_core/cpp-seg2gold.h':
    long *CppMapping(long *segmentation, int *gold, long nentries, double low_threshold, double high_threshold)



## 1. seg2gold
def Mapping(segmentation, gold, low_threshold=0.10, high_threshold=0.80):
    cdef np.ndarray[long, ndim=3, mode='c'] cpp_segmentation
    cpp_segmentation = np.ascontiguousarray(segmentation, dtype=ctypes.c_int64)
    cdef np.ndarray[int, ndim=3, mode='c'] cpp_gold
    cpp_gold = np.ascontiguousarray(gold, dtype=ctypes.c_int32)

    max_segmentation = np.amax(segmentation) + 1

    cdef long *mapping = CppMapping(&(cpp_segmentation[0,0,0]), &(cpp_gold[0,0,0]), segmentation.size, low_threshold, high_threshold)

    cdef long[:] tmp_mapping = <long[:max_segmentation]> mapping;

    return np.asarray(tmp_mapping)

## 2. seg2seg
# map the labels from this segmentation
def MapLabels(segmentation, mapping):
    # get the size of the data
    zres, yres, xres = segmentation.shape
    nentries = segmentation.size

    cdef np.ndarray[long, ndim=3, mode='c'] cpp_segmentation
    cpp_segmentation = np.ascontiguousarray(segmentation, dtype=ctypes.c_int64)
    cdef np.ndarray[long, ndim=1, mode='c'] cpp_mapping
    cpp_mapping = np.ascontiguousarray(mapping, dtype=ctypes.c_int64)

    cdef long *mapped_segmentation = CppMapLabels(&(cpp_segmentation[0,0,0]), &(cpp_mapping[0]), nentries)

    cdef long[:] tmp_segmentation = <long[:segmentation.size]> mapped_segmentation

    return np.reshape(np.asarray(tmp_segmentation), (zres, yres, xres))



# remove the components less than min size
def RemoveSmallConnectedComponents(segmentation, threshold=64):
    if threshold == 0: return segmentation

    nentries = segmentation.size
    zres, yres, xres = segmentation.shape

    cdef np.ndarray[long, ndim=3, mode='c'] cpp_segmentation
    cpp_segmentation = np.ascontiguousarray(segmentation, dtype=ctypes.c_int64)
    
    # call the c++ function
    cdef long *updated_segmentation = CppRemoveSmallConnectedComponents(&(cpp_segmentation[0,0,0]), threshold, nentries)

    # turn into python numpy array
    cdef long[:] tmp_segmentation = <long[:segmentation.size]> updated_segmentation

    # reshape the array to the original shape
    thresholded_segmentation = np.reshape(np.asarray(tmp_segmentation), (zres, yres, xres))	
    return np.copy(thresholded_segmentation)



def ForceConnectivity(segmentation):
    # transform into c array
    cdef np.ndarray[long, ndim=3, mode='c'] cpp_segmentation
    cpp_segmentation = np.ascontiguousarray(segmentation, dtype=ctypes.c_int64)
    zres, yres, xres = segmentation.shape

    # call the c++ function
    cdef long *cpp_components = CppForceConnectivity(&(cpp_segmentation[0,0,0]), zres, yres, xres)

    # turn into python numpy array
    cdef long[:] tmp_components = <long[:zres*yres*xres]> cpp_components
    
    # reshape the array to the original shape
    components = np.reshape(np.asarray(tmp_components), (zres, yres, xres)).astype(np.int32)

    # find which segments have multiple components
    return components

def prune_and_renum(np.ndarray[uint64_t,ndim=1] seg,
                    int sizeThreshold=1):
    # renumber the components in descending order by size
    segId,segSizes = np.unique(seg, return_counts=True)
    descOrder = np.argsort(segSizes)[::-1]
    renum = np.zeros(int(segId.max()+1),dtype=np.uint64)
    segId = segId[descOrder]
    segSizes = segSizes[descOrder]
    renum[segId] = np.arange(1,len(segId)+1)

    if sizeThreshold>0:
        renum[segId[segSizes<=sizeThreshold]] = 0
        segSizes = segSizes[segSizes>sizeThreshold]

    seg = renum[seg]
    return (seg, segSizes)

def marker_watershed(np.ndarray[uint64_t,ndim=1] marker,
                     np.ndarray[uint64_t,ndim=1] node1,
                     np.ndarray[uint64_t,ndim=1] node2,
                     np.ndarray[float,ndim=1] edgeWeight,
                     int sizeThreshold=1):
    cdef int nVert = marker.shape[0]
    cdef int nEdge = node1.shape[0]
    marker = np.ascontiguousarray(marker)
    node1 = np.ascontiguousarray(node1)
    node2 = np.ascontiguousarray(node2)
    edgeWeight = np.ascontiguousarray(edgeWeight)
    cdef np.ndarray[uint64_t,ndim=1] seg = np.zeros(nVert,dtype=np.uint64)
    marker_watershed_cpp(nVert, &marker[0],
                         nEdge, &node1[0], &node2[0], &edgeWeight[0],
                         &seg[0]);
    (seg,segSizes) = prune_and_renum(seg,sizeThreshold)
    return (seg, segSizes)

def connected_components_affgraph(aff,nhood=mknhood3d()):
    (node1,node2,edge) = affgraph_to_edgelist(aff,nhood)
    (seg,segSizes) = connected_components(int(np.prod(aff.shape[1:])),node1,node2,edge)
    seg = seg.reshape(aff.shape[1:])
    return (seg,segSizes)


def connected_components(int nVert,
                         np.ndarray[uint64_t,ndim=1] node1,
                         np.ndarray[uint64_t,ndim=1] node2,
                         np.ndarray[int,ndim=1] edgeWeight,
                         int sizeThreshold=1):
    cdef int nEdge = node1.shape[0]
    node1 = np.ascontiguousarray(node1)
    node2 = np.ascontiguousarray(node2)
    edgeWeight = np.ascontiguousarray(edgeWeight)
    cdef np.ndarray[uint64_t,ndim=1] seg = np.zeros(nVert,dtype=np.uint64)
    connected_components_cpp(nVert,
                             nEdge, &node1[0], &node2[0], &edgeWeight[0],
                             &seg[0]);
    (seg,segSizes) = prune_and_renum(seg,sizeThreshold)
    return (seg, segSizes)
