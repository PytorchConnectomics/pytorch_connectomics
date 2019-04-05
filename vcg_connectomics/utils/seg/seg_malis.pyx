import numpy as np
cimport numpy as np
from libc.stdint cimport uint64_t,int32_t

cdef extern from "cpp/seg_malis/cpp-malis_core.h":
    void preCompute(const uint64_t* conn_dims, const int32_t* nhood_data, const uint64_t* nhood_dims,
        uint64_t* pre_ve, uint64_t* pre_prodDims, int32_t* pre_nHood);

    void malis_loss_weights_cpp_both(const uint64_t* segTrue,
               const uint64_t* conn_dims, const int32_t* nhood_data, const uint64_t* nhood_dims,
               const float* edgeWeight, float* nPairPerEdge,
               uint64_t* pre_ve, uint64_t* pre_prodDims, int32_t* pre_nHood, 
               const int pos, const float weight_opt);

    void malis_loss_weights_cpp(const uint64_t* segTrue,
               const uint64_t* conn_dims, const int32_t* nhood_data, const uint64_t* nhood_dims,
                   const float* edgeWeight,
                   const int pos,
                   float* nPairPerEdge);


def malis_init(np.ndarray[uint64_t, ndim=1] conn_dims,
                np.ndarray[int32_t, ndim=1] nhood_data,
               np.ndarray[uint64_t, ndim=1] nhood_dims):
    cdef np.ndarray[uint64_t, ndim=1] pre_ve = np.zeros(2,dtype=np.uint64)
    cdef np.ndarray[uint64_t, ndim=1] pre_prodDims = np.zeros(3,dtype=np.uint64)
    cdef np.ndarray[int32_t, ndim=1] pre_nHood = np.zeros(3,dtype=np.int32)
    preCompute(&conn_dims[0], &nhood_data[0], &nhood_dims[0], &pre_ve[0], &pre_prodDims[0], &pre_nHood[0])
    return pre_ve, pre_prodDims, pre_nHood

# combine positive and negative weight
def malis_loss_weights_both(np.ndarray[uint64_t, ndim=1] segTrue,
                np.ndarray[uint64_t, ndim=1] conn_dims,
                np.ndarray[int32_t, ndim=1] nhood_data,
                np.ndarray[uint64_t, ndim=1] nhood_dims,
                np.ndarray[uint64_t, ndim=1] pre_ve,
                np.ndarray[uint64_t, ndim=1] pre_prodDims,
                np.ndarray[int32_t, ndim=1] pre_nHood,
                np.ndarray[float, ndim=1] edgeWeight,
                np.ndarray[float, ndim=1] gtWeight,
                float weight_opt):
    segTrue = np.ascontiguousarray(segTrue)
    cdef np.ndarray[float, ndim=1] nPairPerEdge = np.zeros(edgeWeight.shape[0],dtype=np.float32)
    cdef np.ndarray[float, ndim=1] tmpWeight = np.ascontiguousarray(np.minimum(edgeWeight,gtWeight))
    # can't be done in one cpp call, as the MST is different based on weight
    # add positive weight
    malis_loss_weights_cpp_both(&segTrue[0],
                   &conn_dims[0], &nhood_data[0], &nhood_dims[0], &tmpWeight[0], &nPairPerEdge[0],
                   &pre_ve[0], &pre_prodDims[0], &pre_nHood[0], 1, weight_opt);
    # add negative weight
    if np.count_nonzero(np.unique(segTrue)) > 1: # at least two segments
        tmpWeight = np.ascontiguousarray(np.maximum(edgeWeight,gtWeight))
        malis_loss_weights_cpp_both(&segTrue[0],
                   &conn_dims[0], &nhood_data[0], &nhood_dims[0], &tmpWeight[0], &nPairPerEdge[0],
                   &pre_ve[0], &pre_prodDims[0], &pre_nHood[0], 0, weight_opt);

    return nPairPerEdge

# get either positive or negative weight
def malis_loss_weights(np.ndarray[uint64_t, ndim=1] segTrue,
                np.ndarray[uint64_t, ndim=1] conn_dims,
                np.ndarray[int32_t, ndim=1] nhood_data,
                np.ndarray[uint64_t, ndim=1] nhood_dims,
                np.ndarray[float, ndim=1] edgeWeight,
                int pos):
    segTrue = np.ascontiguousarray(segTrue)
    edgeWeight = np.ascontiguousarray(edgeWeight)
    cdef np.ndarray[float, ndim=1] nPairPerEdge = np.zeros(edgeWeight.shape[0],dtype=np.float32)
    malis_loss_weights_cpp(&segTrue[0],
                   &conn_dims[0], &nhood_data[0], &nhood_dims[0], &edgeWeight[0],
                   pos,
                   &nPairPerEdge[0]);
    return nPairPerEdge


