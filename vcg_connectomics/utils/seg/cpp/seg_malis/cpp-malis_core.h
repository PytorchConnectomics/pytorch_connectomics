#ifndef MALIS_CPP_H
#define MALIS_CPP_H

void preCompute(const uint64_t* conn_dims, const int32_t* nhood_data, const uint64_t* nhood_dims,
        uint64_t* pre_ve, uint64_t* pre_prodDims, int32_t* pre_nHood);

// add precomputation and both pos/neg
void malis_loss_weights_cpp_both(const uint64_t* seg,
               const uint64_t* conn_dims, const int32_t* nhood_data, const uint64_t* nhood_dims,
               const float* edgeWeight, float* nPairPerEdge,
               const uint64_t* pre_ve, const uint64_t* pre_prodDims, const int32_t* pre_nHood,
               const int pos, const float weight_opt);

// add precomputation
void malis_loss_weights_cpp_pre(const uint64_t* seg,
               const uint64_t* conn_dims, const int32_t* nhood_data, const uint64_t* nhood_dims,
               const float* edgeWeight, float* nPairPerEdge,
               const uint64_t* pre_ve, const uint64_t* pre_prodDims, const int32_t* pre_nHood, const int pos);

void malis_loss_weights_cpp(const uint64_t* seg,
               const uint64_t* conn_dims, const int32_t* nhood_data, const uint64_t* nhood_dims,
               const float* edgeWeight, const int pos,
               float* nPairPerEdge);

// utility function
#endif
