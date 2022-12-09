#include <vector>

#include "tensorstore/context.h"
#include "tensorstore/open.h"
#include "tensorstore/index_space/dim_expression.h"

namespace ts = tensorstore;

#pragma once
#ifndef _WINDOWS

/**
 * Custom exception type.
 */
class TensorStoreANNException : public std::exception {
  std::string what_msg;

 public:
  TensorStoreANNException(std::string&& what_msg)
      : what_msg("TensorStoreANNException: " + what_msg) {
  }
  ~TensorStoreANNException() = default;

  const char* what() const noexcept override {
    return what_msg.c_str();
  }
};

struct TensorsPointSliceRead {
  size_t    pt_idx;
  float*    embedding_buf;
  unsigned* num_nbrs_buf;
  unsigned* nbrhood_buf;
};

class TensorStoreSliceReader {
 private:
  ts::TensorStore<float>    store_embedding;
  ts::TensorStore<unsigned> store_num_nbrs;
  ts::TensorStore<unsigned> store_nbrhood;

 public:
  TensorStoreSliceReader();
  ~TensorStoreSliceReader();

  // Open & close ops
  // Blocking calls
  void open(const std::string& tensors_filename_prefix, size_t num_pts,
            size_t num_dims, size_t max_nbrs_per_pt);

  // process batch of tensorstore slices read requests in parallel
  // NOTE :: blocking call
  void read(std::vector<std::vector<TensorsPointSliceRead>>& read_reqs,
            bool async = false, bool skip_embedding = false,
            bool skip_neighbors = false);
};

#endif
