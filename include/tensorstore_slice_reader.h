#include <vector>

#include "tensorstore/context.h"
#include "tensorstore/open.h"
#include "tensorstore/index_space/dim_expression.h"

namespace ts = tensorstore;

#pragma once
#ifndef _WINDOWS

struct TensorSliceRead {};

class TensorStoreSliceReader {
 private:
  ts::TensorStore<float>    store_embedding;
  ts::TensorStore<unsigned> store_num_nbrs;
  ts::TensorStore<unsigned> store_nbrhood;

 public:
  TensorStoreSliceReader();
  ~TensorStoreSliceReader();

  // register thread-id for a context
  void register_thread();

  // de-register thread-id for a context
  void deregister_thread();
  void deregister_all_threads();

  // Open & close ops
  // Blocking calls
  void open(const std::string& tensors_filename_prefix);
  void close();

  // process batch of tensorstore slice read requests in parallel
  // NOTE :: blocking call
  void read(std::vector<TensorSliceRead>& read_reqs, bool async = false);
};

#endif
