#include "tensorstore_slice_reader.h"

#include <cassert>
#include <cstdio>
#include <iostream>
#include <functional>
#include <algorithm>
#include "utils.h"

#include "tensorstore/context.h"
#include "tensorstore/open.h"
#include "tensorstore/index_space/dim_expression.h"

namespace {
  static constexpr size_t TENSORSTORE_CACHE_POOL_SIZE = 5000000000;  // ~5GB

  template<typename V>
  static ts::TensorStore<V> open_tensorstore(ts::Context &      context,
                                             const std::string &filename,
                                             const std::vector<int64_t> &dims,
                                             const char *use_remote_addr) {
    std::string dtype_str;
    if constexpr (std::is_same<V, float>::value)
      dtype_str = "<f4";
    else if constexpr (std::is_same<V, int8_t>::value)
      dtype_str = "|i1";
    else if constexpr (std::is_same<V, int32_t>::value)
      dtype_str = "<i4";
    else if constexpr (std::is_same<V, uint8_t>::value)
      dtype_str = "|u1";
    else if constexpr (std::is_same<V, uint32_t>::value)
      dtype_str = "<u4";

    auto open_result =
        use_remote_addr
            ? ts::Open<V>(
                  {{"driver", "zarr"},
                   {"kvstore",
                    {{"driver", "http"},
                     {"base_url", use_remote_addr},
                     {"path", filename}}},
                   {"cache_pool",
                    {{"total_bytes_limit", TENSORSTORE_CACHE_POOL_SIZE}}},
                   {"recheck_cached_data", false},
                   {"metadata", {{"dtype", dtype_str}, {"shape", dims}}}},
                  context, ts::OpenMode::open, ts::ReadWriteMode::read)
                  .result()
            : ts::Open<V>(
                  {{"driver", "zarr"},
                   {"kvstore", {{"driver", "file"}, {"path", filename}}},
                   {"cache_pool",
                    {{"total_bytes_limit", TENSORSTORE_CACHE_POOL_SIZE}}},
                   {"recheck_cached_data", false},
                   {"metadata", {{"dtype", dtype_str}, {"shape", dims}}}},
                  context, ts::OpenMode::open, ts::ReadWriteMode::read)
                  .result();
    if (!open_result.ok())
      throw TensorStoreANNException("failed to open TensorStore instance: " +
                                    open_result.status().ToString());

    return std::move(open_result.value());
  }

  template<typename V>
  static auto tensor2d_submit_read_slice(ts::TensorStore<V> &store, int64_t dim,
                                         const std::vector<int64_t> &idxs) {
    return ts::Read<ts::zero_origin>(
        store |
        ts::Dims(dim).IndexArraySlice(ts::UnownedToShared(
            ts::Array<ts::Index>(const_cast<int64_t *>(idxs.data()),
                                 {static_cast<int64_t>(idxs.size())}))));
  }

  template<typename V>
  static void tensor2d_resolve_read_future(
      ts::Future<ts::Array<ts::Shared<V>>> future,
      const std::vector<V *> &             bufs) {
    auto read_result = future.result();
    if (!read_result.ok())
      throw TensorStoreANNException("failed to resolve read future: " +
                                    read_result.status().ToString());

    auto array = read_result.value();
    if (array.rank() != 2)
      throw TensorStoreANNException("data tensor's rank is not 2: " +
                                    std::to_string(array.rank()));
    if (array.shape()[0] != bufs.size())
      throw TensorStoreANNException(
          "buffers vector has mismatch size: " + std::to_string(bufs.size()) +
          " vs. " + std::to_string(array.shape()[0]));

    // copy data to buffers
    size_t num_elems_per_buf = array.shape()[1];
    for (size_t i = 0; i < bufs.size(); ++i) {
      if (bufs[i] != nullptr) {
        memcpy(bufs[i], &array.data()[i * num_elems_per_buf],
               num_elems_per_buf * sizeof(V));
      }
    }
  }
}  // namespace

TensorStoreSliceReader::TensorStoreSliceReader() {
}

TensorStoreSliceReader::~TensorStoreSliceReader() {
}

void TensorStoreSliceReader::open(const std::string &tensors_filename_prefix,
                                  size_t num_pts, size_t num_dims,
                                  size_t      max_nbrs_per_pt,
                                  const char *use_remote_addr) {
  auto context = ts::Context::Default();

  std::vector<int64_t> embedding_dims = {static_cast<int64_t>(num_pts),
                                         static_cast<int64_t>(num_dims)};
  std::string embedding_filename = tensors_filename_prefix + "_embedding.zarr";
  store_embedding = open_tensorstore<float>(context, embedding_filename,
                                            embedding_dims, use_remote_addr);
  std::cerr << "Opened TensorStore tensor: " << embedding_filename << std::endl;

  std::vector<int64_t> num_nbrs_dims = {static_cast<int64_t>(num_pts), 1};
  std::string num_nbrs_filename = tensors_filename_prefix + "_num_nbrs.zarr";
  store_num_nbrs = open_tensorstore<uint32_t>(context, num_nbrs_filename,
                                              num_nbrs_dims, use_remote_addr);
  std::cerr << "Opened TensorStore tensor: " << num_nbrs_filename << std::endl;

  std::vector<int64_t> nbrhood_dims = {static_cast<int64_t>(num_pts),
                                       static_cast<int64_t>(max_nbrs_per_pt)};
  std::string nbrhood_filename = tensors_filename_prefix + "_nbrhood.zarr";
  store_nbrhood = open_tensorstore<uint32_t>(context, nbrhood_filename,
                                             nbrhood_dims, use_remote_addr);
  std::cerr << "Opened TensorStore tensor: " << nbrhood_filename << std::endl;
}

void TensorStoreSliceReader::read(
    std::vector<std::vector<TensorsPointSliceRead>> &read_reqs, bool async,
    bool skip_embedding, bool skip_neighbors) {
  // inner vector is a list of slice indexes to be read in one shot
  // outer vector is a list of such read calls that could be done sync/async
  size_t num_reqs = read_reqs.size();

  std::vector<ts::Future<ts::Array<ts::Shared<float>>>>    embedding_futures;
  std::vector<ts::Future<ts::Array<ts::Shared<unsigned>>>> num_nbrs_futures;
  std::vector<ts::Future<ts::Array<ts::Shared<unsigned>>>> nbrhood_futures;
  embedding_futures.reserve(num_reqs);
  num_nbrs_futures.reserve(num_reqs);
  nbrhood_futures.reserve(num_reqs);

  std::vector<std::vector<int64_t>>    pt_idxs;
  std::vector<std::vector<float *>>    embedding_bufs;
  std::vector<std::vector<unsigned *>> num_nbrs_bufs;
  std::vector<std::vector<unsigned *>> nbrhood_bufs;
  pt_idxs.reserve(num_reqs);
  embedding_bufs.reserve(num_reqs);
  num_nbrs_bufs.reserve(num_reqs);
  nbrhood_bufs.reserve(num_reqs);

  for (auto &&req : read_reqs) {
    pt_idxs.emplace_back();
    std::transform(req.begin(), req.end(), std::back_inserter(pt_idxs.back()),
                   std::mem_fn(&TensorsPointSliceRead::pt_idx));
    embedding_bufs.emplace_back();
    std::transform(req.begin(), req.end(),
                   std::back_inserter(embedding_bufs.back()),
                   std::mem_fn(&TensorsPointSliceRead::embedding_buf));
    num_nbrs_bufs.emplace_back();
    std::transform(req.begin(), req.end(),
                   std::back_inserter(num_nbrs_bufs.back()),
                   std::mem_fn(&TensorsPointSliceRead::num_nbrs_buf));
    nbrhood_bufs.emplace_back();
    std::transform(req.begin(), req.end(),
                   std::back_inserter(nbrhood_bufs.back()),
                   std::mem_fn(&TensorsPointSliceRead::nbrhood_buf));
  }

  for (size_t i = 0; i < num_reqs; ++i) {
    if (!skip_embedding) {
      auto embedding_future =
          tensor2d_submit_read_slice<float>(store_embedding, 0, pt_idxs[i]);
      if (!async)
        tensor2d_resolve_read_future<float>(std::move(embedding_future),
                                            embedding_bufs[i]);
      else
        embedding_futures.push_back(std::move(embedding_future));
    }

    if (!skip_neighbors) {
      auto num_nbrs_future =
          tensor2d_submit_read_slice<unsigned>(store_num_nbrs, 0, pt_idxs[i]);
      if (!async)
        tensor2d_resolve_read_future<unsigned>(std::move(num_nbrs_future),
                                               num_nbrs_bufs[i]);
      else
        num_nbrs_futures.push_back(std::move(num_nbrs_future));

      auto nbrhood_future =
          tensor2d_submit_read_slice<unsigned>(store_nbrhood, 0, pt_idxs[i]);
      if (!async)
        tensor2d_resolve_read_future<unsigned>(std::move(nbrhood_future),
                                               nbrhood_bufs[i]);
      else
        nbrhood_futures.push_back(std::move(nbrhood_future));
    }
  }

  if (async) {
    for (size_t i = 0; i < num_reqs; ++i) {
      if (!skip_embedding) {
        tensor2d_resolve_read_future<float>(std::move(embedding_futures[i]),
                                            embedding_bufs[i]);
      }
      if (!skip_neighbors) {
        tensor2d_resolve_read_future<unsigned>(std::move(num_nbrs_futures[i]),
                                               num_nbrs_bufs[i]);
        tensor2d_resolve_read_future<unsigned>(std::move(nbrhood_futures[i]),
                                               nbrhood_bufs[i]);
      }
    }
  }
}
