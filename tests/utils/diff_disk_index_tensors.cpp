#include <fstream>
#include <iostream>
#include <vector>
#include <tuple>
#include <cassert>

#include "tensorstore/context.h"
#include "tensorstore/open.h"
#include "tensorstore/index_space/dim_expression.h"

namespace ts = tensorstore;

// TODO: simply setting "cache_pool" parameter seems to have no effect
static constexpr size_t TENSORSTORE_CACHE_POOL_SIZE = 1000000000;  // ~1GB
static constexpr size_t READ_BATCH_TOTAL_LIMIT = 100000000;        // ~100MB

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

/**
 * Binary file utilities.
 */
static std::tuple<std::ifstream, size_t> open_binary_file(
    const std::string& filename) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open())
    throw TensorStoreANNException("failed to open binary file: " + filename);

  file.seekg(0, std::ios::end);
  size_t size = file.tellg();
  file.seekg(0, std::ios::beg);

  return std::make_tuple(std::move(file), size);
}

template<typename T>
static void read_binary_file(std::ifstream& file, T* buf, size_t bytes) {
  assert(file.is_open());
  assert(bytes % sizeof(T) == 0);

  if (!file.read(reinterpret_cast<char*>(buf), bytes))
    throw TensorStoreANNException("binary file read failed");
  if (file.gcount() != bytes)
    throw TensorStoreANNException(
        "binary file read fewer bytes than expected: " +
        std::to_string(file.gcount()) + " of " + std::to_string(bytes));
}

/**
 * TensorStore utilities.
 */
template<typename V>
static ts::TensorStore<V> open_tensorstore(ts::Context&                context,
                                           const std::string&          filename,
                                           const std::vector<int64_t>& dims) {
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
      ts::Open<V>(
          {{"driver", "zarr"},
           {"kvstore", {{"driver", "file"}, {"path", filename}}},
           {"cache_pool", {{"total_bytes_limit", TENSORSTORE_CACHE_POOL_SIZE}}},
           {"metadata", {{"dtype", dtype_str}, {"shape", dims}}}},
          context, ts::OpenMode::open, ts::ReadWriteMode::read)
          .result();
  if (!open_result.ok())
    throw TensorStoreANNException("failed to open TensorStore instance: " +
                                  open_result.status().ToString());

  return std::move(open_result.value());
}

template<typename V>
static auto tensor2d_read_slices(ts::TensorStore<V>& store, int64_t dim,
                                 size_t idx_beg, size_t idx_end) {
  // TODO: maybe utilize async I/O?
  auto read_result =
      ts::Read<ts::zero_origin>(
          store | ts::Dims(dim).HalfOpenInterval(static_cast<int64_t>(idx_beg),
                                                 static_cast<int64_t>(idx_end)))
          .result();
  if (!read_result.ok())
    throw TensorStoreANNException(
        "failed to read to tensor " + std::to_string(dim) + "[" +
        std::to_string(idx_beg) + ":" + std::to_string(idx_end) +
        "]: " + read_result.status().ToString());

  return read_result.value();
}

/**
 * Disk index binary file format constants.
 */
static constexpr int32_t DISK_INDEX_META_NR = 9;
static constexpr int32_t DISK_INDEX_META_NC = 1;
static constexpr size_t  DISK_INDEX_META_SIZE =
    DISK_INDEX_META_NR * sizeof(uint64_t);

static constexpr size_t DISK_INDEX_SECTOR_LEN = 4096;

/**
 * Data sectors sweeper.
 */
template<typename V>
static void compare_points_data(std::ifstream&             disk_index_file,
                                ts::TensorStore<V>&        store_embedding,
                                ts::TensorStore<unsigned>& store_num_nbrs,
                                ts::TensorStore<unsigned>& store_nbrhood,
                                size_t num_pts, size_t num_pts_per_sector,
                                size_t max_pt_len, size_t num_dims,
                                size_t max_nbrs_per_pt) {
  char*  sector_buf = new char[DISK_INDEX_SECTOR_LEN];
  size_t done_pts = 0, buffer_pts = 0, buffer_start = 0;

  // use read batching
  size_t num_pts_to_buffer = READ_BATCH_TOTAL_LIMIT / max_pt_len;
  if (num_pts_to_buffer > num_pts)
    num_pts_to_buffer = num_pts;
  assert(num_pts_to_buffer > 0);
  V*        embedding_buf = new V[num_dims * num_pts_to_buffer];
  unsigned* num_nbrs_buf = new unsigned[num_pts_to_buffer];
  unsigned* nbrhood_buf = new unsigned[max_nbrs_per_pt * num_pts_to_buffer];

  auto compare_points_batch = [&]() {
    // read the corresponding array slices from tensors
    auto embedding_array = tensor2d_read_slices<V>(
        store_embedding, 0, buffer_start, buffer_start + buffer_pts);
    auto num_nbrs_array = tensor2d_read_slices<unsigned>(
        store_num_nbrs, 0, buffer_start, buffer_start + buffer_pts);
    auto nbrhood_array = tensor2d_read_slices<unsigned>(
        store_nbrhood, 0, buffer_start, buffer_start + buffer_pts);

    // compare embedding
    if (embedding_array.rank() != 2)
      throw TensorStoreANNException(
          "embedding array [" + std::to_string(buffer_start) + ":" +
          std::to_string(buffer_start + buffer_pts) +
          "] wrong rank: " + std::to_string(embedding_array.rank()));
    if (embedding_array.shape()[0] != buffer_pts)
      throw TensorStoreANNException(
          "embedding array [" + std::to_string(buffer_start) + ":" +
          std::to_string(buffer_start + buffer_pts) +
          "] wrong #points: " + std::to_string(embedding_array.shape()[0]));
    if (embedding_array.shape()[1] != num_dims)
      throw TensorStoreANNException(
          "embedding array [" + std::to_string(buffer_start) + ":" +
          std::to_string(buffer_start + buffer_pts) +
          "] wrong #values: " + std::to_string(embedding_array.shape()[1]));
    for (size_t p = 0; p < buffer_pts; ++p) {
      for (size_t j = 0; j < num_dims; ++j) {
        if (embedding_array.data()[p * num_dims + j] !=
            embedding_buf[p * num_dims + j])
          throw TensorStoreANNException(
              "embedding array [" + std::to_string(buffer_start + p) +
              "] wrong value @ " + std::to_string(j) + ": " +
              std::to_string(embedding_array.data()[p * num_dims + j]) +
              " vs. " + std::to_string(embedding_buf[p * num_dims + j]));
      }
    }

    // compare num_nbrs
    if (num_nbrs_array.rank() != 2)
      throw TensorStoreANNException(
          "num_nbrs array [" + std::to_string(buffer_start) + ":" +
          std::to_string(buffer_start + buffer_pts) +
          "] wrong rank: " + std::to_string(num_nbrs_array.rank()));
    if (num_nbrs_array.shape()[0] != buffer_pts)
      throw TensorStoreANNException(
          "num_nbrs array [" + std::to_string(buffer_start) + ":" +
          std::to_string(buffer_start + buffer_pts) +
          "] wrong #points: " + std::to_string(num_nbrs_array.shape()[0]));
    if (num_nbrs_array.shape()[1] != 1)
      throw TensorStoreANNException(
          "num_nbrs array [" + std::to_string(buffer_start) + ":" +
          std::to_string(buffer_start + buffer_pts) +
          "] wrong #values: " + std::to_string(num_nbrs_array.shape()[1]));
    for (size_t p = 0; p < buffer_pts; ++p) {
      if (num_nbrs_array.data()[p] != num_nbrs_buf[p])
        throw TensorStoreANNException(
            "num_nbrs array [" + std::to_string(buffer_start + p) +
            "] wrong value @ : " + std::to_string(num_nbrs_array.data()[p]) +
            " vs. " + std::to_string(num_nbrs_buf[p]));
    }

    // compare nbrhood
    if (nbrhood_array.rank() != 2)
      throw TensorStoreANNException(
          "nbrhood array [" + std::to_string(buffer_start) + ":" +
          std::to_string(buffer_start + buffer_pts) +
          "] wrong rank: " + std::to_string(nbrhood_array.rank()));
    if (nbrhood_array.shape()[0] != buffer_pts)
      throw TensorStoreANNException(
          "nbrhood array [" + std::to_string(buffer_start) + ":" +
          std::to_string(buffer_start + buffer_pts) +
          "] wrong #points: " + std::to_string(nbrhood_array.shape()[0]));
    if (nbrhood_array.shape()[1] != max_nbrs_per_pt)
      throw TensorStoreANNException(
          "nbrhood array [" + std::to_string(buffer_start) + ":" +
          std::to_string(buffer_start + buffer_pts) +
          "] wrong #values: " + std::to_string(nbrhood_array.shape()[1]));
    for (size_t p = 0; p < buffer_pts; ++p) {
      for (size_t j = 0; j < max_nbrs_per_pt; ++j) {
        if (nbrhood_array.data()[p * max_nbrs_per_pt + j] !=
            nbrhood_buf[p * max_nbrs_per_pt + j])
          throw TensorStoreANNException(
              "nbrhood array [" + std::to_string(buffer_start + p) +
              "] wrong value @ " + std::to_string(j) + ": " +
              std::to_string(nbrhood_array.data()[p * max_nbrs_per_pt + j]) +
              " vs. " + std::to_string(nbrhood_buf[p * max_nbrs_per_pt + j]));
      }
    }
  };

  // loop through all points in input binary disk index, gather in buffer
  // when batch is full, read out corresponding arrays from tensors and compare
  while (done_pts < num_pts) {
    size_t sector_pts = num_pts_per_sector;
    if (done_pts + sector_pts > num_pts)
      sector_pts = num_pts - done_pts;

    read_binary_file<char>(disk_index_file, sector_buf, DISK_INDEX_SECTOR_LEN);

    for (size_t i = 0; i < sector_pts; ++i) {
      char* buf_cursor = &sector_buf[i * max_pt_len];
      V*    embedding = reinterpret_cast<V*>(buf_cursor);
      buf_cursor += sizeof(V) * num_dims;
      unsigned num_nbrs = *reinterpret_cast<unsigned*>(buf_cursor);
      buf_cursor += sizeof(unsigned);
      unsigned* nbrhood = reinterpret_cast<unsigned*>(buf_cursor);

      // copy into read batch
      memcpy(embedding_buf + num_dims * buffer_pts, embedding,
             sizeof(V) * num_dims);
      memcpy(num_nbrs_buf + buffer_pts, &num_nbrs, sizeof(unsigned));
      memcpy(nbrhood_buf + max_nbrs_per_pt * buffer_pts, nbrhood,
             sizeof(unsigned) * max_nbrs_per_pt);
      done_pts++;
      buffer_pts++;

      // if read batch full, compare with arrays in tensors
      if (buffer_pts == num_pts_to_buffer) {
        compare_points_batch();
        buffer_start += buffer_pts;
        buffer_pts = 0;
      }

      if (done_pts % num_pts_to_buffer == 0)
        std::cout << "  compared " << done_pts << " points..." << std::endl;
    }
  }

  if (buffer_pts > 0)
    compare_points_batch();

  std::cout << "  comparison of " << num_pts << " points SUCCESS" << std::endl;
  delete[] sector_buf;
  delete[] embedding_buf;
  delete[] num_nbrs_buf;
  delete[] nbrhood_buf;
}

/**
 * Main body.
 */
template<typename V>
void compare_disk_index_and_tensors(
    const std::string& disk_index_filename,
    const std::string& tensors_filename_prefix) {
  // open binary file
  std::ifstream disk_index_file;
  size_t        disk_index_filesize;
  std::tie(disk_index_file, disk_index_filesize) =
      open_binary_file(disk_index_filename);
  if (disk_index_filesize < DISK_INDEX_META_SIZE + 2 * sizeof(int32_t))
    throw TensorStoreANNException("disk_index filesize too small: " +
                                  std::to_string(disk_index_filesize));

  // read metadata
  int32_t meta_nr = 0, meta_nc = 0;
  read_binary_file<int32_t>(disk_index_file, &meta_nr, sizeof(int32_t));
  if (meta_nr != DISK_INDEX_META_NR)
    throw TensorStoreANNException("disk_index meta_nr != " +
                                  std::to_string(DISK_INDEX_META_NR));
  read_binary_file<int32_t>(disk_index_file, &meta_nc, sizeof(int32_t));
  if (meta_nc != DISK_INDEX_META_NC)
    throw TensorStoreANNException("disk_index meta_nc != " +
                                  std::to_string(DISK_INDEX_META_NC));

  uint64_t metadata[DISK_INDEX_META_NR];
  read_binary_file<uint64_t>(disk_index_file, metadata, DISK_INDEX_META_SIZE);

  uint64_t num_pts = metadata[0];
  uint64_t num_dims = metadata[1];
  uint64_t medoid = metadata[2];
  uint64_t max_pt_len = metadata[3];
  uint64_t num_pts_per_sector = metadata[4];
  uint64_t vamana_frozen_num = metadata[5];
  uint64_t vamana_frozen_loc = metadata[6];
  uint64_t append_reorder_data = metadata[7];
  uint64_t file_size = metadata[8];
  if (file_size != disk_index_filesize)
    throw TensorStoreANNException(
        "disk_index metadata filesize field mismatch: " +
        std::to_string(file_size) + " vs. " +
        std::to_string(disk_index_filesize));

  uint64_t max_nbrs_per_pt =
      (max_pt_len - num_dims * sizeof(V) - sizeof(unsigned)) / sizeof(unsigned);

  std::cout << "Disk index metadata --" << std::endl
            << "  #points:            " << num_pts << std::endl
            << "  #dims:              " << num_dims << std::endl
            << "  medoid:             " << medoid << std::endl
            << "  max point len:      " << max_pt_len << std::endl
            << "  max #nbrs of point: " << max_nbrs_per_pt << std::endl
            << "  #points per sector: " << num_pts_per_sector << std::endl
            << "  vamana frozen num:  " << vamana_frozen_num << std::endl
            << "  vamana frozen loc:  " << vamana_frozen_loc << std::endl
            << "  append reorder:     " << append_reorder_data << std::endl
            << "  file size:          " << file_size << std::endl;

  // open tensorstore tensors
  auto context = ts::Context::Default();

  std::vector<int64_t> embedding_dims = {static_cast<int64_t>(num_pts),
                                         static_cast<int64_t>(num_dims)};
  std::string embedding_filename = tensors_filename_prefix + "_embedding.zarr";
  auto        store_embedding =
      open_tensorstore<V>(context, embedding_filename, embedding_dims);

  std::vector<int64_t> num_nbrs_dims = {static_cast<int64_t>(num_pts), 1};
  std::string num_nbrs_filename = tensors_filename_prefix + "_num_nbrs.zarr";
  auto        store_num_nbrs =
      open_tensorstore<uint32_t>(context, num_nbrs_filename, num_nbrs_dims);

  std::vector<int64_t> nbrhood_dims = {static_cast<int64_t>(num_pts),
                                       static_cast<int64_t>(max_nbrs_per_pt)};
  std::string nbrhood_filename = tensors_filename_prefix + "_nbrhood.zarr";
  auto        store_nbrhood =
      open_tensorstore<uint32_t>(context, nbrhood_filename, nbrhood_dims);

  std::cout << "Tensors metadata --" << std::endl
            << "  embedding domain shape:      "
            << store_embedding.domain().shape() << std::endl
            << "            domain origin:     "
            << store_embedding.domain().origin() << std::endl
            << "            dtype:             " << store_embedding.dtype()
            << std::endl
            << "  num_neighbors domain shape:  "
            << store_num_nbrs.domain().shape() << std::endl
            << "                domain origin: "
            << store_num_nbrs.domain().origin() << std::endl
            << "                dtype:         " << store_num_nbrs.dtype()
            << std::endl
            << "  neighborhood domain shape:   "
            << store_nbrhood.domain().shape() << std::endl
            << "               domain origin:  "
            << store_nbrhood.domain().origin() << std::endl
            << "               dtype:          " << store_nbrhood.dtype()
            << std::endl;

  // read sectors one by one, extract nodes data and compare with tensorstore
  disk_index_file.seekg(DISK_INDEX_SECTOR_LEN, std::ios::beg);
  std::cout << "Comparing embedding & neighborhood data --" << std::endl;
  compare_points_data<V>(disk_index_file, store_embedding, store_num_nbrs,
                         store_nbrhood, num_pts, num_pts_per_sector, max_pt_len,
                         num_dims, max_nbrs_per_pt);
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    std::cout << "Usage: " << std::string(argv[0])
              << " <data_type> <disk_index_filename> <output_filename_prefix>"
              << std::endl;
    std::cout << "  valid data_type: float | int8 | uint8" << std::endl;
    return 1;
  }

  std::string data_type(argv[1]);
  std::string disk_index_filename(argv[2]);
  std::string tensors_filename_prefix(argv[3]);

  if (data_type == "float")
    compare_disk_index_and_tensors<float>(disk_index_filename,
                                          tensors_filename_prefix);
  else if (data_type == "int8")
    compare_disk_index_and_tensors<int8_t>(disk_index_filename,
                                           tensors_filename_prefix);
  else if (data_type == "uint8")
    compare_disk_index_and_tensors<uint8_t>(disk_index_filename,
                                            tensors_filename_prefix);
  else
    throw TensorStoreANNException("unsupported data type: " + data_type);

  return 0;
}
