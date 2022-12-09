#include <iostream>
#include <stdexcept>

#include "tensorstore/context.h"
#include "tensorstore/open.h"
#include "tensorstore/index_space/dim_expression.h"

namespace ts = tensorstore;

template<typename ElemT, typename StoreT = ts::TensorStore<>>
void write_array(StoreT& store, const ElemT* arr,
                 const std::vector<int64_t>& dims) {
  auto domain = store.domain();
  auto shape = domain.shape();
  auto origin = domain.origin();
  auto dtype = store.dtype();
  std::cout << "Before write | domain.shape(): " << shape << std::endl;
  std::cout << "Before write | domain.origin(): " << origin << std::endl;
  std::cout << "Before write | store.dtype(): " << dtype << std::endl;

  auto array = ts::Array(arr, dims);

  auto write_future = ts::Write(ts::UnownedToShared(array), store);
  auto write_result = write_future.result();
  if (write_result.ok()) {
    std::cout << "Written array of dims: < ";
    for (auto&& d : dims) {
      std::cout << d << " ";
    }
    std::cout << ">" << std::endl;
  } else {
    std::cerr << "Write failed: " << write_result.status() << std::endl;
    throw std::runtime_error("tensorstore write failed");
  }
}

template<typename ElemT, typename StoreT = ts::TensorStore<>>
void read_array(StoreT& store) {
  auto domain = store.domain();
  auto shape = domain.shape();
  auto origin = domain.origin();
  auto dtype = store.dtype();
  std::cout << "Before read | store.domain().shape(): " << shape << std::endl;
  std::cout << "Before read | store.domain().origin(): " << origin << std::endl;
  std::cout << "Before read | store.dtype(): " << dtype << std::endl;

  if (dtype != ts::dtype_v<ElemT>) {
    std::cerr << "Unexpected data type: " << dtype << std::endl;
    throw std::runtime_error("unexpected data type");
  }

  if (shape.size() < 2) {
    std::cerr << "Array rank too low: " << shape.size() << std::endl;
    throw std::runtime_error("array rank too low");
  }
  if (shape[0] < 8 || shape[1] < 8) {
    std::cerr << "Array's first two dimensions too small: " << shape
              << std::endl;
    throw std::runtime_error("array dimension too small");
  }

  // a simple DimExpression that takes slice [5, 8) of the first
  // two dimensions
  auto dim_expr = ts::Dims(0, 1).HalfOpenInterval(5, 8);

  auto read_future = ts::Read<ts::zero_origin>(store | dim_expr);
  auto read_result = read_future.result();
  if (read_result.ok()) {
    auto array = read_result.value();
    std::cout << "Read array: " << array << std::endl;
  } else {
    std::cerr << "Read failed: " << read_result.status() << std::endl;
    throw std::runtime_error("tensorstore read failed");
  }
}

int main(void) {
  // create example C array
  std::vector<int64_t> dims = {30, 50, 25};
  int64_t              volume = 1;
  for (auto&& d : dims) {
    volume *= d;
  }
  uint32_t* arr = new uint32_t[volume];
  for (int64_t i = 0; i < volume; ++i) {
    arr[i] = static_cast<uint32_t>(i);
  }

  // open a tensorstore instance
  auto context = ts::Context::Default();
  auto open_future =
      ts::Open({{"driver", "zarr"},
                {"kvstore", {{"driver", "memory"}}},
                {"metadata", {{"dtype", "<u4"}, {"shape", dims}}}},
               context, ts::OpenMode::create | ts::OpenMode::delete_existing,
               ts::ReadWriteMode::read_write);
  auto open_result = open_future.result();
  if (!open_result.ok()) {
    std::cerr << "Open failed: " << open_result.status() << std::endl;
    throw std::runtime_error("tensorstore open failed");
  }
  auto store = open_result.value();

  write_array(store, arr, dims);
  read_array<uint32_t>(store);

  delete[] arr;
  return 0;
}
