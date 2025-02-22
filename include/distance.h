#pragma once
#include "windows_customizations.h"

namespace diskann {
  enum Metric { L2 = 0, INNER_PRODUCT = 1, COSINE = 2, FAST_L2 = 3 };

  template<typename T>
  class Distance {
   public:
    virtual float compare(const T *a, const T *b, uint32_t length) const = 0;
    virtual ~Distance() {
    }
  };

  class DistanceCosineInt8 : public Distance<int8_t> {
   public:
    DISKANN_DLLEXPORT virtual float compare(const int8_t *a, const int8_t *b,
                                            uint32_t length) const;
  };

  class DistanceL2Int8 : public Distance<int8_t> {
   public:
    DISKANN_DLLEXPORT virtual float compare(const int8_t *a, const int8_t *b,
                                            uint32_t size) const;
  };

  // AVX implementations. Borrowed from HNSW code.
  class AVXDistanceL2Int8 : public Distance<int8_t> {
   public:
    DISKANN_DLLEXPORT virtual float compare(const int8_t *a, const int8_t *b,
                                            uint32_t length) const;
  };

  class DistanceCosineFloat : public Distance<float> {
   public:
    DISKANN_DLLEXPORT virtual float compare(const float *a, const float *b,
                                            uint32_t length) const;
  };

  class DistanceL2Float : public Distance<float> {
   public:
#ifdef _WINDOWS
    DISKANN_DLLEXPORT virtual float compare(const float *a, const float *b,
                                            uint32_t size) const;
#else
    DISKANN_DLLEXPORT virtual float compare(const float *a, const float *b,
                                            uint32_t size) const
        __attribute__((hot));
#endif
  };

  class AVXDistanceL2Float : public Distance<float> {
   public:
    DISKANN_DLLEXPORT virtual float compare(const float *a, const float *b,
                                            uint32_t length) const;
  };

  class SlowDistanceL2Float : public Distance<float> {
   public:
    DISKANN_DLLEXPORT virtual float compare(const float *a, const float *b,
                                            uint32_t length) const;
  };

  class SlowDistanceCosineUInt8 : public Distance<uint8_t> {
   public:
    DISKANN_DLLEXPORT virtual float compare(const uint8_t *a, const uint8_t *b,
                                            uint32_t length) const;
  };

  class DistanceL2UInt8 : public Distance<uint8_t> {
   public:
    DISKANN_DLLEXPORT virtual float compare(const uint8_t *a, const uint8_t *b,
                                            uint32_t size) const;
  };

  // Simple implementations for non-AVX machines. Compiler can optimize.
  template<typename T>
  class SlowDistanceL2Int : public Distance<T> {
   public:
    // Implementing here because this is a template function
    DISKANN_DLLEXPORT virtual float compare(const T *a, const T *b,
                                            uint32_t length) const {
      uint32_t result = 0;
      for (uint32_t i = 0; i < length; i++) {
        result += ((int32_t)((int16_t) a[i] - (int16_t) b[i])) *
                  ((int32_t)((int16_t) a[i] - (int16_t) b[i]));
      }
      return (float) result;
    }
  };

  template<typename T>
  class DistanceInnerProduct : public Distance<T> {
   public:
    inline float inner_product(const T *a, const T *b, unsigned size) const;

    inline float compare(const T *a, const T *b, unsigned size) const {
      float result = inner_product(a, b, size);
      //      if (result < 0)
      //      return std::numeric_limits<float>::max();
      //      else
      return -result;
    }
  };

  template<typename T>
  class DistanceFastL2
      : public DistanceInnerProduct<T> {  // currently defined only for float.
                                          // templated for future use.
   public:
    float norm(const T *a, unsigned size) const;
    float compare(const T *a, const T *b, float norm, unsigned size) const;
  };

  class AVXDistanceInnerProductFloat : public Distance<float> {
   public:
    DISKANN_DLLEXPORT virtual float compare(const float *a, const float *b,
                                            uint32_t length) const;
  };

  class AVXNormalizedCosineDistanceFloat : public Distance<float> {
   private:
    AVXDistanceInnerProductFloat _innerProduct;

   public:
    DISKANN_DLLEXPORT virtual float compare(const float *a, const float *b,
                                            uint32_t length) const {
      // Inner product returns negative values to indicate distance.
      // This will ensure that cosine is between -1 and 1.
      return 1.0f + _innerProduct.compare(a, b, length);
    }
  };

  template<typename T>
  Distance<T> *get_distance_function(Metric m);

}  // namespace diskann
