#pragma once

#include "tamm/gpu_streams.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace rmm {

/**
 * @brief Strongly-typed non-owning wrapper for CUDA streams with default constructor.
 *
 * This wrapper is simply a "view": it does not own the lifetime of the stream it wraps.
 */
class cuda_stream_view {
public:
  constexpr cuda_stream_view()                                   = default;
  constexpr cuda_stream_view(cuda_stream_view const&)            = default;
  constexpr cuda_stream_view(cuda_stream_view&&)                 = default;
  constexpr cuda_stream_view& operator=(cuda_stream_view const&) = default;
  constexpr cuda_stream_view& operator=(cuda_stream_view&&)      = default;
  ~cuda_stream_view()                                            = default;

  // Disable construction from literal 0
  constexpr cuda_stream_view(int)            = delete; //< Prevent cast from 0
  constexpr cuda_stream_view(std::nullptr_t) = delete; //< Prevent cast from nullptr

  /**
   * @brief Implicit conversion from gpuStream_t.
   */
  constexpr cuda_stream_view(gpuStream_t stream) noexcept: stream_{stream} {}

  /**
   * @brief Get the wrapped stream.
   *
   * @return gpuStream_t The wrapped stream.
   */
  [[nodiscard]] constexpr gpuStream_t value() const noexcept { return stream_; }

  /**
   * @brief Implicit conversion to gpuStream_t.
   */
  constexpr operator gpuStream_t() const noexcept { return value(); }

  /**
   * @brief Synchronize the viewed CUDA/HIP/SYCL stream.
   *
   * Calls `cudaStreamSynchronize(), hipStreamSynchronize(), sycl::event::wait_and_throw()`.
   *
   * @throw rmm::cuda_error if stream synchronization fails
   */
  void synchronize() const {
#if defined(USE_CUDA)
    cudaStreamSynchronize(stream_);
#elif defined(USE_HIP)
    hipStreamSynchronize(stream_);
#elif defined(USE_DPCPP)
    stream_.wait_and_throw();
#endif
  }

  /**
   * @brief Synchronize the viewed CUDA stream. Does not throw if there is an error.
   *
   * Calls `cudaStreamSynchronize()` and asserts if there is an error.
   */
  void synchronize_no_throw() const noexcept {
#if defined(USE_CUDA)
    cudaStreamSynchronize(stream_);
#elif defined(USE_HIP)
    hipStreamSynchronize(stream_);
#elif defined(USE_DPCPP)
    stream_.wait();
#endif
  }

private:
  gpuStream_t stream_{};
};

/**
 * @brief Equality comparison operator for streams
 *
 * @param lhs The first stream view to compare
 * @param rhs The second stream view to compare
 * @return true if equal, false if unequal
 */
inline bool operator==(cuda_stream_view lhs, cuda_stream_view rhs) {
  return lhs.value() == rhs.value();
}

/**
 * @brief Inequality comparison operator for streams
 *
 * @param lhs The first stream view to compare
 * @param rhs The second stream view to compare
 * @return true if unequal, false if equal
 */
inline bool operator!=(cuda_stream_view lhs, cuda_stream_view rhs) { return not(lhs == rhs); }

/**
 * @brief Output stream operator for printing / logging streams
 *
 * @param os The output ostream
 * @param sv The cuda_stream_view to output
 * @return std::ostream& The output ostream
 */
inline std::ostream& operator<<(std::ostream& os, cuda_stream_view stream) {
  os << stream.value();
  return os;
}

} // namespace rmm
