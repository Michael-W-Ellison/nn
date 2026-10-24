// File: src/nn/compute.hpp
//
// How matrix products are computed. With DPAN_WITH_BLAS, products use the
// BLAS sgemm routine (it uses its own threads). Otherwise, and when BLAS is
// switched off at run time, large products split their output rows across
// threads; each output value is still computed by one thread in a fixed
// order, so results do not depend on the thread count.
#pragma once

#include <cstddef>
#include <functional>

namespace dpan {
namespace nn {

/// True if DPAN was built with BLAS (DPAN_WITH_BLAS)
bool BlasAvailable();

/// Use BLAS for matrix products when available (default: on)
void SetUseBlas(bool use);
bool UsingBlas();

/// Threads for matrix products without BLAS (default: hardware threads,
/// at most 8; 1 = no threads)
void SetThreadCount(size_t threads);
size_t ThreadCount();

/// Run body(begin, end) over [0, count) in contiguous chunks on up to
/// ThreadCount() threads, if `work` (multiply-adds) is large enough to be
/// worth it; otherwise body(0, count) on the calling thread
void ParallelFor(size_t count, size_t work, const std::function<void(size_t, size_t)>& body);

}  // namespace nn
}  // namespace dpan
