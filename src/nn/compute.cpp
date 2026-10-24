// File: src/nn/compute.cpp
#include "nn/compute.hpp"

#include <algorithm>
#include <atomic>
#include <thread>
#include <vector>

namespace dpan {
namespace nn {

namespace {

std::atomic<bool> use_blas{true};

size_t DefaultThreads() {
    size_t hardware = std::thread::hardware_concurrency();
    return std::max<size_t>(1, std::min<size_t>(8, hardware));
}

std::atomic<size_t> thread_count{DefaultThreads()};

/// Below this many multiply-adds, starting threads costs more than it saves
constexpr size_t kParallelWork = size_t{1} << 20;

}  // namespace

bool BlasAvailable() {
#ifdef DPAN_HAS_BLAS
    return true;
#else
    return false;
#endif
}

void SetUseBlas(bool use) { use_blas = use; }
bool UsingBlas() { return BlasAvailable() && use_blas; }

void SetThreadCount(size_t threads) { thread_count = std::max<size_t>(1, threads); }
size_t ThreadCount() { return thread_count; }

void ParallelFor(size_t count, size_t work, const std::function<void(size_t, size_t)>& body) {
    size_t threads = std::min(ThreadCount(), count);
    if (threads <= 1 || work < kParallelWork) {
        body(0, count);
        return;
    }
    std::vector<std::thread> workers;
    workers.reserve(threads - 1);
    size_t chunk = (count + threads - 1) / threads;
    for (size_t t = 1; t < threads; ++t) {
        size_t begin = t * chunk, end = std::min(count, begin + chunk);
        if (begin < end) workers.emplace_back(body, begin, end);
    }
    body(0, std::min(count, chunk));
    for (auto& worker : workers) worker.join();
}

}  // namespace nn
}  // namespace dpan
