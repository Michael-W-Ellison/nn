// File: tests/test_temp_path.hpp
//
// Unique temporary paths for tests. ctest runs test cases as separate
// processes in parallel, so fixed /tmp names make tests interfere.
#pragma once

#include <gtest/gtest.h>
#include <atomic>
#include <filesystem>
#include <string>
#include <unistd.h>

namespace dpan {
namespace test {

/// Path under the system temp directory, unique per process, test and call
inline std::string UniqueTempPath(const std::string& prefix) {
    static std::atomic<unsigned> counter{0};
    std::string name = prefix + "_" + std::to_string(::getpid());
    if (const auto* info = ::testing::UnitTest::GetInstance()->current_test_info()) {
        name += std::string("_") + info->test_suite_name() + "_" + info->name();
    }
    name += "_" + std::to_string(counter.fetch_add(1));
    return (std::filesystem::temp_directory_path() / name).string();
}

}  // namespace test
}  // namespace dpan
