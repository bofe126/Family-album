#pragma once

#include "pch.h"

namespace face_recognition {

class PerformanceMonitor {
public:
    static PerformanceMonitor& getInstance();
    void beginSection(const std::string& name);
    void endSection(const std::string& name);
    void printStatistics();

private:
    PerformanceMonitor() = default;
    PerformanceMonitor(const PerformanceMonitor&) = delete;
    PerformanceMonitor& operator=(const PerformanceMonitor&) = delete;

    struct Section {
        std::chrono::time_point<std::chrono::high_resolution_clock> start;
        int64_t total_time = 0;
        int64_t min_time = std::numeric_limits<int64_t>::max();
        int64_t max_time = 0;
        int count = 0;
    };

    std::mutex m_mutex;
    std::unordered_map<std::string, Section> m_sections;
};

#define BEGIN_PROFILE(name) PerformanceMonitor::getInstance().beginSection(name)
#define END_PROFILE(name) PerformanceMonitor::getInstance().endSection(name)

} // namespace face_recognition
