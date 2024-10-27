#include "performance.h"
#include "utils.h"

namespace face_recognition {

PerformanceMonitor& PerformanceMonitor::getInstance() {
    static PerformanceMonitor instance;
    return instance;
}

void PerformanceMonitor::beginSection(const std::string& name) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_sections[name].start = std::chrono::high_resolution_clock::now();
}

void PerformanceMonitor::endSection(const std::string& name) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto& section = m_sections[name];
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - section.start).count();
    
    section.total_time += duration;
    section.count++;
    section.min_time = std::min(section.min_time, duration);
    section.max_time = std::max(section.max_time, duration);

    LOG(name + " 性能统计: " + 
        "当前=" + std::to_string(duration) + "ms, " +
        "平均=" + std::to_string(section.total_time / section.count) + "ms, " +
        "最小=" + std::to_string(section.min_time) + "ms, " +
        "最大=" + std::to_string(section.max_time) + "ms, " +
        "次数=" + std::to_string(section.count));
}

void PerformanceMonitor::printStatistics() {
    std::lock_guard<std::mutex> lock(m_mutex);
    LOG("性能统计汇总:");
    for (const auto& pair : m_sections) {
        const auto& name = pair.first;
        const auto& section = pair.second;
        LOG(name + ": " +
            "平均=" + std::to_string(section.total_time / section.count) + "ms, " +
            "最小=" + std::to_string(section.min_time) + "ms, " +
            "最大=" + std::to_string(section.max_time) + "ms, " +
            "总次数=" + std::to_string(section.count));
    }
}

} // namespace face_recognition
