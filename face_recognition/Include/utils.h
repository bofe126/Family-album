#pragma once

#include "pch.h"
#include "types.h"
#include "model_cache.h"

namespace face_recognition {

// 字符串转换函数
std::wstring ConvertToWideString(const std::string& str);
std::string ConvertToUTF8(const std::wstring& wstr);
void setLocale();

// 日志相关
void initializeLogFile();
void log(const std::string& file, int line, const std::string& message);
#define LOG(message) log(__FILE__, __LINE__, message)

// ONNX Runtime 相关
OrtEnv* getGlobalOrtEnv();

// 数学工具函数
float sigmoid(float x);
float iou(const Boxf& a, const Boxf& b);

// 图像处理工具
cv::Mat loadImage(const std::string& image_path);
cv::Mat processLargeImage(const cv::Mat& image);
void initializeOpenCV();

// NMS 函数
void nms_bboxes_kps(std::vector<BoxfWithLandmarks>& input,
                    std::vector<BoxfWithLandmarks>& output,
                    float iou_threshold, 
                    unsigned int topk);

// 性能计时器
class Timer {
public:
    explicit Timer(const std::string& name);
    ~Timer();

private:
    std::string m_name;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
};

// 资源监控
class ResourceMonitor {
public:
    static ResourceMonitor& getInstance();
    void startMonitoring();
    void stopMonitoring();

private:
    ResourceMonitor() = default;
    ResourceMonitor(const ResourceMonitor&) = delete;
    ResourceMonitor& operator=(const ResourceMonitor&) = delete;
};

// 内存池
class MemoryPool {
public:
    static MemoryPool& getInstance();
    void* allocate(size_t size);
    void deallocate(void* ptr);

private:
    MemoryPool() = default;
    std::mutex m_mutex;
    std::unordered_map<void*, size_t> m_allocations;
    size_t m_totalMemory{0};
};

// 路径处理工具
std::string normalizePath(const std::string& path);

} // namespace face_recognition
