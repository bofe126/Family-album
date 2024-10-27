#include "utils.h"
#include "ort_env.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <mutex>
#include <Windows.h>

namespace face_recognition {

// 全局变量
namespace {
    std::string logFilePath;  // 日志文件路径
    std::mutex logMutex;      // 日志互斥锁
}

// 字符串转换函数实现
std::wstring ConvertToWideString(const std::string& str) {
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), NULL, 0);
    std::wstring wstrTo(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), &wstrTo[0], size_needed);
    return wstrTo;
}

std::string ConvertToUTF8(const std::wstring& wstr) {
    if (wstr.empty()) return std::string();
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), NULL, 0, NULL, NULL);
    std::string strTo(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), &strTo[0], size_needed, NULL, NULL);
    return strTo;
}

void setLocale() {
    std::setlocale(LC_ALL, ".UTF-8");
}

// 日志相关实现
void initializeLogFile() {
    char buffer[MAX_PATH];
    GetModuleFileNameA(NULL, buffer, MAX_PATH);
    fs::path exePath(buffer);
    logFilePath = (exePath.parent_path() / "face_recognition_log.txt").string();
    std::ofstream logfile(logFilePath, std::ios::trunc);
    logfile.close();
}

void log(const std::string& file, int line, const std::string& message) {
    std::lock_guard<std::mutex> lock(logMutex);
    
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    std::tm tm;
    localtime_s(&tm, &in_time_t);
    ss << std::put_time(&tm, "%Y-%m-%d %X");
    
    std::string filename = fs::path(file).filename().string();
    std::string log_message = "[" + ss.str() + "] [" + filename + ":" + 
                             std::to_string(line) + "]: " + message;
    
    std::cout << log_message << std::endl;
    static std::ofstream logfile(logFilePath, std::ios::app);
    logfile << log_message << std::endl;
    logfile.flush();
}

// ONNX Runtime 相关实现
OrtEnv* getGlobalOrtEnv() {
    return OrtEnvironment::getInstance().getEnv();
}

// 数学工具函数实现
float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float iou(const Boxf& a, const Boxf& b) {
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    
    float xx1 = std::max(a.x1, b.x1);
    float yy1 = std::max(a.y1, b.y1);
    float xx2 = std::min(a.x2, b.x2);
    float yy2 = std::min(a.y2, b.y2);
    
    float w = std::max(0.0f, xx2 - xx1);
    float h = std::max(0.0f, yy2 - yy1);
    
    float inter = w * h;
    return inter / (area_a + area_b - inter);
}

// 图像处理工具实现
cv::Mat loadImage(const std::string& image_path) {
    try {
        // 1. 规范化路径
        std::string normalized_path = image_path;
        std::replace(normalized_path.begin(), normalized_path.end(), '\\', '/');
        LOG("加载图像: " + normalized_path);

        // 2. 使用 Windows API 读取文件
        std::wstring wide_path = ConvertToWideString(normalized_path);
        HANDLE hFile = CreateFileW(wide_path.c_str(), 
                                     GENERIC_READ, 
                                     FILE_SHARE_READ, 
                                     NULL, 
                                     OPEN_EXISTING, 
                                     FILE_ATTRIBUTE_NORMAL, 
                                     NULL);
        
        if (hFile == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("Failed to open file: " + normalized_path);
        }

        // 3. 读取文件内容
        DWORD fileSize = GetFileSize(hFile, NULL);
        std::vector<BYTE> buffer(fileSize);
        DWORD bytesRead;
        
        if (!ReadFile(hFile, buffer.data(), fileSize, &bytesRead, NULL)) {
            CloseHandle(hFile);
            throw std::runtime_error("Failed to read file: " + normalized_path);
        }
        CloseHandle(hFile);

        // 4. 从内存解码图像
        cv::Mat data(1, fileSize, CV_8UC1, buffer.data());
        cv::Mat image = cv::imdecode(data, cv::IMREAD_COLOR);
        
        if (image.empty()) {
            throw std::runtime_error("Failed to decode image: " + normalized_path);
        }

        LOG("图像加载成功，尺寸: " + std::to_string(image.cols) + "x" + std::to_string(image.rows));
        return image;
    } catch (const std::exception& e) {
        LOG("图像加载失败: " + std::string(e.what()));
        throw;
    }
}

cv::Mat processLargeImage(const cv::Mat& image) {
    if (image.cols > 2000 || image.rows > 2000) {
        cv::Mat resized;
        double scale = 2000.0 / std::max(image.cols, image.rows);
        cv::resize(image, resized, cv::Size(), scale, scale, 
                  image.size().area() > 4000000 ? cv::INTER_AREA : cv::INTER_LINEAR);
        return resized;
    }
    return image;
}

// Timer 实现
Timer::Timer(const std::string& name) : m_name(name) {
    m_start = std::chrono::high_resolution_clock::now();
}

Timer::~Timer() {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - m_start).count();
    LOG(m_name + " 耗时: " + std::to_string(duration) + "ms");
}

// NMS 实现
void nms_bboxes_kps(std::vector<BoxfWithLandmarks>& input,
                    std::vector<BoxfWithLandmarks>& output,
                    float iou_threshold, 
                    unsigned int topk) {
    std::sort(input.begin(), input.end(), 
              [](const BoxfWithLandmarks& a, const BoxfWithLandmarks& b) {
                  return a.box.score > b.box.score;
              });
    
    std::vector<bool> is_merged(input.size(), false);
    
    for (size_t i = 0; i < input.size() && output.size() < topk; ++i) {
        if (is_merged[i]) continue;
        
        output.push_back(input[i]);
        
        for (size_t j = i + 1; j < input.size(); ++j) {
            if (is_merged[j]) continue;
            
            if (iou(input[i].box, input[j].box) > iou_threshold) {
                is_merged[j] = true;
            }
        }
    }
}

// 添加路径规范化函数
std::string normalizePath(const std::string& path) {
    std::string normalized = path;
    std::replace(normalized.begin(), normalized.end(), '\\', '/');
    return normalized;
}

} // namespace face_recognition
