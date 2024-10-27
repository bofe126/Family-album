#include <onnxruntime_c_api.h>
#define NOMINMAX
#include <windows.h>
#include <algorithm>
// OpenCV headers
#include <opencv2/opencv.hpp>

// STL headers
#include <vector>
#include <iostream>
#include <filesystem>
#include <string>
#include <omp.h>
#include <memory>
#include <stdexcept>
#include <codecvt>
#include <locale>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <queue>
#include <thread>
#include <functional>
#include <condition_variable>

// Local headers
#include "Include/half.hpp"

// 2. 前向声明
class FaceDetector;
class ArcFaceExtractor;

// 3. 基础结构体定义
struct ScaleParams {
    float ratio;
    float dw;
    float dh;
    bool flag;
    
    ScaleParams() : ratio(1.0f), dw(0.0f), dh(0.0f), flag(false) {}
};

struct Boxf {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
    std::string label_text;
    bool flag;
};

struct Landmarks {
    std::vector<cv::Point2f> points;
    bool flag;
};

struct BoxfWithLandmarks {
    Boxf box;
    Landmarks landmarks;
    bool flag;
};

struct DetectionResult {
    int num_faces;
    std::vector<cv::Rect> faces;
    std::vector<std::vector<uint8_t>> face_data;
    std::vector<std::vector<float>> face_features;

    DetectionResult() : num_faces(0) {}
};

// 4. 类定义
class FaceDetector {
public:
    struct Config {
        int input_width = 640;
        int input_height = 640;
        float score_threshold = 0.5f;
        float iou_threshold = 0.45f;
        bool enable_landmarks = true;
        float scale_factor = 1.0f;
        bool use_letterbox = true;
        
        Config() = default;
    };

    FaceDetector(const char* modelPath, ModelType type = ModelType::YOLOV5, const Config& config = Config());
    ~FaceDetector();

    std::vector<BoxfWithLandmarks> detect(const cv::Mat& image, float score_threshold);

private:
    OrtSession* m_session;
    ModelType m_type;
    std::string m_output_node_name;
    std::string m_input_name;
    Config m_config;
    ScaleParams m_scale_params;

    cv::Mat preprocess(const cv::Mat& image);
};

// 5. 实现文件中的全局变量和函数
namespace {
    std::string logFilePath;
    std::mutex logMutex;
    OrtEnv* g_ort_env = nullptr;
    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
}

// 6. 工具函数声明
std::wstring ConvertToWideString(const std::string& str);
std::string ConvertToUTF8(const std::wstring& wstr);
void setLocale();
void initializeLogFile();
void log(const std::string& file, int line, const std::string& message);
OrtEnv* getGlobalOrtEnv();
cv::Mat loadImage(const std::string& image_path);

// 7. 主要功能函数声明
DetectionResult detect_faces_impl(const std::string& image_path, 
                                FaceDetector& detector, 
                                ArcFaceExtractor& arcface_extractor, 
                                int max_faces, 
                                float score_threshold);

// 8. 导出函数声明
extern "C" {
    __declspec(dllexport) int detect_faces(const char* image_path, 
                                         int* faces, 
                                         uint8_t** face_data, 
                                         int* face_data_sizes, 
                                         float** face_features, 
                                         int max_faces, 
                                         float score_threshold);

    __declspec(dllexport) float compare_faces(float* features1, 
                                            float* features2, 
                                            int feature_size);

    __declspec(dllexport) void cleanup_detection(uint8_t** face_data, 
                                               float** face_features, 
                                               int num_faces);
}

// 9. 实现部分
// ... (其余实现代码保持不变)

// 添加类型别名定义
using SizeType = int;  // 用于表示大小的类型
using CoordType = int; // 用于表示坐标的类型

// 函数声明
std::wstring ConvertToWideString(const std::string& multibyteStr);
void setLocale();
std::string ConvertToUTF8(const std::wstring& wstr);

enum class ModelType {
    YOLOV5,
    RETINAFACE
};

namespace fs = std::filesystem;

// 全局变量的声明和初始化
namespace {
    std::string logFilePath;  // 日志文件路径
    std::mutex logMutex;      // 日志互斥锁
    OrtEnv* g_ort_env = nullptr;
    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
}

void initializeLogFile() {
    char buffer[MAX_PATH];
    GetModuleFileNameA(NULL, buffer, MAX_PATH);
    std::filesystem::path exePath(buffer);
    logFilePath = (exePath.parent_path() / "face_recognition_log.txt").string();
    std::ofstream logfile(logFilePath, std::ios::trunc);
    logfile.close();
}

// 添加日志宏定义以自动传递文件名和行号
#define LOG(message) log(__FILE__, __LINE__, message)

void log(const std::string& file, int line, const std::string& message) {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    std::tm tm;
    localtime_s(&tm, &in_time_t);
    ss << std::put_time(&tm, "%Y-%m-%d %X");
    
    // 使用 std::filesystem 提取文件名
    std::string filename = fs::path(file).filename().string();
    
    // 在日志中添加文名行号
    std::cout << "[" << ss.str() << "] [" << filename << ":" << line << "]: " << message << std::endl;
    
    static std::ofstream logfile(logFilePath, std::ios::app);
    logfile << "[" << ss.str() << "] [" << filename << ":" << line << "]: " << message << std::endl;
    logfile.flush(); // 确保立即写入文件
}

// 局 OrtEnv
OrtEnv* getGlobalOrtEnv() {
    if (g_ort_env == nullptr) {
        OrtStatus* status = g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "GlobalOrtEnv", &g_ort_env);
        if (status != nullptr) {
            const char* error_message = g_ort->GetErrorMessage(status);
            g_ort->ReleaseStatus(status);
            throw std::runtime_error(std::string("Failed to create OrtEnv: ") + error_message);
        }
    }
    return g_ort_env;
}

// 添加 sigmoid 函数
float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// 计算两个边界框的 IOU
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

// 添加资源监控类
class ResourceMonitor {
public:
    static ResourceMonitor& getInstance() {
        static ResourceMonitor instance;
        return instance;
    }

    void startMonitoring() {
        LOG("开始资源监控");
    }

    void stopMonitoring() {
        LOG("停止资源监控");
    }

private:
    ResourceMonitor() {} // 私有构造函数
};

// NMS 函数实现
void nms_bboxes_kps(std::vector<BoxfWithLandmarks>& input,
                    std::vector<BoxfWithLandmarks>& output,
                    float iou_threshold, unsigned int topk)
{
    std::sort(input.begin(), input.end(), [](const BoxfWithLandmarks& a, const BoxfWithLandmarks& b) {
        return a.box.score > b.box.score;
    });

    std::vector<bool> is_merged(input.size(), false);

    unsigned int count = 0;
    for (unsigned int i = 0; i < input.size(); ++i)
    {
        if (is_merged[i]) continue;

        output.push_back(input[i]);

        for (size_t j = i + 1; j < input.size(); ++j)
        {
            if (is_merged[j]) continue;

            if (iou(input[i].box, input[j].box) > iou_threshold)
            {
                is_merged[j] = true;
            }
        }

        // 添加量限制
        count += 1;
        if (count >= topk || count >= 1000)  // 设置一个合理的上限，如1000
            break;
    }
}

// 1. 在文开头添加ScaleParams构体
struct ScaleParams {
    float ratio;  // 缩放例
    float dw;     // x方向padding
    float dh;     // y方向padding
    bool flag;
    
    ScaleParams() : ratio(1.0f), dw(0.0f), dh(0.0f), flag(false) {}
};

// 在文件开头添加
class Timer {
public:
    Timer(const std::string& name) : m_name(name) {
        m_start = std::chrono::high_resolution_clock::now();
    }
    
    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - m_start).count();
        LOG(m_name + " 耗时: " + std::to_string(duration) + "ms");
    }

private:
    std::string m_name;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
};

// 1. 添加 ArcFaceExtractor 类定义（在 FaceDetector 类之前）
class ArcFaceExtractor {
public:
    ArcFaceExtractor(const char* modelPath) {
        try {
            LOG("初始化 ArcFaceExtractor");
            LOG("模型路径: " + std::string(modelPath));
            
            OrtEnv* env = getGlobalOrtEnv();
            
            // 创建会话选项
            OrtSessionOptions* session_options;
            OrtStatus* status = g_ort->CreateSessionOptions(&session_options);
            if (status != nullptr) {
                const char* error_message = g_ort->GetErrorMessage(status);
                g_ort->ReleaseStatus(status);
                throw std::runtime_error(std::string("Failed to create session options: ") + error_message);
            }
            
            // 设置线程数
            g_ort->SetIntraOpNumThreads(session_options, 1);
            g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_EXTENDED);
            
            // 创建会话
            std::wstring wideModelPath = ConvertToWideString(modelPath);
            status = g_ort->CreateSession(env, wideModelPath.c_str(), session_options, &m_session);
            if (status != nullptr) {
                const char* error_message = g_ort->GetErrorMessage(status);
                g_ort->ReleaseStatus(status);
                g_ort->ReleaseSessionOptions(session_options);
                throw std::runtime_error(std::string("Failed to create session: ") + error_message);
            }
            
            g_ort->ReleaseSessionOptions(session_options);
            LOG("ArcFaceExtractor 初始化成功");
            
        } catch (const std::exception& e) {
            LOG("ArcFaceExtractor 初始化失败: " + std::string(e.what()));
            throw;
        }
    }
    
    ~ArcFaceExtractor() {
        if (m_session != nullptr) {
            g_ort->ReleaseSession(m_session);
            m_session = nullptr;
        }
    }
    
    std::vector<float> extract(const cv::Mat& face_image) {
        try {
            // TODO: 实现特征提取
            // 临时返回空向量，后续实现实际的特征提取
            return std::vector<float>(512, 0.0f);
        } catch (const std::exception& e) {
            LOG("特征提取错误: " + std::string(e.what()));
            throw;
        }
    }

private:
    OrtSession* m_session = nullptr;
};

// 添加 DetectionResult 结构体定义
struct DetectionResult {
    int num_faces;
    std::vector<cv::Rect> faces;
    std::vector<std::vector<uint8_t>> face_data;
    std::vector<std::vector<float>> face_features;

    DetectionResult() : num_faces(0) {}
};

// 2. 添加函数前向声明
cv::Mat loadImage(const std::string& image_path);
DetectionResult detect_faces_impl(const std::string& image_path, 
                                FaceDetector& detector, 
                                ArcFaceExtractor& arcface_extractor, 
                                int max_faces, 
                                float score_threshold);

// 3. 修复 std::copy 的使用
extern "C" __declspec(dllexport) int detect_faces(const char* image_path, 
                                                 int* faces, 
                                                 uint8_t** face_data, 
                                                 int* face_data_sizes, 
                                                 float** face_features, 
                                                 int max_faces, 
                                                 float score_threshold) {
    try {
        // 检查模型文件是否存在
        fs::path yolo_model_path = fs::absolute("assets/yolov5s-face.onnx");
        fs::path arcface_model_path = fs::absolute("assets/arcface_model.onnx");
        
        LOG("YOLO模型路径: " + yolo_model_path.string());
        LOG("ArcFace模型路径: " + arcface_model_path.string());
        
        if (!fs::exists(yolo_model_path) || !fs::exists(arcface_model_path)) {
            LOG("错误：模型文件不存在");
            return -1;
        }

        // 创建检测器实例
        static FaceDetector detector(yolo_model_path.string().c_str(), ModelType::YOLOV5);
        static ArcFaceExtractor arcface_extractor(arcface_model_path.string().c_str());

        // 调用impl函数进行检测
        DetectionResult result = detect_faces_impl(image_path, detector, arcface_extractor, max_faces, score_threshold);

        // 修复 std::copy 的使用
        for (int i = 0; i < result.num_faces; i++) {
            // 复制边界框
            faces[i * 4] = result.faces[i].x;
            faces[i * 4 + 1] = result.faces[i].y;
            faces[i * 4 + 2] = result.faces[i].width;
            faces[i * 4 + 3] = result.faces[i].height;

            // 复制人脸数据
            face_data[i] = new uint8_t[result.face_data[i].size()];
            std::copy(result.face_data[i].begin(), 
                     result.face_data[i].end(), 
                     face_data[i]);  // 修复 std::copy 参数
            face_data_sizes[i] = static_cast<int>(result.face_data[i].size());

            // 复制特征向量
            face_features[i] = new float[result.face_features[i].size()];
            std::copy(result.face_features[i].begin(), 
                     result.face_features[i].end(), 
                     face_features[i]);  // 修复 std::copy 参数
        }

        LOG("detect_faces 函数执行完成，处理了 " + std::to_string(result.num_faces) + " 个人脸");
        return result.num_faces;
    } catch (const std::exception& e) {
        LOG("检测失败: " + std::string(e.what()));
        return -1;
    }
}

extern "C" __declspec(dllexport) float compare_faces(float* features1, 
                                                    float* features2, 
                                                    int feature_size) {
    try {
        LOG("开始比对人脸特征");
        
        // 计算余弦相似度
        float dot_product = 0.0f;
        float norm1 = 0.0f;
        float norm2 = 0.0f;
        
        for (int i = 0; i < feature_size; ++i) {
            dot_product += features1[i] * features2[i];
            norm1 += features1[i] * features1[i];
            norm2 += features2[i] * features2[i];
        }
        
        norm1 = std::sqrt(norm1);
        norm2 = std::sqrt(norm2);
        
        float similarity = dot_product / (norm1 * norm2);
        
        LOG("特征比对完成，相似度: " + std::to_string(similarity));
        return similarity;
    } catch (const std::exception& e) {
        LOG("特征比对错误: " + std::string(e.what()));
        return -1.0f;
    }
}

// 添加资源清理函数
extern "C" __declspec(dllexport) void cleanup_detection(uint8_t** face_data, 
                                                       float** face_features, 
                                                       int num_faces) {
    try {
        LOG("开始清理资源");
        
        if (face_data != nullptr) {
            for (int i = 0; i < num_faces; ++i) {
                if (face_data[i] != nullptr) {
                    delete[] face_data[i];
                    face_data[i] = nullptr;
                }
            }
        }
        
        if (face_features != nullptr) {
            for (int i = 0; i < num_faces; ++i) {
                if (face_features[i] != nullptr) {
                    delete[] face_features[i];
                    face_features[i] = nullptr;
                }
            }
        }
        
        LOG("资源清理完成");
    } catch (const std::exception& e) {
        LOG("资源清理错误: " + std::string(e.what()));
    }
}

// 函数定义
std::wstring ConvertToWideString(const std::string& str) {
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), NULL, 0);
    std::wstring wstrTo(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), &wstrTo[0], size_needed);
    return wstrTo;
}

// 修改 DllMain 函数
BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) {
    switch (ul_reason_for_call) {
    case DLL_PROCESS_ATTACH:
        initializeLogFile();
        setLocale();
        SetConsoleOutputCP(CP_UTF8);
        LOG("DLL_PROCESS_ATTACH");
        ResourceMonitor::getInstance().startMonitoring();
        break;
    case DLL_PROCESS_DETACH:
        ResourceMonitor::getInstance().stopMonitoring();
        LOG("DLL_PROCESS_DETACH");
        break;
    }
    return TRUE;
}

// 在文件末尾添加 setLocale 函数定义
void setLocale() {
    std::setlocale(LC_ALL, ".UTF-8");  // 使用UTF-8编码
}

std::string ConvertToUTF8(const std::wstring& wstr) {
    if (wstr.empty()) return std::string();        
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), NULL, 0, NULL, NULL);
    std::string strTo(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), &strTo[0], size_needed, NULL, NULL);
    return strTo;
}

// 在文件开头添加内存池类
class MemoryPool {
public:
    static MemoryPool& getInstance() {
        static MemoryPool instance;
        return instance;
    }

    void* allocate(size_t size) {
        std::lock_guard<std::mutex> lock(m_mutex);
        void* ptr = malloc(size);
        m_allocations[ptr] = size;
        m_totalMemory += size;
        LOG("内存分配: " + std::to_string(size) + " bytes, 总使用: " + 
            std::to_string(m_totalMemory) + " bytes");
        return ptr;
    }

    void deallocate(void* ptr) {
        if (!ptr) return;
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_allocations.find(ptr);
        if (it != m_allocations.end()) {
            m_totalMemory -= it->second;
            m_allocations.erase(it);
            free(ptr);
            LOG("内存释放: " + std::to_string(it->second) + " bytes, 剩余使用: " + 
                std::to_string(m_totalMemory) + " bytes");
        }
    }

private:
    size_t m_totalMemory;
};

// 修改内存分配相关代码
void* allocateMemory(size_t size) {
    return MemoryPool::getInstance().allocate(size);
}

void freeMemory(void* ptr) {
    MemoryPool::getInstance().deallocate(ptr);
}

// 添加模型缓存类
class ModelCache {
public:
    static ModelCache& getInstance() {
        static ModelCache instance;
        return instance;
    }

    OrtSession* getSession(const std::string& modelPath, const std::string& modelType) {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto key = modelPath + "_" + modelType;
        auto it = m_sessions.find(key);
        if (it != m_sessions.end()) {
            LOG("从缓存获取模型会话: " + modelType);
            return it->second;
        }
        return nullptr;
    }

    void addSession(const std::string& modelPath, const std::string& modelType, OrtSession* session) {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto key = modelPath + "_" + modelType;
        m_sessions[key] = session;
        LOG("添加模型会话到缓: " + modelType);
    }

    void clear() {
        std::lock_guard<std::mutex> lock(m_mutex);
        for (auto& pair : m_sessions) {
            if (pair.second) {
                g_ort->ReleaseSession(pair.second);
            }
        }
        m_sessions.clear();
        LOG("清理所有模型会话缓存");
    }

private:
    ModelCache() {}
    std::mutex m_mutex;
    std::unordered_map<std::string, OrtSession*> m_sessions;
};

// 在FaceDetector和ArcFaceExtractor构造函数中使用缓存

// 添加性能分析类
class PerformanceAnalyzer {
public:
    static PerformanceAnalyzer& getInstance() {
        static PerformanceAnalyzer instance;
        return instance;
    }

    void beginSection(const std::string& name) {
        auto& section = m_sections[name];
        section.start = std::chrono::high_resolution_clock::now();
    }

    void endSection(const std::string& name) {
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

private:
    struct Section {
        std::chrono::time_point<std::chrono::high_resolution_clock> start;
        int64_t total_time = 0;
        int64_t min_time = std::numeric_limits<int64_t>::max();
        int64_t max_time = 0;
        int count = 0;
    };

    std::unordered_map<std::string, Section> m_sections;
};

// 添加性能分析宏
#define BEGIN_PROFILE(name) PerformanceAnalyzer::getInstance().beginSection(name)
#define END_PROFILE(name) PerformanceAnalyzer::getInstance().endSection(name)

// 添加线程池类
class ThreadPool {
public:
    static ThreadPool& getInstance() {
        static ThreadPool instance;
        return instance;
    }

    ThreadPool(size_t threads = std::thread::hardware_concurrency()) {
        for(size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this] {
                while(true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        condition.wait(lock, [this] { 
                            return stop || !tasks.empty(); 
                        });
                        
                        if(stop && tasks.empty()) return;
                        
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
        }
        LOG("线程池初始化完成，线程数: " + std::to_string(threads));
    }

    template<class F>
    void enqueue(F&& f) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            tasks.emplace(std::forward<F>(f));
        }
        condition.notify_one();
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for(std::thread &worker: workers) {
            worker.join();
        }
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop = false;
};

// 添加图像金字塔处理大图像
cv::Mat processLargeImage(const cv::Mat& image) {
    if (image.cols > 2000 || image.rows > 2000) {
        cv::Mat resized;
        double scale = 2000.0 / std::max(image.cols, image.rows);
        
        // 使用 INTER_AREA 进行下采样，可能会得到更好的结果
        cv::resize(image, resized, cv::Size(), scale, scale, 
                  image.size().area() > 4000000 ? cv::INTER_AREA : cv::INTER_LINEAR);
        
        LOG("图像缩放: " + std::to_string(image.cols) + "x" + std::to_string(image.rows) + 
            " -> " + std::to_string(resized.cols) + "x" + std::to_string(resized.rows));
        return resized;
    }
    return image;
}

// 添加OpenCV并行后端错误处理
void initializeOpenCV() {
    try {
        cv::setNumThreads(4);  // 设置OpenCV线程数
        
        // 检查 OpenCL 支持
        if (cv::ocl::haveOpenCL()) {
            cv::ocl::setUseOpenCL(true);
            LOG("OpenCL 后端可用并已启用");
        } else {
            LOG("OpenCL 后端不可用");
        }
        
        // 获取 OpenCV 构建信息
        std::string buildInfo = cv::getBuildInformation();
        LOG("OpenCV 构建信息:");
        std::istringstream iss(buildInfo);
        std::string line;
        while (std::getline(iss, line)) {
            if (line.find("CPU") != std::string::npos || 
                line.find("OpenCL") != std::string::npos || 
                line.find("Threading") != std::string::npos) {
                LOG(line);
            }
        }
        
        // 检查可用的 CPU 核心数
        int numCores = cv::getNumberOfCPUs();
        int maxThreads = cv::getNumThreads();
        LOG("CPU 核心数: " + std::to_string(numCores));
        LOG("OpenCV 最大线程数: " + std::to_string(maxThreads));
        
    } catch (const cv::Exception& e) {
        LOG("OpenCV 初始化警告: " + std::string(e.what()));
    }
}

extern "C" __declspec(dllexport) float compare_faces(float* features1, 
                                                    float* features2, 
                                                    int feature_size) {
    try {
        LOG("开始比对人脸特征");
        
        // 计算余弦相似度
        float dot_product = 0.0f;
        float norm1 = 0.0f;
        float norm2 = 0.0f;
        
        for (int i = 0; i < feature_size; ++i) {
            dot_product += features1[i] * features2[i];
            norm1 += features1[i] * features1[i];
            norm2 += features2[i] * features2[i];
        }
        
        norm1 = std::sqrt(norm1);
        norm2 = std::sqrt(norm2);
        
        float similarity = dot_product / (norm1 * norm2);
        
        LOG("特征比对完成，相似度: " + std::to_string(similarity));
        return similarity;
    } catch (const std::exception& e) {
        LOG("特征比对错误: " + std::string(e.what()));
        return -1.0f;
    }
}

// 添加资源清理函数
extern "C" __declspec(dllexport) void cleanup_detection(uint8_t** face_data, 
                                                       float** face_features, 
                                                       int num_faces) {
    try {
        LOG("开始清理资源");
        
        if (face_data != nullptr) {
            for (int i = 0; i < num_faces; ++i) {
                if (face_data[i] != nullptr) {
                    delete[] face_data[i];
                    face_data[i] = nullptr;
                }
            }
        }
        
        if (face_features != nullptr) {
            for (int i = 0; i < num_faces; ++i) {
                if (face_features[i] != nullptr) {
                    delete[] face_features[i];
                    face_features[i] = nullptr;
                }
            }
        }
        
        LOG("资源清理完成");
    } catch (const std::exception& e) {
        LOG("资源清理错误: " + std::string(e.what()));
    }
}

// 函数定义
std::wstring ConvertToWideString(const std::string& str) {
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), NULL, 0);
    std::wstring wstrTo(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), &wstrTo[0], size_needed);
    return wstrTo;
}

// 修改 DllMain 函数
BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) {
    switch (ul_reason_for_call) {
    case DLL_PROCESS_ATTACH:
        initializeLogFile();
        setLocale();
        SetConsoleOutputCP(CP_UTF8);
        LOG("DLL_PROCESS_ATTACH");
        ResourceMonitor::getInstance().startMonitoring();
        break;
    case DLL_PROCESS_DETACH:
        ResourceMonitor::getInstance().stopMonitoring();
        LOG("DLL_PROCESS_DETACH");
        break;
    }
    return TRUE;
}

// 在文件末尾添加 setLocale 函数定义
void setLocale() {
    std::setlocale(LC_ALL, ".UTF-8");  // 使用UTF-8编码
}

std::string ConvertToUTF8(const std::wstring& wstr) {
    if (wstr.empty()) return std::string();        
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), NULL, 0, NULL, NULL);
    std::string strTo(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), &strTo[0], size_needed, NULL, NULL);
    return strTo;
}

// 在文件开头添加内存池类
class MemoryPool {
public:
    static MemoryPool& getInstance() {
        static MemoryPool instance;
        return instance;
    }

    void* allocate(size_t size) {
        std::lock_guard<std::mutex> lock(m_mutex);
        void* ptr = malloc(size);
        m_allocations[ptr] = size;
        m_totalMemory += size;
        LOG("内存分配: " + std::to_string(size) + " bytes, 总使用: " + 
            std::to_string(m_totalMemory) + " bytes");
        return ptr;
    }

    void deallocate(void* ptr) {
        if (!ptr) return;
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_allocations.find(ptr);
        if (it != m_allocations.end()) {
            m_totalMemory -= it->second;
            m_allocations.erase(it);
            free(ptr);
            LOG("内存释放: " + std::to_string(it->second) + " bytes, 剩余使用: " + 
                std::to_string(m_totalMemory) + " bytes");
        }
    }

private:
    MemoryPool() : m_totalMemory(0) {}
    std::mutex m_mutex;
    std::unordered_map<void*, size_t> m_allocations;
    size_t m_totalMemory;
};

// 修改内存分配相关代码
void* allocateMemory(size_t size) {
    return MemoryPool::getInstance().allocate(size);
}

void freeMemory(void* ptr) {
    MemoryPool::getInstance().deallocate(ptr);
}

// 添加模型缓存类
class ModelCache {
public:
    static ModelCache& getInstance() {
        static ModelCache instance;
        return instance;
    }

    OrtSession* getSession(const std::string& modelPath, const std::string& modelType) {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto key = modelPath + "_" + modelType;
        auto it = m_sessions.find(key);
        if (it != m_sessions.end()) {
            LOG("从缓存获取模型会话: " + modelType);
            return it->second;
        }
        return nullptr;
    }

    void addSession(const std::string& modelPath, const std::string& modelType, OrtSession* session) {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto key = modelPath + "_" + modelType;
        m_sessions[key] = session;
        LOG("添加模型会话到缓: " + modelType);
    }

    void clear() {
        std::lock_guard<std::mutex> lock(m_mutex);
        for (auto& pair : m_sessions) {
            if (pair.second) {
                g_ort->ReleaseSession(pair.second);
            }
        }
        m_sessions.clear();
        LOG("清理所有模型会话缓存");
    }

private:
    ModelCache() {}
    std::mutex m_mutex;
    std::unordered_map<std::string, OrtSession*> m_sessions;
};

// 在FaceDetector和ArcFaceExtractor构造函数中使用缓存

// 添加性能分析类
class PerformanceAnalyzer {
public:
    static PerformanceAnalyzer& getInstance() {
        static PerformanceAnalyzer instance;
        return instance;
    }

    void beginSection(const std::string& name) {
        auto& section = m_sections[name];
        section.start = std::chrono::high_resolution_clock::now();
    }

    void endSection(const std::string& name) {
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

private:
    struct Section {
        std::chrono::time_point<std::chrono::high_resolution_clock> start;
        int64_t total_time = 0;
        int64_t min_time = std::numeric_limits<int64_t>::max();
        int64_t max_time = 0;
        int count = 0;
    };

    std::unordered_map<std::string, Section> m_sections;
};

// 添加性能分析宏
#define BEGIN_PROFILE(name) PerformanceAnalyzer::getInstance().beginSection(name)
#define END_PROFILE(name) PerformanceAnalyzer::getInstance().endSection(name)

// 添加线程池类
class ThreadPool {
public:
    static ThreadPool& getInstance() {
        static ThreadPool instance;
        return instance;
    }

    ThreadPool(size_t threads = std::thread::hardware_concurrency()) {
        for(size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this] {
                while(true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        condition.wait(lock, [this] { 
                            return stop || !tasks.empty(); 
                        });
                        
                        if(stop && tasks.empty()) return;
                        
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
        }
        LOG("线程池初始化完成，线程数: " + std::to_string(threads));
    }

    template<class F>
    void enqueue(F&& f) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            tasks.emplace(std::forward<F>(f));
        }
        condition.notify_one();
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for(std::thread &worker: workers) {
            worker.join();
        }
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop = false;
};

// 添加图像金字塔处理大图像
cv::Mat processLargeImage(const cv::Mat& image) {
    if (image.cols > 2000 || image.rows > 2000) {
        cv::Mat resized;
        double scale = 2000.0 / std::max(image.cols, image.rows);
        
        // 使用 INTER_AREA 进行下采样，可能会得到更好的结果
        cv::resize(image, resized, cv::Size(), scale, scale, 
                  image.size().area() > 4000000 ? cv::INTER_AREA : cv::INTER_LINEAR);
        
        LOG("图像缩放: " + std::to_string(image.cols) + "x" + std::to_string(image.rows) + 
            " -> " + std::to_string(resized.cols) + "x" + std::to_string(resized.rows));
        return resized;
    }
    return image;
}

// 添加OpenCV并行后端错误处理
void initializeOpenCV() {
    try {
        cv::setNumThreads(4);  // 设置OpenCV线程数
        
        // 检查 OpenCL 支持
        if (cv::ocl::haveOpenCL()) {
            cv::ocl::setUseOpenCL(true);
            LOG("OpenCL 后端可用并已启用");
        } else {
            LOG("OpenCL 后端不可用");
        }
        
        // 获取 OpenCV 构建信息
        std::string buildInfo = cv::getBuildInformation();
        LOG("OpenCV 构建信息:");
        std::istringstream iss(buildInfo);
        std::string line;
        while (std::getline(iss, line)) {
            if (line.find("CPU") != std::string::npos || 
                line.find("OpenCL") != std::string::npos || 
                line.find("Threading") != std::string::npos) {
                LOG(line);
            }
        }
        
        // 检查可用的 CPU 核心数
        int numCores = cv::getNumberOfCPUs();
        int maxThreads = cv::getNumThreads();
        LOG("CPU 核心数: " + std::to_string(numCores));
        LOG("OpenCV 最大线程数: " + std::to_string(maxThreads));
        
    } catch (const cv::Exception& e) {
        LOG("OpenCV 初始化警告: " + std::string(e.what()));
    }
}
