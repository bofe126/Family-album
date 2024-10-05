#include "onnxruntime_cxx_api.h"
#define NOMINMAX
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <filesystem>
#include <string>
#include <windows.h>
#include <omp.h>
#include <memory>
#include <stdexcept>
#include <codecvt>
#include <locale>
#include <fstream>
#include <chrono>
#include <iomanip>

// 函数声明
std::wstring ConvertToWideString(const std::string& multibyteStr);

enum class ModelType {
    YOLOV5,
    RETINAFACE
};

namespace fs = std::filesystem;

// 在全局范围或在某个初始化函数中
HMODULE onnxruntimeModule = NULL;
bool isDllLoaded = false;

// 全局变量来存储日志文件路径
std::string logFilePath;

void initializeLogFile() {
    // 获取当前执行文件的路径
    char buffer[MAX_PATH];
    GetModuleFileNameA(NULL, buffer, MAX_PATH);
    std::filesystem::path exePath(buffer);
    
    // 设置日志文件路径为执行文件所在目录
    logFilePath = (exePath.parent_path() / "face_recognition_log.txt").string();
    
    // 清空日志文件
    std::ofstream logfile(logFilePath, std::ios::trunc);
    logfile.close();
}

void log(const std::string& message) {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X");
    
    std::cout << "face_recognition.dll Log [" << ss.str() << "]: " << message << std::endl;
    
    static std::ofstream logfile(logFilePath, std::ios::app);
    logfile << "face_recognition.dll Log [" << ss.str() << "]: " << message << std::endl;
    logfile.flush(); // 确保立即写入文件
}

bool loadOnnxRuntime() {
    if (isDllLoaded) {
        log("onnxruntime.dll already loaded");
        return true;
    }

    onnxruntimeModule = LoadLibraryA("onnxruntime.dll");
    if (onnxruntimeModule == NULL) {
        DWORD error = GetLastError();
        log("Failed to load onnxruntime.dll. Error code: " + std::to_string(error));
        return false;
    }
    log("Successfully loaded onnxruntime.dll");

    // 尝试获取 OrtGetApiBase 函数
    auto getApiBase = (decltype(&OrtGetApiBase))GetProcAddress(onnxruntimeModule, "OrtGetApiBase");
    if (getApiBase == nullptr) {
        DWORD error = GetLastError();
        log("Failed to get OrtGetApiBase. Error code: " + std::to_string(error));
        return false;
    }
    log("Successfully got OrtGetApiBase");

    // 尝试调用 OrtGetApiBase
    const OrtApiBase* ortApiBase = getApiBase();
    if (ortApiBase == nullptr) {
        log("OrtGetApiBase returned nullptr");
        return false;
    }
    log("Successfully called OrtGetApiBase");

    isDllLoaded = true;
    return true;
}

BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) {
    switch (ul_reason_for_call) {
    case DLL_PROCESS_ATTACH:
        initializeLogFile(); // 初始化日志文件
        log("DLL_PROCESS_ATTACH");
        if (!loadOnnxRuntime()) {
            return FALSE;
        }
        log("onnxruntime.dll loaded successfully");
        break;
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
        break;
    case DLL_PROCESS_DETACH:
        log("DLL_PROCESS_DETACH");
        if (onnxruntimeModule != NULL) {
            FreeLibrary(onnxruntimeModule);
            log("onnxruntime.dll unloaded");
            isDllLoaded = false;
        }
        break;
    }
    return TRUE;
}

class FaceDetector {
public:
    FaceDetector(const char* modelPath, ModelType type) 
        : m_type(type) {
        try {
            log("进入FaceDetector构造函数");
            log("模型路径: " + std::string(modelPath));
            log("模型类型: " + std::to_string(static_cast<int>(type)));

            log("准备创建Ort::Env");
            try {
                env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "FaceDetector");
                log("Ort::Env创建成功");
            } catch (const Ort::Exception& e) {
                log("Ort::Env创建失败: " + std::string(e.what()));
                throw;
            } catch (const std::exception& e) {
                log("Ort::Env创建时发生标准异常: " + std::string(e.what()));
                throw;
            } catch (...) {
                log("Ort::Env创建时发生未知异常");
                throw;
            }

            log("初始化FaceDetector");
            fs::path fullPath = fs::absolute(modelPath);
            log("完整模型路径: " + fullPath.string());

            log("创建Ort::SessionOptions");
            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(1);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
            log("Ort::SessionOptions创建成功");

            log("转换模型路径为宽字符串");
            std::wstring wideModelPath = ConvertToWideString(fullPath.string());
            log("模型路径转换成功");

            log("创建Ort::Session");
            m_session = std::make_unique<Ort::Session>(env, wideModelPath.c_str(), session_options);
            log("Ort::Session创建成功");

            // 打印模型信息
            Ort::AllocatorWithDefaultOptions allocator;
            size_t num_input_nodes = m_session->GetInputCount();
            size_t num_output_nodes = m_session->GetOutputCount();

            log("输入节点数量: " + std::to_string(num_input_nodes));
            log("输出节点数量: " + std::to_string(num_output_nodes));

            for (size_t i = 0; i < num_input_nodes; i++) {
                auto input_name = m_session->GetInputNameAllocated(i, allocator);
                log("输入 " + std::to_string(i) + " 名称: " + input_name.get());
            }

            for (size_t i = 0; i < num_output_nodes; i++) {
                auto output_name = m_session->GetOutputNameAllocated(i, allocator);
                log("输出 " + std::to_string(i) + " 名称: " + output_name.get());
            }

            log("FaceDetector初始化成功");
        } catch (const Ort::Exception& e) {
            log("ONNX Runtime错误: " + std::string(e.what()));
            throw;
        } catch (const std::exception& e) {
            log("标准异常: " + std::string(e.what()));
            throw;
        } catch (...) {
            log("未知异常");
            throw;
        }
    }

    std::vector<cv::Rect> detect(const cv::Mat& image) {
        return detectYOLOV5(image);
    }

private:
    std::vector<cv::Rect> detectYOLOV5(const cv::Mat& image) {
        const int inputWidth = 640;
        const int inputHeight = 640;
        
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(inputWidth, inputHeight));
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

        std::vector<float> input_tensor(resized.total() * resized.channels());
        resized.convertTo(input_tensor, CV_32F, 1.0f / 255.0f);

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::array<int64_t, 4> input_shape = {1, 3, inputHeight, inputWidth};
        Ort::Value input_tensor_ort = Ort::Value::CreateTensor<float>(memory_info, input_tensor.data(), input_tensor.size(), input_shape.data(), input_shape.size());

        const char* input_names[] = {"images"};
        const char* output_names[] = {"output"};

        auto output_tensors = m_session->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor_ort, 1, output_names, 1);

        std::vector<cv::Rect> faces;
        processPredictions(output_tensors[0], image.cols, image.rows, faces);

        return faces;
    }

    void processPredictions(const Ort::Value& output_tensor, int orig_width, int orig_height, std::vector<cv::Rect>& faces) {
        Ort::TypeInfo type_info = output_tensor.GetTypeInfo();
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        auto shape = tensor_info.GetShape();

        const float* output_data = output_tensor.GetTensorData<float>();

        // YOLOv5 输出形状应该是 [1, num_boxes, 85]
        int64_t num_boxes = shape[1];

        float conf_threshold = 0.25f;  // 置信度阈值
        float iou_threshold = 0.45f;   // IOU阈值

        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;

        for (int64_t i = 0; i < num_boxes; ++i) {
            const float* row = &output_data[i * 85];
            float confidence = row[4];

            if (confidence >= conf_threshold) {
                float x = row[0];
                float y = row[1];
                float w = row[2];
                float h = row[3];

                int left = static_cast<int>((x - 0.5f * w) * static_cast<float>(orig_width));
                int top = static_cast<int>((y - 0.5f * h) * static_cast<float>(orig_height));
                int width = static_cast<int>(w * static_cast<float>(orig_width));
                int height = static_cast<int>(h * static_cast<float>(orig_height));

                boxes.push_back(cv::Rect(left, top, width, height));
                confidences.push_back(confidence);
            }
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, iou_threshold, indices);

        faces.clear();
        for (int idx : indices) {
            faces.push_back(boxes[idx]);
        }
    }

    Ort::Env env;
    std::unique_ptr<Ort::Session> m_session;
    ModelType m_type;
};

struct DetectionResult {
    int num_faces;
    std::vector<cv::Rect> faces;
    std::vector<std::vector<uint8_t>> face_data;
    std::vector<std::vector<float>> face_features;
};

class ArcFaceExtractor {
public:
    ArcFaceExtractor(const char* modelPath) {
        try {
            fs::path fullPath = fs::absolute(modelPath);
            std::cout << "正在加载 ArcFace 模型: " << fullPath.string() << std::endl;
            
            Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ArcFaceExtractor");
            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(1);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

            std::wstring wideModelPath = ConvertToWideString(fullPath.string());
            m_session = std::make_unique<Ort::Session>(env, wideModelPath.c_str(), session_options);

            // 打印模型信息
            Ort::AllocatorWithDefaultOptions allocator;
            size_t num_input_nodes = m_session->GetInputCount();
            size_t num_output_nodes = m_session->GetOutputCount();

            std::cout << "ArcFace 模型输入数量: " << num_input_nodes << std::endl;
            std::cout << "ArcFace 模型输出数量: " << num_output_nodes << std::endl;

            for (size_t i = 0; i < num_input_nodes; i++) {
                auto input_name = m_session->GetInputNameAllocated(i, allocator);
                std::cout << "输入 " << i << " 名称: " << input_name.get() << std::endl;
            }

            for (size_t i = 0; i < num_output_nodes; i++) {
                auto output_name = m_session->GetOutputNameAllocated(i, allocator);
                std::cout << "输出 " << i << " 名称: " << output_name.get() << std::endl;
            }

            std::cout << "ArcFace 模型加载成功" << std::endl;
        } catch (const Ort::Exception& e) {
            std::cerr << "ONNX Runtime error in ArcFaceExtractor: " << e.what() << std::endl;
            throw;
        } catch (const std::exception& e) {
            std::cerr << "Standard exception in ArcFaceExtractor: " << e.what() << std::endl;
            throw;
        } catch (...) {
            std::cerr << "Unknown exception in ArcFaceExtractor" << std::endl;
            throw;
        }
    }

    std::vector<float> extract(const cv::Mat& face_image) {
        cv::Mat resized_face;
        cv::resize(face_image, resized_face, cv::Size(112, 112));
        cv::cvtColor(resized_face, resized_face, cv::COLOR_BGR2RGB);
        
        std::vector<float> input_tensor(resized_face.total() * resized_face.channels());
        resized_face.convertTo(input_tensor, CV_32F, 1.0f / 255.0f);

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::array<int64_t, 4> input_shape = {1, 3, 112, 112};
        Ort::Value input_tensor_ort = Ort::Value::CreateTensor<float>(memory_info, input_tensor.data(), input_tensor.size(), input_shape.data(), input_shape.size());

        const char* input_names[] = {"input"};
        const char* output_names[] = {"output"};

        auto output_tensors = m_session->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor_ort, 1, output_names, 1);

        Ort::TypeInfo type_info = output_tensors[0].GetTypeInfo();
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        auto shape = tensor_info.GetShape();

        std::vector<float> output_vector(shape[0] * shape[1]);
        const float* output_data = output_tensors[0].GetTensorData<float>();
        std::copy(output_data, output_data + output_vector.size(), output_vector.begin());

        return output_vector;
    }

private:
    std::unique_ptr<Ort::Session> m_session;
};

cv::Mat loadImage(const std::string& filename) {
    std::cout << "正在尝试加载图像: " << filename << std::endl;

    HANDLE hFile = CreateFileA(filename.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        std::cerr << "CreateFileA failed. Error code: " << GetLastError() << std::endl;
        throw std::runtime_error("无法打开文件: " + filename);
    }

    DWORD fileSize = GetFileSize(hFile, NULL);
    if (fileSize == INVALID_FILE_SIZE) {
        CloseHandle(hFile);
        std::cerr << "GetFileSize failed. Error code: " << GetLastError() << std::endl;
        throw std::runtime_error("无法获取文件大小: " + filename);
    }

    std::vector<char> buffer(fileSize);
    DWORD bytesRead;
    if (!ReadFile(hFile, buffer.data(), fileSize, &bytesRead, NULL)) {
        CloseHandle(hFile);
        std::cerr << "ReadFile failed. Error code: " << GetLastError() << std::endl;
        throw std::runtime_error("无法读取文件: " + filename);
    }
    CloseHandle(hFile);

    cv::Mat data(1, fileSize, CV_8UC1, buffer.data());
    cv::Mat image = cv::imdecode(data, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "cv::imdecode failed to decode the image." << std::endl;
        throw std::runtime_error("无法解码图像: " + filename);
    }

    std::cout << "成功加载图像: " << filename << std::endl;
    return image;
}

DetectionResult detect_faces_impl(const std::string& image_path, const char* yolov5_model_path, const char* arcface_model_path, int max_faces) {
    try {
        std::wofstream logFile("face_recognition_log.txt", std::ios::app);
        std::wcout.rdbuf(logFile.rdbuf());

        std::cout << "YOLOV5 模型路径: " << yolov5_model_path << std::endl;
        std::cout << "ArcFace 模型路径: " << arcface_model_path << std::endl;

        // 检查模型文件是否存在
        if (!fs::exists(yolov5_model_path)) {
            std::cerr << "YOLOV5 模型文件不存在: " << yolov5_model_path << std::endl;
            //throw std::runtime_error("YOLOV5模型文件不存在");
        }
        if (!fs::exists(arcface_model_path)) {
            std::cerr << "ArcFace 模型文件不存在: " << arcface_model_path << std::endl;
            //throw std::runtime_error("ArcFace模型文件不存在");
        }

        cv::Mat image = loadImage(image_path);
        if (image.empty()) {
            std::string error_message = "无法加载图像: " + image_path;
            std::cerr << error_message << std::endl;
            throw std::runtime_error(error_message);
        }

        std::cout << "正在创建 FaceDetector..." << std::endl;
        static FaceDetector detector(yolov5_model_path, ModelType::YOLOV5);
        std::cout << "正在创建 ArcFaceExtractor..." << std::endl;
        static ArcFaceExtractor arcface_extractor(arcface_model_path);
        std::vector<cv::Rect> detected_faces = detector.detect(image);

        DetectionResult result;
        result.num_faces = std::min(static_cast<int>(detected_faces.size()), max_faces);

        #pragma omp parallel for
        for (int i = 0; i < result.num_faces; i++) {
            cv::Rect face = detected_faces[i];
            // 为了线程安全，使用局部变量并在结束时合并
            std::vector<uint8_t> local_face_data;
            std::vector<float> local_face_features;

            cv::Mat face_image = image(face);
            cv::imencode(".jpg", face_image, local_face_data);
            local_face_features = arcface_extractor.extract(face_image);

            // 保护共享资源的访问
            #pragma omp critical
            {
                result.faces.push_back(face);
                result.face_data.push_back(local_face_data);
                result.face_features.push_back(local_face_features);
            }
        }

        logFile.close();
        return result;
    } catch (const std::exception& e) {
        std::cerr << "Exception in detect_faces_impl: " << e.what() << std::endl;
        throw;
    } catch (...) {
        std::cerr << "Unknown exception in detect_faces_impl" << std::endl;
        throw;
    }
}

extern "C" __declspec(dllexport) int detect_faces(const char* image_path, const char* yolov5_model_path, const char* arcface_model_path,
                     int* faces, int max_faces, uint8_t** face_data, int* face_data_sizes, float** face_features) {
    log("进入detect_faces函数");
    try {
        log("创建FaceDetector");
        static FaceDetector detector(yolov5_model_path, ModelType::YOLOV5);
        log("FaceDetector创建成功");

        DetectionResult result = detect_faces_impl(image_path, yolov5_model_path, arcface_model_path, max_faces);

        for (int i = 0; i < result.num_faces; i++) {
            faces[i * 4] = result.faces[i].x;
            faces[i * 4 + 1] = result.faces[i].y;
            faces[i * 4 + 2] = result.faces[i].width;
            faces[i * 4 + 3] = result.faces[i].height;

            face_data[i] = new uint8_t[result.face_data[i].size()];
            std::copy(result.face_data[i].begin(), result.face_data[i].end(), face_data[i]);
            face_data_sizes[i] = static_cast<int>(result.face_data[i].size());

            face_features[i] = new float[result.face_features[i].size()];
            std::copy(result.face_features[i].begin(), result.face_features[i].end(), face_features[i]);
        }

        return result.num_faces;
    } catch (const std::exception& e) {
        log("detect_faces中的错误: " + std::string(e.what()));
        return -1;
    } catch (...) {
        log("detect_faces中的未知错误");
        return -1;
    }
}

extern "C" __declspec(dllexport) float compare_faces(float* features1, float* features2, int feature_size) {
    cv::Mat f1(1, feature_size, CV_32F, features1);
    cv::Mat f2(1, feature_size, CV_32F, features2);
    
    double dot = f1.dot(f2);
    double norm1 = cv::norm(f1);
    double norm2 = cv::norm(f2);
    double similarity = dot / (norm1 * norm2);
    
    return static_cast<float>(similarity);
}

// 函数定义
std::wstring ConvertToWideString(const std::string& multibyteStr) {
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, multibyteStr.c_str(),
                                         static_cast<int>(multibyteStr.length()), 
                                         nullptr, 0);
    if (size_needed == 0) {
        throw std::runtime_error("Failed to calculate wide string size.");
    }
    std::wstring wideStr(size_needed, 0);
    int result = MultiByteToWideChar(CP_UTF8, 0, multibyteStr.c_str(),
                                     static_cast<int>(multibyteStr.length()), 
                                     &wideStr[0], size_needed);
    if (result == 0) {
        throw std::runtime_error("Failed to convert to wide string.");
    }
    return wideStr;
}