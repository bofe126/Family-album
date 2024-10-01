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

// 函数声明
std::wstring ConvertToWideString(const std::string& multibyteStr);

enum class ModelType {
    YOLOV5,
    RETINAFACE
};

namespace fs = std::filesystem;

// 在全局范围或在某个初始化函数中
HMODULE onnxruntimeModule = NULL;

bool loadOnnxRuntime() {
    onnxruntimeModule = LoadLibraryA("onnxruntime.dll");
    if (onnxruntimeModule == NULL) {
        std::cerr << "Failed to load onnxruntime.dll. Error code: " << GetLastError() << std::endl;
        return false;
    }
    std::cout << "Successfully loaded onnxruntime.dll" << std::endl;
    return true;
}

// 在 DllMain 或初始化函数中调用
BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) {
    switch (ul_reason_for_call) {
    case DLL_PROCESS_ATTACH:
        if (!loadOnnxRuntime()) {
            return FALSE; // 如果加载失败，返回 FALSE 可能会阻止 DLL 被加载
        }
        break;
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
        break;
    case DLL_PROCESS_DETACH:
        if (onnxruntimeModule != NULL) {
            FreeLibrary(onnxruntimeModule);
        }
        break;
    }
    return TRUE;
}

class FaceDetector {
public:
    FaceDetector(const char* modelPath, ModelType type) 
        : m_type(type), env(ORT_LOGGING_LEVEL_WARNING, "FaceDetector") {
        try {
            fs::path fullPath = fs::absolute(modelPath);

            
            std::cout << "初始化 Ort::Env 对象成功";
            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(1);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

            std::wstring wideModelPath = ConvertToWideString(fullPath.string());
            m_session = std::make_unique<Ort::Session>(env, wideModelPath.c_str(), session_options);

            // 打印模型信息
            Ort::AllocatorWithDefaultOptions allocator;
            size_t num_input_nodes = m_session->GetInputCount();
            size_t num_output_nodes = m_session->GetOutputCount();

            std::cout << "Number of inputs: " << num_input_nodes << std::endl;
            std::cout << "Number of outputs: " << num_output_nodes << std::endl;

            for (size_t i = 0; i < num_input_nodes; i++) {
                auto input_name = m_session->GetInputNameAllocated(i, allocator);
                std::cout << "Input " << i << " name: " << input_name.get() << std::endl;
            }

            for (size_t i = 0; i < num_output_nodes; i++) {
                auto output_name = m_session->GetOutputNameAllocated(i, allocator);
                std::cout << "Output " << i << " name: " << output_name.get() << std::endl;
            }

            std::cout << "模型加载成功" << std::endl;
        } catch (const Ort::Exception& e) {
            std::cerr << "ONNX Runtime error in FaceDetector: " << e.what() << std::endl;
            throw;
        } catch (const std::exception& e) {
            std::cerr << "Standard exception in FaceDetector: " << e.what() << std::endl;
            throw;
        } catch (...) {
            std::cerr << "Unknown exception in FaceDetector" << std::endl;
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
    try {
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
        std::cerr << "Error in detect_faces: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown error in detect_faces" << std::endl;
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