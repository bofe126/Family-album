#include <onnxruntime_c_api.h>
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
void setLocale();

enum class ModelType {
    YOLOV5,
    RETINAFACE
};

namespace fs = std::filesystem;

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

// 全局 OrtEnv
OrtEnv* g_ort_env = nullptr;
const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

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

class FaceDetector {
public:
    FaceDetector(const char* modelPath, ModelType type) 
        : m_type(type) {
        try {
            log("进入FaceDetector构造函数");
            log("模型路径: " + std::string(modelPath));
            log("模型类型: " + std::to_string(static_cast<int>(type)));

            OrtEnv* env = getGlobalOrtEnv();
            log("获取全局OrtEnv成功");

            log("初始化FaceDetector");
            fs::path fullPath = fs::absolute(modelPath);
            log("完整模型路径: " + fullPath.string());

            OrtSessionOptions* session_options;
            OrtStatus* status = g_ort->CreateSessionOptions(&session_options);
            if (status != nullptr) {
                const char* error_message = g_ort->GetErrorMessage(status);
                g_ort->ReleaseStatus(status);
                throw std::runtime_error(std::string("Failed to create session options: ") + error_message);
            }

            g_ort->SetIntraOpNumThreads(session_options, 1);
            g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_EXTENDED);

            std::wstring wideModelPath = ConvertToWideString(fullPath.string());
            status = g_ort->CreateSession(env, wideModelPath.c_str(), session_options, &m_session);
            if (status != nullptr) {
                const char* error_message = g_ort->GetErrorMessage(status);
                g_ort->ReleaseStatus(status);
                g_ort->ReleaseSessionOptions(session_options);
                throw std::runtime_error(std::string("Failed to create session: ") + error_message);
            }

            g_ort->ReleaseSessionOptions(session_options);
            log("OrtSession创建成功");

            // 打印模型信息
            size_t num_input_nodes;
            status = g_ort->SessionGetInputCount(m_session, &num_input_nodes);
            if (status != nullptr) {
                const char* error_message = g_ort->GetErrorMessage(status);
                g_ort->ReleaseStatus(status);
                throw std::runtime_error(std::string("Failed to get input count: ") + error_message);
            }

            size_t num_output_nodes;
            status = g_ort->SessionGetOutputCount(m_session, &num_output_nodes);
            if (status != nullptr) {
                const char* error_message = g_ort->GetErrorMessage(status);
                g_ort->ReleaseStatus(status);
                throw std::runtime_error(std::string("Failed to get output count: ") + error_message);
            }

            log("输入节点数量: " + std::to_string(num_input_nodes));
            log("输出节点数量: " + std::to_string(num_output_nodes));

            log("FaceDetector初始化成功");
        } catch (const std::exception& e) {
            log("错误: " + std::string(e.what()));
            throw;
        } catch (...) {
            log("未知异常");
            throw;
        }
    }

    ~FaceDetector() {
        if (m_session != nullptr) {
            g_ort->ReleaseSession(m_session);
        }
    }

    std::vector<cv::Rect> detect(const cv::Mat& image) {
        return detectYOLOV5(image);
    }

private:
    std::vector<cv::Rect> detectYOLOV5(const cv::Mat& image) {
        log("进入 detectYOLOV5 函数");
        const int inputWidth = 640;
        const int inputHeight = 640;
        
        log("原始图像尺寸: " + std::to_string(image.cols) + "x" + std::to_string(image.rows) + ", 类型: " + std::to_string(image.type()));
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(inputWidth, inputHeight));
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
        log("调整后的图像尺寸: " + std::to_string(resized.cols) + "x" + std::to_string(resized.rows) + ", 类型: " + std::to_string(resized.type()));

        std::vector<float> input_tensor(resized.total() * resized.channels());
        resized.convertTo(input_tensor, CV_32F, 1.0f / 255.0f);
        log("输入张量大小: " + std::to_string(input_tensor.size()));

        OrtMemoryInfo* memory_info;
        OrtStatus* status = g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
        if (status != nullptr) {
            const char* error_message = g_ort->GetErrorMessage(status);
            g_ort->ReleaseStatus(status);
            throw std::runtime_error(std::string("Failed to create CPU memory info: ") + error_message);
        }

        std::array<int64_t, 4> input_shape = {1, 3, inputHeight, inputWidth};
        log("输入形状: 1x3x" + std::to_string(inputHeight) + "x" + std::to_string(inputWidth));

        OrtValue* input_tensor_ort;
        status = g_ort->CreateTensorWithDataAsOrtValue(memory_info, input_tensor.data(), input_tensor.size() * sizeof(float), 
                                                           input_shape.data(), input_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor_ort);
        g_ort->ReleaseMemoryInfo(memory_info);
        if (status != nullptr) {
            const char* error_message = g_ort->GetErrorMessage(status);
            g_ort->ReleaseStatus(status);
            throw std::runtime_error(std::string("Failed to create input tensor: ") + error_message);
        }

        const char* input_names[] = {"images"};
        const char* output_names[] = {"output"};
        OrtValue* output_tensor = nullptr;
        log("开始运行 YOLO 模型");
        status = g_ort->Run(m_session, nullptr, input_names, &input_tensor_ort, 1, output_names, 1, &output_tensor);
        g_ort->ReleaseValue(input_tensor_ort);
        if (status != nullptr) {
            const char* error_message = g_ort->GetErrorMessage(status);
            g_ort->ReleaseStatus(status);
            throw std::runtime_error(std::string("Failed to run inference: ") + error_message);
        }
        log("YOLO 模型运行完成");

        // 检查输出张量的形状和内容
        OrtTensorTypeAndShapeInfo* output_info;
        g_ort->GetTensorTypeAndShape(output_tensor, &output_info);
        size_t output_dim_count;
        g_ort->GetDimensionsCount(output_info, &output_dim_count);
        std::vector<int64_t> output_dims(output_dim_count);
        g_ort->GetDimensions(output_info, output_dims.data(), output_dim_count);
        g_ort->ReleaseTensorTypeAndShapeInfo(output_info);

        std::string output_shape = "YOLO 输出张量形状: ";
        for (size_t i = 0; i < output_dim_count; ++i) {
            output_shape += std::to_string(output_dims[i]) + (i < output_dim_count - 1 ? "x" : "");
        }
        log(output_shape);

        // 获取输出数据
        float* output_data;
        status = g_ort->GetTensorMutableData(output_tensor, (void**)&output_data);
        if (status != nullptr) {
            const char* error_message = g_ort->GetErrorMessage(status);
            g_ort->ReleaseStatus(status);
            throw std::runtime_error(std::string("Failed to get output tensor data: ") + error_message);
        }

        // 输出前几个元素的值
        log("YOLO 输出前10个元素:");
        for (int i = 0; i < 10 && i < output_dims[0] * output_dims[1] * output_dims[2]; ++i) {
            log("  Element " + std::to_string(i) + ": " + std::to_string(output_data[i]));
        }

        std::vector<cv::Rect> faces;
        log("开始处理 YOLO 输出");
        processPredictions(output_tensor, image.cols, image.rows, faces);
        g_ort->ReleaseValue(output_tensor);
        log("检测到 " + std::to_string(faces.size()) + " 个人脸");

        return faces;
    }

    void processPredictions(OrtValue* output_tensor, int orig_width, int orig_height, std::vector<cv::Rect>& faces) {
        log("进入 processPredictions 函数");
        log("原始图像尺寸: " + std::to_string(orig_width) + "x" + std::to_string(orig_height));

        OrtTensorTypeAndShapeInfo* tensor_info;
        OrtStatus* status = g_ort->GetTensorTypeAndShape(output_tensor, &tensor_info);
        if (status != nullptr) {
            const char* error_message = g_ort->GetErrorMessage(status);
            g_ort->ReleaseStatus(status);
            throw std::runtime_error(std::string("Failed to get tensor info: ") + error_message);
        }

        size_t dim_count;
        g_ort->GetDimensionsCount(tensor_info, &dim_count);
        std::vector<int64_t> dims(dim_count);
        g_ort->GetDimensions(tensor_info, dims.data(), dim_count);
        
        ONNXTensorElementDataType tensor_type;
        g_ort->GetTensorElementType(tensor_info, &tensor_type);
        log("输出张量数据类型: " + std::to_string(tensor_type));
        
        g_ort->ReleaseTensorTypeAndShapeInfo(tensor_info);

        log("输出张量维度: " + std::to_string(dim_count));
        std::string dims_str = "";
        for (size_t i = 0; i < dim_count; ++i) {
            dims_str += std::to_string(dims[i]) + (i < dim_count - 1 ? "x" : "");
        }
        log("输出张量形状: " + dims_str);

        float* output_data;
        status = g_ort->GetTensorMutableData(output_tensor, (void**)&output_data);
        if (status != nullptr) {
            const char* error_message = g_ort->GetErrorMessage(status);
            g_ort->ReleaseStatus(status);
            throw std::runtime_error(std::string("Failed to get tensor data: ") + error_message);
        }

        // YOLOv5 输出形状应该是 [1, num_boxes, 85]
        if (dim_count != 3 || dims[2] != 85) {
            throw std::runtime_error("Unexpected YOLO output shape: " + dims_str);
        }

        int64_t num_boxes = dims[1];
        log("检测框数量: " + std::to_string(num_boxes));

        float conf_threshold = 0.25f;  // 置信度阈值
        float iou_threshold = 0.45f;   // IOU阈值
        log("置信度阈值: " + std::to_string(conf_threshold) + ", IOU阈值: " + std::to_string(iou_threshold));

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

                log("检测到人脸 #" + std::to_string(boxes.size()) + ": 位置(" + 
                    std::to_string(left) + "," + std::to_string(top) + "), 大小(" + 
                    std::to_string(width) + "x" + std::to_string(height) + "), 置信度: " + 
                    std::to_string(confidence));
            }
        }

        log("初步检测到 " + std::to_string(boxes.size()) + " 个可能的人脸");

        if (boxes.empty()) {
            log("没有检测到任何人脸，跳过 NMS");
            return;
        }

        std::vector<int> indices;
        try {
            cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, iou_threshold, indices);
        } catch (const cv::Exception& e) {
            log("OpenCV 错误: " + std::string(e.what()));
            throw;
        }

        faces.clear();
        for (int idx : indices) {
            faces.push_back(boxes[idx]);
            log("NMS 后保留的人脸 #" + std::to_string(faces.size()) + ": 位置(" + 
                std::to_string(boxes[idx].x) + "," + std::to_string(boxes[idx].y) + "), 大小(" + 
                std::to_string(boxes[idx].width) + "x" + std::to_string(boxes[idx].height) + "), 置信度: " + 
                std::to_string(confidences[idx]));
        }

        log("NMS 后保留 " + std::to_string(faces.size()) + " 个人脸");
    }

    OrtSession* m_session;
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
            
            OrtEnv* env = getGlobalOrtEnv();
            OrtSessionOptions* session_options;
            OrtStatus* status = g_ort->CreateSessionOptions(&session_options);
            if (status != nullptr) {
                const char* error_message = g_ort->GetErrorMessage(status);
                g_ort->ReleaseStatus(status);
                throw std::runtime_error(std::string("Failed to create session options: ") + error_message);
            }

            g_ort->SetIntraOpNumThreads(session_options, 1);
            g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_EXTENDED);

            std::wstring wideModelPath = ConvertToWideString(fullPath.string());
            status = g_ort->CreateSession(env, wideModelPath.c_str(), session_options, &m_session);
            if (status != nullptr) {
                const char* error_message = g_ort->GetErrorMessage(status);
                g_ort->ReleaseStatus(status);
                g_ort->ReleaseSessionOptions(session_options);
                throw std::runtime_error(std::string("Failed to create session: ") + error_message);
            }

            g_ort->ReleaseSessionOptions(session_options);

            // 打印模型信息
            size_t num_input_nodes;
            status = g_ort->SessionGetInputCount(m_session, &num_input_nodes);
            if (status != nullptr) {
                const char* error_message = g_ort->GetErrorMessage(status);
                g_ort->ReleaseStatus(status);
                throw std::runtime_error(std::string("Failed to get input count: ") + error_message);
            }

            size_t num_output_nodes;
            status = g_ort->SessionGetOutputCount(m_session, &num_output_nodes);
            if (status != nullptr) {
                const char* error_message = g_ort->GetErrorMessage(status);
                g_ort->ReleaseStatus(status);
                throw std::runtime_error(std::string("Failed to get output count: ") + error_message);
            }

            std::cout << "ArcFace 模型输入数量: " << num_input_nodes << std::endl;
            std::cout << "ArcFace 模型输出数量: " << num_output_nodes << std::endl;

            std::cout << "ArcFace 模型加载成功" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "ArcFaceExtractor 错误: " << e.what() << std::endl;
            throw;
        } catch (...) {
            std::cerr << "ArcFaceExtractor 未知异常" << std::endl;
            throw;
        }
    }

    ~ArcFaceExtractor() {
        if (m_session != nullptr) {
            g_ort->ReleaseSession(m_session);
        }
    }

    std::vector<float> extract(const cv::Mat& face_image) {
        cv::Mat resized_face;
        cv::resize(face_image, resized_face, cv::Size(112, 112));
        cv::cvtColor(resized_face, resized_face, cv::COLOR_BGR2RGB);
        
        std::vector<float> input_tensor(resized_face.total() * resized_face.channels());
        resized_face.convertTo(input_tensor, CV_32F, 1.0f / 255.0f);

        OrtMemoryInfo* memory_info;
        OrtStatus* status = g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
        if (status != nullptr) {
            const char* error_message = g_ort->GetErrorMessage(status);
            g_ort->ReleaseStatus(status);
            throw std::runtime_error(std::string("Failed to create CPU memory info: ") + error_message);
        }

        std::array<int64_t, 4> input_shape = {1, 3, 112, 112};
        OrtValue* input_tensor_ort;
        status = g_ort->CreateTensorWithDataAsOrtValue(memory_info, input_tensor.data(), input_tensor.size() * sizeof(float), 
                                                           input_shape.data(), input_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor_ort);
        g_ort->ReleaseMemoryInfo(memory_info);
        if (status != nullptr) {
            const char* error_message = g_ort->GetErrorMessage(status);
            g_ort->ReleaseStatus(status);
            throw std::runtime_error(std::string("Failed to create input tensor: ") + error_message);
        }

        const char* input_names[] = {"input"};
        const char* output_names[] = {"output"};
        OrtValue* output_tensor = nullptr;
        status = g_ort->Run(m_session, nullptr, input_names, &input_tensor_ort, 1, output_names, 1, &output_tensor);
        g_ort->ReleaseValue(input_tensor_ort);
        if (status != nullptr) {
            const char* error_message = g_ort->GetErrorMessage(status);
            g_ort->ReleaseStatus(status);
            throw std::runtime_error(std::string("Failed to run inference: ") + error_message);
        }

        OrtTensorTypeAndShapeInfo* tensor_info;
        status = g_ort->GetTensorTypeAndShape(output_tensor, &tensor_info);
        if (status != nullptr) {
            const char* error_message = g_ort->GetErrorMessage(status);
            g_ort->ReleaseStatus(status);
            g_ort->ReleaseValue(output_tensor);
            throw std::runtime_error(std::string("Failed to get tensor info: ") + error_message);
        }

        size_t dim_count;
        g_ort->GetDimensionsCount(tensor_info, &dim_count);
        std::vector<int64_t> dims(dim_count);
        g_ort->GetDimensions(tensor_info, dims.data(), dim_count);
        g_ort->ReleaseTensorTypeAndShapeInfo(tensor_info);

        float* output_data;
        status = g_ort->GetTensorMutableData(output_tensor, (void**)&output_data);
        if (status != nullptr) {
            const char* error_message = g_ort->GetErrorMessage(status);
            g_ort->ReleaseStatus(status);
            g_ort->ReleaseValue(output_tensor);
            throw std::runtime_error(std::string("Failed to get tensor data: ") + error_message);
        }

        std::vector<float> output_vector(dims[0] * dims[1]);
        std::copy(output_data, output_data + output_vector.size(), output_vector.begin());

        g_ort->ReleaseValue(output_tensor);
        return output_vector;
    }

private:
    OrtSession* m_session;
};

cv::Mat loadImage(const std::string& filename) {
    try {
        fs::path filePath(filename);
        log("尝试加载图像: " + filePath.string());
        
        if (!fs::exists(filePath)) {
            log("文件不存在: " + filePath.string());
            throw std::runtime_error("文件不存在: " + filePath.string());
        }
        
        cv::Mat image = cv::imread(filePath.string(), cv::IMREAD_COLOR);
        if (image.empty()) {
            log("无法解码图像: " + filePath.string());
            throw std::runtime_error("无法解码图像: " + filePath.string());
        }
        
        log("成功加载图像: " + filePath.string() + ", 尺寸: " + std::to_string(image.cols) + "x" + std::to_string(image.rows));
        return image;
    } catch (const std::exception& e) {
        log("加载图像时发生异常: " + std::string(e.what()));
        throw;
    }
}

DetectionResult detect_faces_impl(const std::string& image_path, const char* yolov5_model_path, const char* arcface_model_path, int max_faces) {
    try {
        log("开始处理图像: " + image_path);
        
        cv::Mat image = loadImage(image_path);
        if (image.empty()) {
            log("图像加载失败: " + image_path);
            throw std::runtime_error("无法加载图像: " + image_path);
        }
        
        log("图像加载成功，尺寸: " + std::to_string(image.cols) + "x" + std::to_string(image.rows));

        log("YOLOV5 模型路径: " + std::string(yolov5_model_path));
        log("ArcFace 模型路径: " + std::string(arcface_model_path));

        // 检查模型文件是否存在
        if (!fs::exists(yolov5_model_path)) {
            log("YOLOV5 模型文件不存在: " + std::string(yolov5_model_path));
            throw std::runtime_error("YOLOV5模型文件不存在");
        }
        if (!fs::exists(arcface_model_path)) {
            log("ArcFace 模型文件不存在: " + std::string(arcface_model_path));
            throw std::runtime_error("ArcFace模型文件不存在");
        }

        log("正在创建 FaceDetector...");
        static FaceDetector detector(yolov5_model_path, ModelType::YOLOV5);
        log("FaceDetector 创建成功");

        log("正在创建 ArcFaceExtractor...");
        static ArcFaceExtractor arcface_extractor(arcface_model_path);
        log("ArcFaceExtractor 创建成功");

        log("开始检测人脸");
        std::vector<cv::Rect> detected_faces = detector.detect(image);
        log("检测到 " + std::to_string(detected_faces.size()) + " 个人脸");

        DetectionResult result;
        result.num_faces = std::min(static_cast<int>(detected_faces.size()), max_faces);

        #pragma omp parallel for
        for (int i = 0; i < result.num_faces; i++) {
            cv::Rect face = detected_faces[i];
            log("处理第 " + std::to_string(i+1) + " 个人脸");

            std::vector<uint8_t> local_face_data;
            std::vector<float> local_face_features;

            cv::Mat face_image = image(face);
            if (face_image.empty()) {
                log("无法提取人脸图像");
                continue;
            }

            cv::imencode(".jpg", face_image, local_face_data);
            if (local_face_data.empty()) {
                log("人脸图像编码失败");
                continue;
            }

            try {
                local_face_features = arcface_extractor.extract(face_image);
            } catch (const std::exception& e) {
                log("特征提取失败: " + std::string(e.what()));
                continue;
            }

            #pragma omp critical
            {
                result.faces.push_back(face);
                result.face_data.push_back(local_face_data);
                result.face_features.push_back(local_face_features);
            }
        }

        log("人脸检测和特征提取完成");
        return result;
    } catch (const std::exception& e) {
        log("detect_faces_impl 异常: " + std::string(e.what()));
        throw;
    } catch (...) {
        log("detect_faces_impl 未知异常");
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
        setLocale();  // 在这里调用 setLocale
        log("DLL_PROCESS_ATTACH");
        break;
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
        break;
    case DLL_PROCESS_DETACH:
        log("DLL_PROCESS_DETACH");
        break;
    }
    return TRUE;
}

// 在文件末尾添加 setLocale 函数定义
void setLocale() {
    std::setlocale(LC_ALL, "en_US.UTF-8");
}