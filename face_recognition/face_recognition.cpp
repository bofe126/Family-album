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
#include <cstdint>
#include "Include/half.hpp"  // 添加 half.hpp 的包含

// 函数声明
std::wstring ConvertToWideString(const std::string& multibyteStr);
void setLocale();
std::string ConvertToUTF8(const std::wstring& wstr);

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

// 添加日志宏定义以自动传递文件名和
#define LOG(message) log(__FILE__, __LINE__, message)

// 修改 log 函数以仅显示文件名而非完整路径
void log(const std::string& file, int line, const std::string& message) {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    std::tm tm;
    localtime_s(&tm, &in_time_t);
    ss << std::put_time(&tm, "%Y-%m-%d %X");
    
    // 使用 std::filesystem 提取文件名
    std::string filename = fs::path(file).filename().string();
    
    // 在日志中添加文件名行号
    std::cout << "[" << ss.str() << "] [" << filename << ":" << line << "]: " << message << std::endl;
    
    static std::ofstream logfile(logFilePath, std::ios::app);
    logfile << "[" << ss.str() << "] [" << filename << ":" << line << "]: " << message << std::endl;
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
        : m_type(type), m_output_node_name("output0") {
        try {
            LOG("进入FaceDetector构造函数");
            LOG("模型路径: " + std::string(modelPath));
            LOG("模型类型: " + std::to_string(static_cast<int>(type)));

            OrtEnv* env = getGlobalOrtEnv();
            LOG("获取全局OrtEnv成功");

            LOG("初始化FaceDetector");
            fs::path fullPath = fs::absolute(modelPath);
            LOG("完整模型路径: " + fullPath.string());

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
            LOG("开始创建OrtSession");
            status = g_ort->CreateSession(env, wideModelPath.c_str(), session_options, &m_session);
            if (status != nullptr) {
                const char* error_message = g_ort->GetErrorMessage(status);
                g_ort->ReleaseStatus(status);
                g_ort->ReleaseSessionOptions(session_options);
                throw std::runtime_error(std::string("Failed to create session: ") + error_message);
            }
            LOG("OrtSession创建成功");

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

            LOG("模型输入节点数量: " + std::to_string(num_input_nodes));
            LOG("模型输出节点数量: " + std::to_string(num_output_nodes));

            LOG("FaceDetector初始化成功，使用输出节点名称: " + m_output_node_name);
        } catch (const std::exception& e) {
            LOG("FaceDetector初始化错误: " + std::string(e.what()));
            throw;
        } catch (...) {
            LOG("FaceDetector初始化未知异常");
            throw;
        }
    }

    ~FaceDetector() {
        LOG("进入FaceDetector析构函数");
        if (m_session != nullptr) {
            try {
                g_ort->ReleaseSession(m_session);
                LOG("OrtSession释放成功");
            } catch (...) {
                LOG("OrtSession释放时发生异常");
            }
            m_session = nullptr;
        }
        LOG("FaceDetector析构函数完成");
    }

    std::vector<cv::Rect> detect(const cv::Mat& image) {
        return detectYOLOV5(image);
    }

private:
    std::vector<cv::Rect> detectYOLOV5(const cv::Mat& image) {
        LOG("进入 detectYOLOV5 函数");
        const int inputWidth = 640;
        const int inputHeight = 640;
        
        LOG("原始图像尺寸: " + std::to_string(image.cols) + "x" + std::to_string(image.rows) + ", 类型: " + std::to_string(image.type()));
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(inputWidth, inputHeight));
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
        LOG("调整后的图像尺寸: " + std::to_string(resized.cols) + "x" + std::to_string(resized.rows) + ", 类型: " + std::to_string(resized.type()));

        cv::Mat floatMat;
        resized.convertTo(floatMat, CV_32F, 1.0f / 255.0f);

        // 将 float 转换为 float16
        std::vector<half_float::half> input_tensor(inputWidth * inputHeight * 3);
        for (int i = 0; i < floatMat.total() * floatMat.channels(); ++i) {
            input_tensor[i] = half_float::half_cast<half_float::half>(floatMat.ptr<float>()[i]);
        }

        LOG("输入张量大小: " + std::to_string(input_tensor.size()));
        LOG("输入张量度: 1x3x" + std::to_string(inputHeight) + "x" + std::to_string(inputWidth));

        OrtMemoryInfo* memory_info;
        OrtStatus* status = g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
        if (status != nullptr) {
            const char* error_message = g_ort->GetErrorMessage(status);
            g_ort->ReleaseStatus(status);
            throw std::runtime_error(std::string("Failed to create CPU memory info: ") + error_message);
        }

        std::array<int64_t, 4> input_shape = {1, 3, inputHeight, inputWidth};
        LOG("输入形状: 1x3x" + std::to_string(inputHeight) + "x" + std::to_string(inputWidth));

        OrtValue* input_tensor_ort;
        status = g_ort->CreateTensorWithDataAsOrtValue(
            memory_info,
            input_tensor.data(),
            input_tensor.size() * sizeof(half_float::half),
            input_shape.data(),
            input_shape.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
            &input_tensor_ort
        );
        g_ort->ReleaseMemoryInfo(memory_info);
        if (status != nullptr) {
            const char* error_message = g_ort->GetErrorMessage(status);
            g_ort->ReleaseStatus(status);
            LOG("创建输入张量失败: " + std::string(error_message));
            throw std::runtime_error(std::string("Failed to create input tensor: ") + error_message);
        }

        const char* input_names[] = {"images"};
        const char* output_names[] = {m_output_node_name.c_str()};
        OrtValue* output_tensor = nullptr;
        LOG("开始运行 YOLO 模型，使用输出节点名称: " + m_output_node_name);
        status = g_ort->Run(
            m_session,
            nullptr,
            input_names,
            &input_tensor_ort,
            1,
            output_names,
            1,
            &output_tensor
        );
        g_ort->ReleaseValue(input_tensor_ort);
        if (status != nullptr) {
            const char* error_message = g_ort->GetErrorMessage(status);
            g_ort->ReleaseStatus(status);
            throw std::runtime_error(std::string("Failed to run inference: ") + error_message);
        }
        LOG("YOLO 模型运行完成");

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
        LOG(output_shape);

        // 获取输出数据
        float* output_data;
        status = g_ort->GetTensorMutableData(output_tensor, (void**)&output_data);
        if (status != nullptr) {
            const char* error_message = g_ort->GetErrorMessage(status);
            g_ort->ReleaseStatus(status);
            g_ort->ReleaseValue(output_tensor);
            throw std::runtime_error(std::string("Failed to get output tensor data: ") + error_message);
        }

        // 输出前几个元素的值
        LOG("YOLO 输出前10个元素:");
        for (int i = 0; i < 10 && i < output_dims[0] * output_dims[1] * output_dims[2]; ++i) {
            LOG("  Element " + std::to_string(i) + ": " + std::to_string(output_data[i]));
        }

        std::vector<cv::Rect> faces;
        try {
            LOG("开始处理 YOLO 输出");
            processPredictions(output_tensor, image.cols, image.rows, faces);
        } catch (const std::exception& e) {
            LOG("处理 YOLO 输出时发生错误: " + std::string(e.what()));
            g_ort->ReleaseValue(output_tensor);
            throw;
        }
        g_ort->ReleaseValue(output_tensor);
        LOG("检测到 " + std::to_string(faces.size()) + " 个人脸");

        return faces;
    }

    void processPredictions(OrtValue* output_tensor, int orig_width, int orig_height, std::vector<cv::Rect>& faces) {
        LOG("进入 processPredictions 函数");
        LOG("原始图像尺寸: " + std::to_string(orig_width) + "x" + std::to_string(orig_height));

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
        LOG("输出张量数据类型: " + std::to_string(tensor_type));
        
        g_ort->ReleaseTensorTypeAndShapeInfo(tensor_info);

        LOG("输出张量维度: " + std::to_string(dim_count));
        std::string dims_str = "";
        for (size_t i = 0; i < dim_count; ++i) {
            dims_str += std::to_string(dims[i]) + (i < dim_count - 1 ? "x" : "");
        }
        LOG("输出张量形状: " + dims_str);

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
        LOG("检测框数量: " + std::to_string(num_boxes));

        float conf_threshold = 0.6f;  // 提高置信度阈值
        float iou_threshold = 0.5f;   // 稍微提高IOU阈值
        LOG("置信度阈值: " + std::to_string(conf_threshold) + ", IOU阈值: " + std::to_string(iou_threshold));

        const float max_area_ratio = 0.5f; // 最大允许的人脸区域占图像面积的比例
        const int image_area = orig_width * orig_height;

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

                // 计算检测框面积
                int area = static_cast<int>(w * h * image_area);
                if (area > image_area * max_area_ratio) {
                    LOG("检测到的区域过大，可能不是人脸，跳过");
                    continue;
                }

                // 将相对坐标转换为绝对坐标
                int left = static_cast<int>((x - 0.5f * w) * orig_width);
                int top = static_cast<int>((y - 0.5f * h) * orig_height);
                int width = static_cast<int>(w * orig_width);
                int height = static_cast<int>(h * orig_height);

                // 添加边界检查和修正
                left = std::max(0, std::min(left, orig_width - 1));
                top = std::max(0, std::min(top, orig_height - 1));
                width = std::min(width, orig_width - left);
                height = std::min(height, orig_height - top);

                if (width > 0 && height > 0) {
                    boxes.push_back(cv::Rect(left, top, width, height));
                    confidences.push_back(confidence);

                    LOG("检测到人脸 #" + std::to_string(boxes.size()) + ": 位置(" + 
                        std::to_string(left) + "," + std::to_string(top) + "), 大小(" + 
                        std::to_string(width) + "x" + std::to_string(height) + "), 置信度: " + 
                        std::to_string(confidence));
                }
            }
        }

        LOG("初步检测到 " + std::to_string(boxes.size()) + " 个可能的人脸");

        if (boxes.empty() || confidences.empty()) {
            LOG("没有检测到任何人脸或置信度为空，跳过NMS");
            return;
        }

        std::vector<int> indices;
        try {
            cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, iou_threshold, indices);
        } catch (const cv::Exception& e) {
            LOG("OpenCV NMS 错误: " + std::string(e.what()));
            throw;
        }

        faces.clear();
        for (int idx : indices) {
            faces.push_back(boxes[idx]);
            LOG("NMS 后保留的人脸 #" + std::to_string(faces.size()) + ": 位置(" + 
                std::to_string(boxes[idx].x) + "," + std::to_string(boxes[idx].y) + "), 大小(" + 
                std::to_string(boxes[idx].width) + "x" + std::to_string(boxes[idx].height) + "), 置信度: " + 
                std::to_string(confidences[idx]));
        }

        LOG("NMS 后保留 " + std::to_string(faces.size()) + " 个人脸");
    }

    OrtSession* m_session;
    ModelType m_type;
    std::string m_output_node_name;
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
            LOG("正在加载 ArcFace 模型: " + fullPath.string());
            
            OrtEnv* env = getGlobalOrtEnv();
            LOG("获取全局OrtEnv成功");

            OrtSessionOptions* session_options;
            OrtStatus* status = g_ort->CreateSessionOptions(&session_options);
            if (status != nullptr) {
                const char* error_message = g_ort->GetErrorMessage(status);
                g_ort->ReleaseStatus(status);
                throw std::runtime_error(std::string("Failed to create session options: ") + error_message);
            }
            LOG("创建会话选项成功");

            g_ort->SetIntraOpNumThreads(session_options, 1);
            g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_EXTENDED);
            LOG("设置会话选项成功");

            std::wstring wideModelPath = ConvertToWideString(fullPath.string());
            LOG("开始创建OrtSession");
            status = g_ort->CreateSession(env, wideModelPath.c_str(), session_options, &m_session);
            if (status != nullptr) {
                const char* error_message = g_ort->GetErrorMessage(status);
                g_ort->ReleaseStatus(status);
                g_ort->ReleaseSessionOptions(session_options);
                throw std::runtime_error(std::string("Failed to create session: ") + error_message);
            }
            LOG("OrtSession创建成功");

            g_ort->ReleaseSessionOptions(session_options);

            // 获取输入节点名称
            size_t num_input_nodes;
            status = g_ort->SessionGetInputCount(m_session, &num_input_nodes);
            if (status != nullptr) {
                throw std::runtime_error(std::string("Failed to get input count: ") + g_ort->GetErrorMessage(status));
            }
            LOG("获取输入节点数量成功: " + std::to_string(num_input_nodes));
            
            if (num_input_nodes == 0) {
                throw std::runtime_error("Model has no inputs");
            }

            // 获取默认分配器
            OrtAllocator* allocator;
            status = g_ort->GetAllocatorWithDefaultOptions(&allocator);
            if (status != nullptr) {
                const char* error_message = g_ort->GetErrorMessage(status);
                g_ort->ReleaseStatus(status);
                throw std::runtime_error(std::string("Failed to get allocator: ") + error_message);
            }

            // 假设我们只使用第一个输入
            char* input_name = nullptr;
            LOG("尝试获取输入节点名称");
            status = g_ort->SessionGetInputName(m_session, 0, allocator, &input_name);
            if (status != nullptr) {
                const char* error_message = g_ort->GetErrorMessage(status);
                g_ort->ReleaseStatus(status);
                throw std::runtime_error(std::string("Failed to get input name: ") + error_message);
            }
            LOG("成功获取输入节点名称");

            if (input_name == nullptr) {
                throw std::runtime_error("Input name is null");
            }

            // 使用input_name后，需要释放内存
            m_input_name = std::string(input_name);
            LOG("ArcFace 模型输入节点名称: " + m_input_name);

            g_ort->AllocatorFree(allocator, input_name);
            LOG("释放输入节点名称内存成功");

            // 获取输出节点名称
            size_t num_output_nodes;
            status = g_ort->SessionGetOutputCount(m_session, &num_output_nodes);
            if (status != nullptr) {
                throw std::runtime_error(std::string("Failed to get output count: ") + g_ort->GetErrorMessage(status));
            }
            LOG("获取输出节点数量成功: " + std::to_string(num_output_nodes));
            
            if (num_output_nodes == 0) {
                throw std::runtime_error("Model has no outputs");
            }

            char* output_name = nullptr;
            status = g_ort->SessionGetOutputName(m_session, 0, allocator, &output_name);
            if (status != nullptr) {
                const char* error_message = g_ort->GetErrorMessage(status);
                g_ort->ReleaseStatus(status);
                throw std::runtime_error(std::string("Failed to get output name: ") + error_message);
            }
            
            if (output_name == nullptr) {
                throw std::runtime_error("Output name is null");
            }

            m_output_name = std::string(output_name);
            LOG("ArcFace 模型输出节点名称: " + m_output_name);

            g_ort->AllocatorFree(allocator, output_name);

            LOG("ArcFaceExtractor 初始化成功");
        } catch (const std::exception& e) {
            LOG("ArcFaceExtractor 错误: " + std::string(e.what()));
            throw;
        } catch (...) {
            LOG("ArcFaceExtractor 未知异常");
            throw;
        }
    }

    ~ArcFaceExtractor() {
        if (m_session != nullptr) {
            g_ort->ReleaseSession(m_session);
        }
    }

    std::vector<float> extract(const cv::Mat& face_image) {
        LOG("开始特征提取，输入图像尺寸: " + std::to_string(face_image.cols) + "x" + std::to_string(face_image.rows) + ", 类型: " + std::to_string(face_image.type()));
        
        // 确保输入图像是正确的大小和类型
        cv::Mat processed_face;
        if (face_image.size() != cv::Size(112, 112) || face_image.type() != CV_32F) {
            cv::resize(face_image, processed_face, cv::Size(112, 112));
            processed_face.convertTo(processed_face, CV_32F);
            processed_face /= 255.0f;
        } else {
            processed_face = face_image.clone();
        }
        
        LOG("处理后的图像尺寸: " + std::to_string(processed_face.cols) + "x" + std::to_string(processed_face.rows) + ", 类型: " + std::to_string(processed_face.type()));

        std::vector<float> input_tensor(processed_face.total() * processed_face.channels());
        cv::Mat flat = processed_face.reshape(1, processed_face.total() * processed_face.channels());
        std::memcpy(input_tensor.data(), flat.data, flat.total() * sizeof(float));

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

        const char* input_names[] = {m_input_name.c_str()};  // 使用存储的输入节点名称
        const char* output_names[] = {m_output_name.c_str()};  // 使用存储的输出节点名称
        OrtValue* output_tensor = nullptr;
        LOG("开始运行 ArcFace 模型，使用输出节点名称: " + m_output_name);
        status = g_ort->Run(m_session, nullptr, input_names, &input_tensor_ort, 1, output_names, 1, &output_tensor);
        g_ort->ReleaseValue(input_tensor_ort);
        if (status != nullptr) {
            const char* error_message = g_ort->GetErrorMessage(status);
            g_ort->ReleaseStatus(status);
            throw std::runtime_error(std::string("Failed to run inference: ") + error_message);
        }
        LOG("ArcFace 模型运行完成");

        float* output_data;
        status = g_ort->GetTensorMutableData(output_tensor, (void**)&output_data);
        if (status != nullptr) {
            const char* error_message = g_ort->GetErrorMessage(status);
            g_ort->ReleaseStatus(status);
            g_ort->ReleaseValue(output_tensor);
            throw std::runtime_error(std::string("Failed to get tensor data: ") + error_message);
        }

        // 获取张量的类型和形状信息
        OrtTensorTypeAndShapeInfo* tensor_info;
        status = g_ort->GetTensorTypeAndShape(output_tensor, &tensor_info);
        if (status != nullptr) {
            const char* error_message = g_ort->GetErrorMessage(status);
            g_ort->ReleaseStatus(status);
            g_ort->ReleaseValue(output_tensor);
            throw std::runtime_error(std::string("Failed to get tensor type and shape: ") + error_message);
        }

        // 获取张量的元素数量
        size_t output_tensor_size;
        status = g_ort->GetTensorShapeElementCount(tensor_info, &output_tensor_size);
        g_ort->ReleaseTensorTypeAndShapeInfo(tensor_info);
        if (status != nullptr) {
            const char* error_message = g_ort->GetErrorMessage(status);
            g_ort->ReleaseStatus(status);
            g_ort->ReleaseValue(output_tensor);
            throw std::runtime_error(std::string("Failed to get tensor shape element count: ") + error_message);
        }

        LOG("特征提取完成，特征向量大小: " + std::to_string(output_tensor_size));

        // 修改这一行，使用 static_cast 进行安全的类型转换
        std::vector<float> output_vector(output_data, output_data + static_cast<ptrdiff_t>(output_tensor_size));
        g_ort->ReleaseValue(output_tensor);

        return output_vector;
    }

private:
    OrtSession* m_session;
    std::string m_input_name;  // 新增成员变量来存储输入节点名称
    std::string m_output_name;  // 新增成员变量来存储输出节点名称
};

// 其他函数的实现



cv::Mat loadImage(const std::string& filename) {
    try {
        std::wstring wideFilename = ConvertToWideString(filename);
        LOG("尝试加载图像: " + filename);
        LOG("宽字符文件路径: " + ConvertToUTF8(wideFilename));
        LOG("当前工作目录: " + fs::current_path().string());
        
        if (!fs::exists(wideFilename)) {
            LOG("文件不存在: " + filename);
            throw std::runtime_error("文件不存在: " + filename);
        }
        
        cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);
        if (image.empty()) {
            LOG("无法解码图像: " + filename);
            throw std::runtime_error("无法解码图像: " + filename);
        }
        
        LOG("成功加载图像: " + filename + ", 尺寸: " + std::to_string(image.cols) + "x" + std::to_string(image.rows));
        return image;
    } catch (const std::exception& e) {
        LOG("加载图像时发生异常: " + std::string(e.what()));
        throw;
    }
}

DetectionResult detect_faces_impl(const std::string& image_path, FaceDetector& detector, ArcFaceExtractor& arcface_extractor, int max_faces) {
    try {
        LOG("开始处理像: " + image_path);
        
        cv::Mat image = loadImage(image_path);
        if (image.empty()) {
            LOG("图像加载失败: " + image_path);
            throw std::runtime_error("无法加载图像: " + image_path);
        }
        
        LOG("图像加载成功，尺寸: " + std::to_string(image.cols) + "x" + std::to_string(image.rows));

        LOG("开始检测人脸");
        std::vector<cv::Rect> detected_faces;
        try {
            detected_faces = detector.detect(image);
            LOG("检测到 " + std::to_string(detected_faces.size()) + " 个人脸");
        } catch (const std::exception& e) {
            LOG("人脸检测失败: " + std::string(e.what()));
            throw;
        }

        DetectionResult result;
        result.num_faces = std::min(static_cast<int>(detected_faces.size()), max_faces);

        for (int i = 0; i < result.num_faces; i++) {
            cv::Rect face = detected_faces[i];
            LOG("处理第 " + std::to_string(i+1) + " 个人脸");

            // 确保人脸区域在图像范围内
            face.x = std::max(0, face.x);
            face.y = std::max(0, face.y);
            face.width = std::min(face.width, image.cols - face.x);
            face.height = std::min(face.height, image.rows - face.y);

            if (face.width <= 0 || face.height <= 0) {
                LOG("无效的人脸区域跳过");
                continue;
            }

            cv::Mat face_image = image(face).clone();  // 使用clone()创建一个副本
            if (face_image.empty()) {
                LOG("无法提取人脸图像");
                continue;
            }

            // 调整人脸图像大小以适应 ArcFace 模型的输入求（通常是 112x112）
            cv::Mat resized_face;
            cv::resize(face_image, resized_face, cv::Size(112, 112));

            std::vector<uint8_t> face_data;
            cv::imencode(".jpg", resized_face, face_data);
            if (face_data.empty()) {
                LOG("人脸图像编码失败");
                continue;
            }

            std::vector<float> face_features;
            try {
                face_features = arcface_extractor.extract(resized_face);
                LOG("成功提取人脸特征，特征向量大小: " + std::to_string(face_features.size()));
            } catch (const std::exception& e) {
                LOG("特征提取失败: " + std::string(e.what()));
                LOG("跳过当前人脸，继续处理下一个");
                continue;
            }

            if (face_features.empty()) {
                LOG("警告：提取的特征向量为空，跳过当前脸");
                continue;
            }

            result.faces.push_back(face);
            result.face_data.push_back(face_data);
            result.face_features.push_back(face_features);
        }

        LOG("人脸检测和特征提取完成，成功处理 " + std::to_string(result.faces.size()) + " 个人脸");
        return result;
    } catch (const std::exception& e) {
        LOG("detect_faces_impl 错误: " + std::string(e.what()));
        throw;
    } catch (...) {
        LOG("detect_faces_impl 未知异常");
        throw;
    }
}


extern "C" __declspec(dllexport) int detect_faces(const char* image_path, const char* yolov5_model_path, const char* arcface_model_path,
                     int* faces, int max_faces, uint8_t** face_data, int* face_data_sizes, float** face_features) {
    LOG("进入detect_faces函数");
    try {
        LOG("创建FaceDetector");
        static FaceDetector detector(yolov5_model_path, ModelType::YOLOV5);
        LOG("FaceDetector创建成功");

        LOG("创建ArcFaceExtractor");
        static ArcFaceExtractor arcface_extractor(arcface_model_path);
        LOG("ArcFaceExtractor创建成功");

        DetectionResult result = detect_faces_impl(image_path, detector, arcface_extractor, max_faces);

        // 确保不会超出检测到的人脸数量
        int faces_to_process = std::min(result.num_faces, max_faces);
        faces_to_process = std::min(faces_to_process, static_cast<int>(result.faces.size()));

        LOG("处理 " + std::to_string(faces_to_process) + " 个人脸");

        for (int i = 0; i < faces_to_process; i++) {
            if (i >= result.faces.size() || i >= result.face_data.size() || i >= result.face_features.size()) {
                LOG("警告：结果数组索引越界，停止处理");
                break;
            }

            faces[i * 4] = result.faces[i].x;
            faces[i * 4 + 1] = result.faces[i].y;
            faces[i * 4 + 2] = result.faces[i].width;
            faces[i * 4 + 3] = result.faces[i].height;

            face_data[i] = new uint8_t[result.face_data[i].size()];
            std::copy(result.face_data[i].begin(), result.face_data[i].end(), face_data[i]);
            face_data_sizes[i] = static_cast<int>(std::min(result.face_data[i].size(), static_cast<size_t>(std::numeric_limits<int>::max())));

            face_features[i] = new float[result.face_features[i].size()];
            std::copy(result.face_features[i].begin(), result.face_features[i].end(), face_features[i]);

            LOG("成功处理第 " + std::to_string(i+1) + " 个人脸");
        }

        LOG("detect_faces函数执行完成，处理了 " + std::to_string(faces_to_process) + " 个人脸");
        return faces_to_process;
    } catch (const std::exception& e) {
        LOG("detect_faces中的错误: " + std::string(e.what()));
        return -1;
    } catch (...) {
        LOG("detect_faces中的未知错误");
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
        setLocale();
        SetConsoleOutputCP(CP_UTF8);  // 设置控制台输出编码 UTF-8
        LOG("DLL_PROCESS_ATTACH");
        break;
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
        break;
    case DLL_PROCESS_DETACH:
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