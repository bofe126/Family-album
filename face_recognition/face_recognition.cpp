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
#include "Include/half.hpp"  // 确保这个路径是正确的

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
    
    // 在日志中添加文名行号
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

// 添加 sigmoid 函数
float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// BoxfWithLandmarks 结构体定义
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

// NMS 函数实现
void nms_bboxes_kps(std::vector<BoxfWithLandmarks>& input, std::vector<BoxfWithLandmarks>& output, float iou_threshold) {
    std::sort(input.begin(), input.end(), [](const BoxfWithLandmarks& a, const BoxfWithLandmarks& b) {
        return a.box.score > b.box.score;
    });

    std::vector<bool> is_merged(input.size(), false);

    for (size_t i = 0; i < input.size(); ++i) {
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

class FaceDetector {
public:
    FaceDetector(const char* modelPath, ModelType type) 
        : m_type(type) {
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

            // 获取输出节点名称
            OrtAllocator* allocator;
            g_ort->GetAllocatorWithDefaultOptions(&allocator);
            g_ort->SessionGetOutputCount(m_session, &num_output_nodes);
            char* output_name;
            g_ort->SessionGetOutputName(m_session, 0, allocator, &output_name);
            m_output_node_name = std::string(output_name);
            g_ort->AllocatorFree(allocator, output_name);
            LOG("输出节点名称: " + m_output_node_name);

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

    std::vector<cv::Rect> detect(const cv::Mat& image, float score_threshold) {
        std::vector<cv::Rect> faces;
        std::vector<float> scores;
        std::vector<std::vector<cv::Point2f>> landmarks;

        if (m_type == ModelType::YOLOV5) {
            detectYOLOV5(image, faces, scores, landmarks, score_threshold);
        } else if (m_type == ModelType::RETINAFACE) {
            // 如果有 RetinaFace 的实现，也需要更新
            // detectRetinaFace(image, faces, scores, landmarks, score_threshold);
        }

        // 如果需要进行额外的后处理，可以在这里添加

        return faces;
    }

private:
    // 在 FaceDetector 类中添加 letterbox 函数
    cv::Mat letterbox(const cv::Mat &src, int target_height, int target_width, 
                      float *scale, float *offset_x, float *offset_y)
    {
        int width = src.cols;
        int height = src.rows;
        *scale = std::min((float)target_height / height, (float)target_width / width);
        int scaled_height = static_cast<int>(height * (*scale));
        int scaled_width = static_cast<int>(width * (*scale));
        *offset_x = (target_width - scaled_width) / 2.0f;
        *offset_y = (target_height - scaled_height) / 2.0f;
        cv::Mat dst(target_height, target_width, CV_8UC3, cv::Scalar(114, 114, 114));
        cv::Mat roi(dst, cv::Rect(static_cast<int>(*offset_x), static_cast<int>(*offset_y), scaled_width, scaled_height));
        cv::resize(src, roi, roi.size(), 0, 0, cv::INTER_LINEAR);
        return dst;
    }

    // 然后在类的私有部分或 .cpp 文件中实现这个函数
    void detectYOLOV5(const cv::Mat& image, std::vector<cv::Rect>& faces, 
                      std::vector<float>& scores, std::vector<std::vector<cv::Point2f>>& landmarks,
                      float score_threshold) {
        LOG("进入 detectYOLOV5 函数");
        const int inputWidth = 640;
        const int inputHeight = 640;
        float scale, offset_x, offset_y;
        cv::Mat letterboxed = letterbox(image, inputHeight, inputWidth, &scale, &offset_x, &offset_y);
        
        cv::Mat floatMat;
        letterboxed.convertTo(floatMat, CV_32F, 1.0f / 255.0f);

        std::vector<float> input_tensor(inputWidth * inputHeight * 3);
        std::memcpy(input_tensor.data(), floatMat.data, floatMat.total() * sizeof(float));

        LOG("输入张量大小: " + std::to_string(input_tensor.size()));
        LOG("输入张量维度: 1x3x" + std::to_string(inputHeight) + "x" + std::to_string(inputWidth));

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
            input_tensor.size() * sizeof(float),
            input_shape.data(),
            input_shape.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,  // 改为 FLOAT
            &input_tensor_ort
        );
        g_ort->ReleaseMemoryInfo(memory_info);
        if (status != nullptr) {
            const char* error_message = g_ort->GetErrorMessage(status);
            g_ort->ReleaseStatus(status);
            LOG("创建输入张量失败: " + std::string(error_message));
            throw std::runtime_error(std::string("Failed to create input tensor: ") + error_message);
        }

        // 获取输入节点名称
        OrtAllocator* allocator;
        g_ort->GetAllocatorWithDefaultOptions(&allocator);
        size_t num_input_nodes;
        g_ort->SessionGetInputCount(m_session, &num_input_nodes);
        std::vector<const char*> input_node_names;
        for (size_t i = 0; i < num_input_nodes; i++) {
            char* input_name;
            g_ort->SessionGetInputName(m_session, i, allocator, &input_name);
            input_node_names.push_back(input_name);
            LOG("输入节点名称: " + std::string(input_name));
        }

        // 使用正确的输入节点名称
        OrtValue* output_tensor = nullptr;
        LOG("开始运行 YOLO 模型，使用输入节点名称: " + std::string(input_node_names[0]));
        const char* output_name = m_output_node_name.c_str();
        status = g_ort->Run(
            m_session,
            nullptr,
            input_node_names.data(),
            &input_tensor_ort,
            1,
            &output_name,
            1,
            &output_tensor
        );

        // 释放输入节点名称内存
        for (const char* name : input_node_names) {
            g_ort->AllocatorFree(allocator, const_cast<char*>(name));
        }

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

        // 更新输出处理逻辑
        const int num_anchors = static_cast<int>(output_dims[1]);
        const int num_classes = 1;
        const int num_params = num_classes + 5 + 10;
        // 使用传入的 score_threshold
        const float conf_threshold = score_threshold;

        std::vector<BoxfWithLandmarks> bbox_kps_collection;

        for (int i = 0; i < num_anchors; ++i) {
            const float* row = &output_data[i * num_params];
            float obj_conf = row[4];
            float cls_conf = row[15];

            if (obj_conf < conf_threshold || cls_conf < conf_threshold) continue;

            float cx = row[0];
            float cy = row[1];
            float w = row[2];
            float h = row[3];

            float x1 = ((cx - w / 2.f) - offset_x) / scale;
            float y1 = ((cy - h / 2.f) - offset_y) / scale;
            float x2 = ((cx + w / 2.f) - offset_x) / scale;
            float y2 = ((cy + h / 2.f) - offset_y) / scale;

            BoxfWithLandmarks box_kps;
            box_kps.box.x1 = std::max(0.f, x1);
            box_kps.box.y1 = std::max(0.f, y1);
            box_kps.box.x2 = std::min(static_cast<float>(image.cols) - 1.f, x2);
            box_kps.box.y2 = std::min(static_cast<float>(image.rows) - 1.f, y2);
            box_kps.box.score = cls_conf;
            box_kps.box.label = 1;
            box_kps.box.label_text = "face";
            box_kps.box.flag = true;

            // 处理关键点
            for (int k = 0; k < 5; ++k) {
                cv::Point2f kps;
                kps.x = (row[5 + k*2] - offset_x) / scale;
                kps.y = (row[5 + k*2 + 1] - offset_y) / scale;
                kps.x = std::min(std::max(0.f, kps.x), static_cast<float>(image.cols) - 1.f);
                kps.y = std::min(std::max(0.f, kps.y), static_cast<float>(image.rows) - 1.f);
                box_kps.landmarks.points.push_back(kps);
            }
            box_kps.landmarks.flag = true;
            box_kps.flag = true;

            bbox_kps_collection.push_back(box_kps);
        }

        // 应用 NMS
        std::vector<BoxfWithLandmarks> nms_result;
        nms_bboxes_kps(bbox_kps_collection, nms_result, 0.5f);  // 使用 0.5 作为 IOU 阈值

        // 转换结果到输出格式
        faces.clear();
        scores.clear();
        landmarks.clear();
        for (const auto& box_kps : nms_result) {
            faces.emplace_back(
                static_cast<int>(box_kps.box.x1),
                static_cast<int>(box_kps.box.y1),
                static_cast<int>(box_kps.box.x2 - box_kps.box.x1),
                static_cast<int>(box_kps.box.y2 - box_kps.box.y1)
            );
            scores.push_back(box_kps.box.score);
            landmarks.push_back(box_kps.landmarks.points);
        }

        LOG("NMS 后保留了 " + std::to_string(faces.size()) + " 个人脸");

        // 可视化检测结果
        cv::Mat result_image = image.clone();
        for (size_t i = 0; i < faces.size(); ++i) {
            cv::rectangle(result_image, faces[i], cv::Scalar(0, 255, 0), 2);
            for (const auto& point : landmarks[i]) {
                cv::circle(result_image, point, 2, cv::Scalar(0, 0, 255), -1);
            }
            cv::putText(result_image, std::to_string(scores[i]), 
                        cv::Point(faces[i].x, faces[i].y - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
        }
        cv::imwrite("detection_result.jpg", result_image);
        LOG("检测结果已保存到 detection_result.jpg");

        g_ort->ReleaseValue(output_tensor);
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

            // 假设我们只使用第一个入
            char* input_name = nullptr;
            LOG("尝试获输入节点名称");
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

DetectionResult detect_faces_impl(const std::string& image_path, FaceDetector& detector, ArcFaceExtractor& arcface_extractor, int max_faces, float score_threshold) {
    LOG("开始处理图像: " + image_path);
    cv::Mat image = loadImage(image_path);
    if (image.empty()) {
        throw std::runtime_error("Failed to load image: " + image_path);
    }
    LOG("图像加载成功，尺寸: " + std::to_string(image.cols) + "x" + std::to_string(image.rows));

    LOG("开始检测人脸");
    std::vector<cv::Rect> detected_faces = detector.detect(image, score_threshold);
    LOG("检测到 " + std::to_string(detected_faces.size()) + " 个潜在人脸");

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
            LOG("无效的人脸区域，跳过");
            continue;
        }

        cv::Mat face_image = image(face).clone();
        if (face_image.empty()) {
            LOG("无法提取人脸图像");
            continue;
        }

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
}


extern "C" __declspec(dllexport) int detect_faces(const char* image_path, int* faces, uint8_t** face_data, int* face_data_sizes, float** face_features, int max_faces, float score_threshold) {
    try {
        static FaceDetector detector("assets/yolov5s-face.onnx", ModelType::YOLOV5);
        static ArcFaceExtractor arcface_extractor("assets/arcface_model.onnx");

        DetectionResult result = detect_faces_impl(image_path, detector, arcface_extractor, max_faces, score_threshold);

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
            face_data_sizes[i] = static_cast<int>(result.face_data[i].size());

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