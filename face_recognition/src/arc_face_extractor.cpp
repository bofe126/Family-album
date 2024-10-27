#include "arc_face_extractor.h"
#include "utils.h"
#include "ort_env.h"
#include <opencv2/imgproc.hpp>

namespace face_recognition {

ArcFaceExtractor::ArcFaceExtractor(const char* modelPath) {
    try {
        Timer timer("ArcFaceExtractor初始化");
        LOG("初始化ArcFaceExtractor");
        LOG("模型路径: " + std::string(modelPath));
        
        // 使用 OrtEnvironment 单例
        const OrtApi* ort = OrtEnvironment::getInstance().getApi();
        OrtEnv* env = OrtEnvironment::getInstance().getEnv();
        
        // 尝试从缓存获取会话
        m_session = ModelCache::getInstance().getSession(modelPath, "ArcFaceExtractor");
        if (m_session != nullptr) {
            LOG("从缓存获取会话成功");
            return;
        }

        // 创建会话选项
        OrtSessionOptions* session_options;
        OrtStatus* status = ort->CreateSessionOptions(&session_options);
        if (status != nullptr) {
            const char* error_message = ort->GetErrorMessage(status);
            ort->ReleaseStatus(status);
            throw std::runtime_error(std::string("Failed to create session options: ") + error_message);
        }

        ort->SetIntraOpNumThreads(session_options, 1);
        ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_EXTENDED);

        // 创建会话
        std::wstring wideModelPath = ConvertToWideString(modelPath);
        status = ort->CreateSession(env, wideModelPath.c_str(), session_options, &m_session);
        if (status != nullptr) {
            const char* error_message = ort->GetErrorMessage(status);
            ort->ReleaseStatus(status);
            ort->ReleaseSessionOptions(session_options);
            throw std::runtime_error(std::string("Failed to create session: ") + error_message);
        }

        // 添加到缓存
        ModelCache::getInstance().addSession(modelPath, "ArcFaceExtractor", m_session);

        ort->ReleaseSessionOptions(session_options);
        LOG("ArcFaceExtractor初始化成功");
        
    } catch (const std::exception& e) {
        LOG("ArcFaceExtractor初始化失败: " + std::string(e.what()));
        throw;
    }
}

ArcFaceExtractor::~ArcFaceExtractor() {
    // 不需要释放 m_session，因为它由 ModelCache 管理
}

std::vector<float> ArcFaceExtractor::extract(const cv::Mat& face_image) {
    Timer timer("特征提取");
    try {
        // 1. 预处理图像
        cv::Mat processed = preprocess(face_image);
        
        // 获取 OrtEnvironment 实例
        const OrtApi* ort = OrtEnvironment::getInstance().getApi();
        
        // 2. 准备输入数据
        std::vector<float> input_tensor(processed.total() * processed.channels());
        cv::Mat flat = processed.reshape(1, input_tensor.size());
        std::memcpy(input_tensor.data(), flat.data, input_tensor.size() * sizeof(float));

        // 3. 创建ONNX Runtime输入
        OrtMemoryInfo* memory_info;
        ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);

        std::array<int64_t, 4> input_shape = {1, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH};
        OrtValue* input_tensor_ort = nullptr;
        ort->CreateTensorWithDataAsOrtValue(
            memory_info,
            input_tensor.data(),
            input_tensor.size() * sizeof(float),
            input_shape.data(),
            input_shape.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &input_tensor_ort
        );

        // 4. 运行推理
        const char* input_names[] = {m_input_name.c_str()};
        const char* output_names[] = {m_output_name.c_str()};
        
        OrtValue* output_tensor = nullptr;
        ort->Run(m_session, nullptr, input_names, &input_tensor_ort, 1, output_names, 1, &output_tensor);

        // 5. 获取输出数据
        float* output_data = nullptr;
        ort->GetTensorMutableData(output_tensor, (void**)&output_data);

        // 6. 后处理
        std::vector<float> features = postprocess(output_data);

        // 7. 清理资源
        ort->ReleaseValue(input_tensor_ort);
        ort->ReleaseValue(output_tensor);
        ort->ReleaseMemoryInfo(memory_info);

        return features;
    } catch (const std::exception& e) {
        LOG("特征提取错误: " + std::string(e.what()));
        throw;
    }
}

cv::Mat ArcFaceExtractor::preprocess(const cv::Mat& face_image) {
    Timer timer("ArcFace预处理");
    cv::Mat processed;
    
    // 1. 调整大小到标准输入尺寸
    cv::resize(face_image, processed, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));
    
    // 2. 转换为浮点型并归一化
    processed.convertTo(processed, CV_32F, 1.0/255.0);
    
    // 3. 转换颜色空间
    cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);
    
    return processed;
}

std::vector<float> ArcFaceExtractor::postprocess(float* output_data) {
    Timer timer("ArcFace后处理");
    
    // 复制特征向量
    std::vector<float> features(output_data, output_data + FEATURE_DIM);
    
    // L2归一化
    float norm = 0.0f;
    for (float f : features) {
        norm += f * f;
    }
    norm = std::sqrt(norm);
    
    for (float& f : features) {
        f /= norm;
    }
    
    return features;
}

} // namespace face_recognition
