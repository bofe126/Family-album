#pragma once

#include "pch.h"
#include "types.h"

namespace face_recognition {

class ArcFaceExtractor {
public:
    // 构造函数，加载模型
    explicit ArcFaceExtractor(const char* modelPath);
    
    // 析构函数
    ~ArcFaceExtractor();
    
    // 禁用拷贝构造和赋值操作
    ArcFaceExtractor(const ArcFaceExtractor&) = delete;
    ArcFaceExtractor& operator=(const ArcFaceExtractor&) = delete;
    
    // 提取人脸特征
    std::vector<float> extract(const cv::Mat& face_image);

private:
    // ONNX Runtime 会话
    OrtSession* m_session = nullptr;
    
    // 输入输出节点名称
    std::string m_input_name = "input";
    std::string m_output_name = "output";
    
    // 模型配置
    static constexpr int INPUT_WIDTH = 112;
    static constexpr int INPUT_HEIGHT = 112;
    static constexpr int INPUT_CHANNELS = 3;
    static constexpr int FEATURE_DIM = 512;
    
    // 预处理图像
    cv::Mat preprocess(const cv::Mat& face_image);
    
    // 后处理特征
    std::vector<float> postprocess(float* output_data);
};

} // namespace face_recognition
