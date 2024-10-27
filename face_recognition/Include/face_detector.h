#pragma once

#include "pch.h"
#include "types.h"

namespace face_recognition {

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

} // namespace face_recognition
