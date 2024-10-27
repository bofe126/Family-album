#pragma once

#include "pch.h"

namespace face_recognition {

enum class ModelType {
    YOLOV5,
    RETINAFACE
};

struct ScaleParams {
    float ratio;  // 缩放比例
    float dw;     // x方向padding
    float dh;     // y方向padding
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

using SizeType = int;   // 用于表示大小的类型
using CoordType = int;  // 用于表示坐标的类型

} // namespace face_recognition
