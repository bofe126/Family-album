#pragma once

#include "pch.h"
#include "types.h"
#include "face_detector.h"
#include "arc_face_extractor.h"

namespace face_recognition {

// 导出函数声明
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

// 内部函数声明
DetectionResult detect_faces_impl(const std::string& image_path, 
                                FaceDetector& detector, 
                                ArcFaceExtractor& arcface_extractor, 
                                int max_faces, 
                                float score_threshold);

} // namespace face_recognition
