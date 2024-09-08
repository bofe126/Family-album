#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <iostream>
#include <filesystem>

// 添加人脸特征提取函数
std::vector<float> extractArcFaceFeatures(const cv::Mat& face_image, cv::dnn::Net& arcface_net) {
    cv::Mat blob = cv::dnn::blobFromImage(face_image, 1.0/128, cv::Size(112, 112), cv::Scalar(0, 0, 0), true, false);
    arcface_net.setInput(blob);
    cv::Mat features = arcface_net.forward();
    cv::normalize(features, features);
    return std::vector<float>(features.begin<float>(), features.end<float>());
}

extern "C" __declspec(dllexport) int detect_faces(const char* imagePath, const char* prototxtPath, const char* caffeModelPath, const char* arcfaceModelPath, int* faces, int max_faces, unsigned char** face_data, int* face_data_sizes, float** face_features) {
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << imagePath << std::endl;
        return 0;
    }

    cv::dnn::Net face_net = cv::dnn::readNetFromCaffe(prototxtPath, caffeModelPath);
    cv::dnn::Net arcface_net = cv::dnn::readNetFromONNX(arcfaceModelPath);

    cv::Mat blob = cv::dnn::blobFromImage(image, 1.0, cv::Size(300, 300), cv::Scalar(104.0, 177.0, 123.0));
    face_net.setInput(blob);
    cv::Mat detections = face_net.forward();

    std::vector<cv::Rect> detected_faces;
    for (int i = 0; i < detections.size[2]; i++) {
        float* detection = detections.ptr<float>(0, 0, i);
        float confidence = detection[2];

        if (confidence > 0.5) {
            int x = static_cast<int>(detection[3] * image.cols);
            int y = static_cast<int>(detection[4] * image.rows);
            int width = static_cast<int>((detection[5] - detection[3]) * image.cols);
            int height = static_cast<int>((detection[6] - detection[4]) * image.rows);

            detected_faces.push_back(cv::Rect(x, y, width, height));
        }
    }

    int valid_faces = 0;
    for (const auto& face : detected_faces) {
        if (valid_faces >= max_faces) break;

        cv::Mat face_roi = image(face);
        cv::Mat resized_face;
        cv::resize(face_roi, resized_face, cv::Size(112, 112));
        
        std::vector<float> features = extractArcFaceFeatures(resized_face, arcface_net);

        faces[valid_faces*4] = face.x;
        faces[valid_faces*4+1] = face.y;
        faces[valid_faces*4+2] = face.width;
        faces[valid_faces*4+3] = face.height;

        std::vector<uchar> buf;
        cv::imencode(".jpg", face_roi, buf);
        face_data[valid_faces] = new unsigned char[buf.size()];
        std::copy(buf.begin(), buf.end(), face_data[valid_faces]);
        face_data_sizes[valid_faces] = static_cast<int>(buf.size());

        face_features[valid_faces] = new float[features.size()];
        std::copy(features.begin(), features.end(), face_features[valid_faces]);

        valid_faces++;
    }

    return valid_faces;
}

extern "C" __declspec(dllexport) float compare_faces(float* features1, float* features2, int feature_size) {
    cv::Mat f1(1, feature_size, CV_32F, features1);
    cv::Mat f2(1, feature_size, CV_32F, features2);
    return cv::norm(f1, f2, cv::NORM_L2);
}