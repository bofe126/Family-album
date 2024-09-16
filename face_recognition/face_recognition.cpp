#define NOMINMAX
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <winrt/Windows.AI.MachineLearning.h>
#include <winrt/Windows.Foundation.Collections.h>
#include <winrt/Windows.Media.h>
#include <winrt/Windows.Storage.h>
#include <winrt/base.h>
#include <vector>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <locale>
#include <codecvt>
#include <windows.h>
#include <omp.h>
#include <memory>
#include <stdexcept>

// 添加以下行来解决 IVectorView 的问题
#include <winrt/Windows.Foundation.Collections.h>

using namespace winrt;
using namespace Windows::AI::MachineLearning;
using namespace Windows::Foundation::Collections;
using namespace Windows::Media;
using namespace Windows::Storage;

enum class ModelType {
    YOLOV5,
    RETINAFACE
};

class FaceDetector {
public:
    FaceDetector(const wchar_t* modelPath, ModelType type) 
        : m_type(type) {
        try {
            m_model = LearningModel::LoadFromFilePath(modelPath);
            m_session = LearningModelSession(m_model);
        } catch (const hresult_error& e) {
            std::wcerr << L"AI.MachineLearning error: " << e.message().c_str() << std::endl;
            throw;
        }
    }

    std::vector<cv::Rect> detect(const cv::Mat& image) {
        if (m_type == ModelType::YOLOV5) {
            return detectYOLOV5(image);
        } else {
            return detectRetinaFace(image);
        }
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

        auto input_tensor_view = TensorFloat::CreateFromArray({1, 3, inputHeight, inputWidth}, input_tensor);
        auto binding = LearningModelBinding(m_session);
        binding.Bind(L"images", input_tensor_view);

        auto results = m_session.Evaluate(binding, L"output");

        auto output = results.Outputs().Lookup(L"output").as<TensorFloat>();
        auto output_shape = output.Shape();
        auto output_data = output.GetAsVectorView();
        std::vector<float> output_tensor;
        output_tensor.reserve(output_data.Size());
        for (uint32_t i = 0; i < output_data.Size(); ++i) {
            output_tensor.push_back(output_data.GetAt(i));
        }

        std::vector<cv::Rect> faces;
        const size_t num_anchors = output_shape.GetAt(1);
        const size_t num_classes = output_shape.GetAt(2) - 5;

        for (size_t i = 0; i < num_anchors; ++i) {
            const float* row = &output_tensor[i * (num_classes + 5)];
            float confidence = row[4];
            if (confidence > 0.5) {  // 置信度阈值
                float x = row[0];
                float y = row[1];
                float w = row[2];
                float h = row[3];

                int left = static_cast<int>((x - w / 2) * image.cols);
                int top = static_cast<int>((y - h / 2) * image.rows);
                int width = static_cast<int>(w * image.cols);
                int height = static_cast<int>(h * image.rows);

                faces.push_back(cv::Rect(left, top, width, height));
            }
        }

        return faces;
    }

    std::vector<cv::Rect> detectRetinaFace(const cv::Mat& image) {
        // RetinaFace 检测实现
        return {};
    }

    LearningModel m_model{nullptr};
    LearningModelSession m_session{nullptr};
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
    ArcFaceExtractor(const wchar_t* modelPath) {
        try {
            m_model = LearningModel::LoadFromFilePath(modelPath);
            m_session = LearningModelSession(m_model);
        } catch (const hresult_error& e) {
            std::wcerr << L"AI.MachineLearning error in ArcFaceExtractor: " << e.message().c_str() << std::endl;
            throw;
        }
    }

    std::vector<float> extract(const cv::Mat& face_image) {
        cv::Mat resized_face;
        cv::resize(face_image, resized_face, cv::Size(112, 112));
        cv::cvtColor(resized_face, resized_face, cv::COLOR_BGR2RGB);
        
        std::vector<float> input_tensor(resized_face.total() * resized_face.channels());
        resized_face.convertTo(input_tensor, CV_32F, 1.0f / 255.0f);

        auto input_tensor_view = TensorFloat::CreateFromArray({1, 3, 112, 112}, input_tensor);
        auto binding = LearningModelBinding(m_session);
        binding.Bind(L"input", input_tensor_view);

        auto results = m_session.Evaluate(binding, L"output");

        auto output = results.Outputs().Lookup(L"output").as<TensorFloat>();
        auto output_data = output.GetAsVectorView();
        std::vector<float> output_tensor;
        output_tensor.reserve(output_data.Size());
        for (uint32_t i = 0; i < output_data.Size(); ++i) {
            output_tensor.push_back(output_data.GetAt(i));
        }

        return output_tensor;
    }

private:
    LearningModel m_model{nullptr};
    LearningModelSession m_session{nullptr};
};

DetectionResult detect_faces_impl(const std::string& image_path, const wchar_t* yolov5_model_path, const wchar_t* arcface_model_path, int max_faces) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        throw std::runtime_error("Failed to load image: " + image_path);
    }

    static FaceDetector detector(yolov5_model_path, ModelType::YOLOV5);
    static ArcFaceExtractor arcface_extractor(arcface_model_path);
    std::vector<cv::Rect> detected_faces = detector.detect(image);

    DetectionResult result;
    result.num_faces = std::min(static_cast<int>(detected_faces.size()), max_faces);

    #pragma omp parallel for
    for (int i = 0; i < result.num_faces; i++) {
        cv::Rect face = detected_faces[i];
        result.faces.push_back(face);

        cv::Mat face_image = image(face);
        std::vector<uchar> face_data_vec;
        cv::imencode(".jpg", face_image, face_data_vec);
        result.face_data.push_back(face_data_vec);

        std::vector<float> features = arcface_extractor.extract(face_image);
        result.face_features.push_back(features);
    }

    return result;
}

extern "C" __declspec(dllexport) int detect_faces(const char* image_path, const wchar_t* yolov5_model_path, const wchar_t* arcface_model_path,
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