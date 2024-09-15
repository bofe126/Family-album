#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
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

enum class ModelType {
    YOLOV5,
    RETINAFACE
};

class FaceDetector {
public:
    FaceDetector(const std::string& modelPath, ModelType type) 
        : m_type(type) {
        try {
            Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "FaceDetector");
            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(1);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
            m_session = std::make_unique<Ort::Session>(env, modelPath.c_str(), session_options);
        } catch (const Ort::Exception& e) {
            std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
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
        
        cv::Mat blob;
        cv::dnn::blobFromImage(image, blob, 1.0/255.0, cv::Size(inputWidth, inputHeight), cv::Scalar(), true, false);
        
        std::vector<Ort::Value> input_tensors;
        std::vector<Ort::Value> output_tensors;
        
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        input_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info, blob.ptr<float>(), blob.total() * blob.channels(), {1, 3, inputWidth, inputHeight}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        
        try {
            output_tensors = m_session->Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_names.size(), output_names.data(), output_names.size());
        } catch (const Ort::Exception& e) {
            std::cerr << "ONNX Runtime error during inference: " << e.what() << std::endl;
            return {};
        }
        
        // 处理输出张量，提取人脸边界框
        std::vector<cv::Rect> faces;
        const float* output_data = output_tensors[0].GetTensorData<float>();
        const size_t num_anchors = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape()[1];
        const size_t num_classes = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape()[2] - 5;

        for (size_t i = 0; i < num_anchors; ++i) {
            const float* row = output_data + i * (num_classes + 5);
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
        // ...
        return {};
    }

    std::unique_ptr<Ort::Session> m_session;
    ModelType m_type;
    std::vector<const char*> input_names = {"images"};
    std::vector<const char*> output_names = {"output"};
};

struct DetectionResult {
    int num_faces;
    std::vector<cv::Rect> faces;
    std::vector<std::vector<uint8_t>> face_data;
    std::vector<std::vector<float>> face_features;
};

class ArcFaceExtractor {
public:
    ArcFaceExtractor(const std::string& modelPath) {
        try {
            Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ArcFaceExtractor");
            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(1);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
            m_session = std::make_unique<Ort::Session>(env, modelPath.c_str(), session_options);
        } catch (const Ort::Exception& e) {
            std::cerr << "ONNX Runtime error in ArcFaceExtractor: " << e.what() << std::endl;
            throw;
        }
    }

    std::vector<float> extract(const cv::Mat& face_image) {
        cv::Mat resized_face;
        cv::resize(face_image, resized_face, cv::Size(112, 112));
        cv::Mat float_image;
        resized_face.convertTo(float_image, CV_32F, 1.0 / 255.0);

        cv::Mat channels[3];
        cv::split(float_image, channels);
        
        std::vector<float> input_tensor;
        input_tensor.insert(input_tensor.end(), channels[2].begin<float>(), channels[2].end<float>());
        input_tensor.insert(input_tensor.end(), channels[1].begin<float>(), channels[1].end<float>());
        input_tensor.insert(input_tensor.end(), channels[0].begin<float>(), channels[0].end<float>());

        std::vector<Ort::Value> input_tensors;
        std::vector<Ort::Value> output_tensors;

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        input_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info, input_tensor.data(), input_tensor.size(), {1, 3, 112, 112}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        try {
            output_tensors = m_session->Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_names.size(), output_names.data(), output_names.size());
        } catch (const Ort::Exception& e) {
            std::cerr << "ONNX Runtime error during ArcFace inference: " << e.what() << std::endl;
            return {};
        }

        const float* output_data = output_tensors[0].GetTensorData<float>();
        size_t output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

        return std::vector<float>(output_data, output_data + output_size);
    }

private:
    std::unique_ptr<Ort::Session> m_session;
    std::vector<const char*> input_names = {"input"};
    std::vector<const char*> output_names = {"output"};
};

DetectionResult detect_faces_impl(const std::string& image_path, const std::string& yolov5_model_path, const std::string& arcface_model_path, int max_faces) {
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

        // 提取人脸图像
        cv::Mat face_image = image(face);
        std::vector<uchar> face_data_vec;
        cv::imencode(".jpg", face_image, face_data_vec);
        result.face_data.push_back(face_data_vec);

        // 使用ArcFace模型提取特征
        std::vector<float> features = arcface_extractor.extract(face_image);
        result.face_features.push_back(features);
    }

    return result;
}

extern "C" __declspec(dllexport) int detect_faces(const char* image_path, const char* yolov5_model_path, const char* arcface_model_path,
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
            face_data_sizes[i] = result.face_data[i].size();

            face_features[i] = new float[result.face_features[i].size()];
            std::copy(result.face_features[i].begin(), result.face_features[i].end(), face_features[i]);
        }

        return result.num_faces;
    } catch (const std::exception& e) {
        std::cerr << "Error in detect_faces: " << e.what() << std::endl;
        return -1;
    }
}

extern "C" __declspec(dllexport) float compare_faces(float* features1, float* features2, int feature_size) {
    cv::Mat f1(1, feature_size, CV_32F, features1);
    cv::Mat f2(1, feature_size, CV_32F, features2);
    
    // 计算余弦相似度
    double dot = f1.dot(f2);
    double norm1 = cv::norm(f1);
    double norm2 = cv::norm(f2);
    double similarity = dot / (norm1 * norm2);
    
    return static_cast<float>(similarity);
}