#include "face_recognition.h"
#include "face_detector.h"
#include "arc_face_extractor.h"
#include "utils.h"
#include <filesystem>

namespace face_recognition {

DetectionResult detect_faces_impl(const std::string& image_path, 
                                FaceDetector& detector, 
                                ArcFaceExtractor& arcface_extractor, 
                                int max_faces, 
                                float score_threshold) {
    Timer timer("detect_faces_impl");
    try {
        LOG("开始处理图像: " + image_path);
        
        // 1. 加载图像
        cv::Mat image = loadImage(image_path);
        if (image.empty()) {
            throw std::runtime_error("Failed to load image: " + image_path);
        }

        // 2. 人脸检测
        std::vector<BoxfWithLandmarks> detected_faces = detector.detect(image, score_threshold);
        
        // 3. 准备返回结果
        DetectionResult result;
        result.num_faces = std::min(static_cast<int>(detected_faces.size()), max_faces);
        result.faces.reserve(result.num_faces);
        result.face_data.reserve(result.num_faces);
        result.face_features.reserve(result.num_faces);

        LOG("检测到 " + std::to_string(detected_faces.size()) + " 个人脸，处理前 " + 
            std::to_string(result.num_faces) + " 个");

        // 4. 处理每个检测到的人脸
        for (int i = 0; i < result.num_faces; ++i) {
            const auto& face = detected_faces[i];
            
            // 确保边界框在图像范围内
            int x = std::max(0, static_cast<int>(face.box.x1));
            int y = std::max(0, static_cast<int>(face.box.y1));
            int width = std::min(static_cast<int>(face.box.x2 - face.box.x1), image.cols - x);
            int height = std::min(static_cast<int>(face.box.y2 - face.box.y1), image.rows - y);
            
            // 检查边界框是否有效
            if (width <= 0 || height <= 0) {
                LOG("警告：跳过无效的人脸边界框: x=" + std::to_string(x) + 
                    ", y=" + std::to_string(y) + 
                    ", width=" + std::to_string(width) + 
                    ", height=" + std::to_string(height));
                continue;
            }

            cv::Rect face_rect(x, y, width, height);
            result.faces.push_back(face_rect);
            
            // 提取人脸区域
            cv::Mat face_img = image(face_rect);
            
            // 转换为字节数组
            std::vector<uchar> face_data_buffer;
            cv::imencode(".jpg", face_img, face_data_buffer);
            result.face_data.push_back(face_data_buffer);
            
            // 提取特征
            std::vector<float> features = arcface_extractor.extract(face_img);
            result.face_features.push_back(features);
            
            LOG("处理第 " + std::to_string(i+1) + " 个人脸完成");
        }

        // 更新实际处理的人脸数量
        result.num_faces = result.faces.size();
        return result;
    } catch (const std::exception& e) {
        LOG("detect_faces_impl 错误: " + std::string(e.what()));
        throw;
    }
}

extern "C" __declspec(dllexport) int detect_faces(const char* image_path, 
                                                 int* faces, 
                                                 uint8_t** face_data, 
                                                 int* face_data_sizes, 
                                                 float** face_features, 
                                                 int max_faces, 
                                                 float score_threshold) {
    Timer timer("detect_faces");
    try {
        // 检查模型文件是否存在
        fs::path yolo_model_path = fs::absolute("assets/yolov5s-face.onnx");
        fs::path arcface_model_path = fs::absolute("assets/arcface_model.onnx");
        
        // 规范化模型路径
        std::string yolo_path = normalizePath(yolo_model_path.string());
        std::string arcface_path = normalizePath(arcface_model_path.string());
        
        LOG("YOLO模型路径: " + yolo_path);
        LOG("ArcFace模型路径: " + arcface_path);
        
        if (!fs::exists(yolo_model_path) || !fs::exists(arcface_model_path)) {
            LOG("错误：模型文件不存在");
            return -1;
        }

        // 创建检测器实例，使用规范化的路径
        static FaceDetector detector(yolo_path.c_str(), ModelType::YOLOV5);
        static ArcFaceExtractor arcface_extractor(arcface_path.c_str());

        // 调用impl函数进行检测，传入原始路径
        DetectionResult result = detect_faces_impl(image_path, detector, arcface_extractor, 
                                                 max_faces, score_threshold);

        // 复制结果
        for (int i = 0; i < result.num_faces; i++) {
            // 复制边界框
            faces[i * 4] = result.faces[i].x;
            faces[i * 4 + 1] = result.faces[i].y;
            faces[i * 4 + 2] = result.faces[i].width;
            faces[i * 4 + 3] = result.faces[i].height;

            // 复制人脸数据
            face_data[i] = new uint8_t[result.face_data[i].size()];
            std::copy(result.face_data[i].begin(), 
                     result.face_data[i].end(), 
                     face_data[i]);
            face_data_sizes[i] = static_cast<int>(result.face_data[i].size());

            // 复制特征向量
            face_features[i] = new float[result.face_features[i].size()];
            std::copy(result.face_features[i].begin(), 
                     result.face_features[i].end(), 
                     face_features[i]);
        }

        LOG("detect_faces 函数执行完成，处理了 " + std::to_string(result.num_faces) + " 个人脸");
        return result.num_faces;
    } catch (const std::exception& e) {
        LOG("检测失败: " + std::string(e.what()));
        return -1;
    }
}

extern "C" __declspec(dllexport) float compare_faces(float* features1, 
                                                    float* features2, 
                                                    int feature_size) {
    Timer timer("compare_faces");
    try {
        LOG("开始比对人脸特征");
        
        float dot_product = 0.0f;
        float norm1 = 0.0f;
        float norm2 = 0.0f;
        
        for (int i = 0; i < feature_size; ++i) {
            dot_product += features1[i] * features2[i];
            norm1 += features1[i] * features1[i];
            norm2 += features2[i] * features2[i];
        }
        
        norm1 = std::sqrt(norm1);
        norm2 = std::sqrt(norm2);
        
        float similarity = dot_product / (norm1 * norm2);
        
        LOG("特征比对完成，相似度: " + std::to_string(similarity));
        return similarity;
    } catch (const std::exception& e) {
        LOG("特征比对错误: " + std::string(e.what()));
        return -1.0f;
    }
}

extern "C" __declspec(dllexport) void cleanup_detection(uint8_t** face_data, 
                                                       float** face_features, 
                                                       int num_faces) {
    Timer timer("cleanup_detection");
    try {
        LOG("开始清理资源");
        
        if (face_data != nullptr) {
            for (int i = 0; i < num_faces; ++i) {
                if (face_data[i] != nullptr) {
                    delete[] face_data[i];
                    face_data[i] = nullptr;
                }
            }
        }
        
        if (face_features != nullptr) {
            for (int i = 0; i < num_faces; ++i) {
                if (face_features[i] != nullptr) {
                    delete[] face_features[i];
                    face_features[i] = nullptr;
                }
            }
        }
        
        LOG("资源清理完成");
    } catch (const std::exception& e) {
        LOG("资源清理错误: " + std::string(e.what()));
    }
}

} // namespace face_recognition
