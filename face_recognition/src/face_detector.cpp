#include "face_detector.h"
#include "utils.h"
#include "ort_env.h"
#include <opencv2/imgproc.hpp>

namespace face_recognition {

FaceDetector::FaceDetector(const char* modelPath, ModelType type, const Config& config)
    : m_type(type), m_config(config), m_session(nullptr) {
    try {
        Timer timer("FaceDetector初始化");
        LOG("初始化FaceDetector");
        LOG("模型路径: " + std::string(modelPath));
        
        // 使用 OrtEnvironment 单例
        const OrtApi* ort = OrtEnvironment::getInstance().getApi();
        OrtEnv* env = OrtEnvironment::getInstance().getEnv();
        
        // 尝试从缓存获取会话
        m_session = ModelCache::getInstance().getSession(modelPath, "FaceDetector");
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
        ModelCache::getInstance().addSession(modelPath, "FaceDetector", m_session);

        ort->ReleaseSessionOptions(session_options);

        // 获取输入输出节点名称
        OrtAllocator* allocator;
        ort->GetAllocatorWithDefaultOptions(&allocator);

        size_t num_output_nodes;
        ort->SessionGetOutputCount(m_session, &num_output_nodes);
        char* output_name;
        ort->SessionGetOutputName(m_session, 0, allocator, &output_name);
        m_output_node_name = std::string(output_name);
        ort->AllocatorFree(allocator, output_name);

        char* input_name;
        ort->SessionGetInputName(m_session, 0, allocator, &input_name);
        m_input_name = std::string(input_name);
        ort->AllocatorFree(allocator, input_name);

        LOG("FaceDetector初始化成功");
    } catch (const std::exception& e) {
        LOG("FaceDetector初始化错误: " + std::string(e.what()));
        throw;
    }
}

FaceDetector::~FaceDetector() {
    // 不需要释放 m_session，因为它由 ModelCache 管理
}

std::vector<BoxfWithLandmarks> FaceDetector::detect(const cv::Mat& image, float score_threshold) {
    Timer timer("人脸检测");
    try {
        if (image.empty()) {
            throw std::runtime_error("输入图像为空");
        }

        // 获取 OrtEnvironment 实例
        const OrtApi* ort = OrtEnvironment::getInstance().getApi();

        // 1. 预处理
        cv::Mat preprocessed = preprocess(image);
        
        // 2. 准备输入数据
        std::vector<float> input_tensor(preprocessed.total() * preprocessed.channels());
        cv::Mat flat = preprocessed.reshape(1, input_tensor.size());
        std::memcpy(input_tensor.data(), flat.data, input_tensor.size() * sizeof(float));

        // 3. 创建ONNX Runtime输入
        OrtMemoryInfo* memory_info;
        ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);

        std::array<int64_t, 4> input_shape = {1, 3, m_config.input_height, m_config.input_width};
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
        const char* output_names[] = {m_output_node_name.c_str()};
        
        OrtValue* output_tensor = nullptr;
        ort->Run(m_session, nullptr, input_names, &input_tensor_ort, 1, output_names, 1, &output_tensor);

        // 5. 获取输出数据
        float* output_data = nullptr;
        ort->GetTensorMutableData(output_tensor, (void**)&output_data);

        // 6. 后处理
        std::vector<BoxfWithLandmarks> detections;
        std::vector<std::pair<float, int>> confidence_indices;  // 用于排序
        const int num_anchors = 25200;  // YOLO输出的anchor数量
        const int num_classes = 1;      // 只有一个类别：人脸
        const int landmark_points = 5;   // 5个关键点
        const int output_dim = 5 + num_classes + landmark_points * 2;  // 每个anchor的输出维度

        // 收集有效的检测结果
        for (int i = 0; i < num_anchors; ++i) {
            const float* row = output_data + i * output_dim;
            float obj_conf = row[4];
            float cls_conf = row[5];
            float confidence = obj_conf * cls_conf;

            // 首先检查置信度
            if (confidence < score_threshold) {
                continue;
            }

            // 解码边界框
            float cx = row[0];
            float cy = row[1];
            float w = row[2];
            float h = row[3];

            // 检查边界框的有效性
            if (w <= 0 || h <= 0) {
                continue;
            }

            float x1 = cx - w/2;
            float y1 = cy - h/2;
            float x2 = cx + w/2;
            float y2 = cy + h/2;

            // 检查边界框坐标的有效性
            if (x1 >= x2 || y1 >= y2) {
                continue;
            }

            // 检查关键点的有效性
            bool valid_landmarks = true;
            for (int j = 0; j < landmark_points; ++j) {
                float kps_x = row[6 + j*2];
                float kps_y = row[6 + j*2 + 1];
                if (std::isnan(kps_x) || std::isnan(kps_y) || 
                    std::isinf(kps_x) || std::isinf(kps_y)) {
                    valid_landmarks = false;
                    break;
                }
            }

            if (!valid_landmarks) {
                continue;
            }

            // 只有通过所有检查的检测框才会被添加到排序列表中
            confidence_indices.push_back({confidence, i});
        }

        // 按置信度排序
        std::sort(confidence_indices.begin(), confidence_indices.end(),
                 [](const auto& a, const auto& b) { return a.first > b.first; });

        // 打印前10个最高置信度的anchor信息
        LOG("前10个最高置信度的anchor:");
        for (int i = 0; i < std::min(10, static_cast<int>(confidence_indices.size())); ++i) {
            float confidence = confidence_indices[i].first;
            int idx = confidence_indices[i].second;
            const float* row = output_data + idx * output_dim;
            
            float cx = row[0];
            float cy = row[1];
            float w = row[2];
            float h = row[3];
            
            std::stringstream ss;
            ss << std::fixed << std::setprecision(4)
               << "Anchor " << idx 
               << ": confidence=" << confidence
               << ", obj_conf=" << row[4]
               << ", cls_conf=" << row[5]
               << ", box=(" << cx << "," << cy << "," << w << "," << h << ")";
            LOG(ss.str());
        }

        // 处理有效的检测结果
        for (const auto& pair : confidence_indices) {
            int i = pair.second;
            const float* row = output_data + i * output_dim;
            
            BoxfWithLandmarks detection;
            detection.box.x1 = row[0] - row[2]/2;
            detection.box.y1 = row[1] - row[3]/2;
            detection.box.x2 = row[0] + row[2]/2;
            detection.box.y2 = row[1] + row[3]/2;
            detection.box.score = pair.first;
            detection.box.label = 0;
            detection.box.flag = true;

            // 解码关键点
            detection.landmarks.points.reserve(landmark_points);
            for (int j = 0; j < landmark_points; ++j) {
                float kps_x = row[6 + j*2];
                float kps_y = row[6 + j*2 + 1];
                detection.landmarks.points.emplace_back(kps_x, kps_y);
            }
            detection.landmarks.flag = true;
            detection.flag = true;

            detections.push_back(detection);
        }

        // 修改 NMS 处理部分
        std::vector<BoxfWithLandmarks> nms_results;
        nms_bboxes_kps(detections, nms_results, m_config.iou_threshold, 50);  // 限制最大检测数量为50

        // 按置信度排序并只保留前N个
        if (nms_results.size() > 10) {  // 设置一个合理的上限
            std::sort(nms_results.begin(), nms_results.end(),
                     [](const BoxfWithLandmarks& a, const BoxfWithLandmarks& b) {
                         return a.box.score > b.box.score;
                     });
            nms_results.resize(10);  // 只保留置信度最高的10个
        }

        LOG("检测到 " + std::to_string(nms_results.size()) + " 个人脸，置信度阈值: " + 
            std::to_string(score_threshold) + ", IOU阈值: " + std::to_string(m_config.iou_threshold));

        // 8. 坐标转换回原图尺寸
        for (auto& det : nms_results) {
            // 转换边界框坐标
            if (m_config.use_letterbox) {
                det.box.x1 = (det.box.x1 - m_scale_params.dw) / m_scale_params.ratio;
                det.box.y1 = (det.box.y1 - m_scale_params.dh) / m_scale_params.ratio;
                det.box.x2 = (det.box.x2 - m_scale_params.dw) / m_scale_params.ratio;
                det.box.y2 = (det.box.y2 - m_scale_params.dh) / m_scale_params.ratio;

                // 转换关键点坐标
                for (auto& pt : det.landmarks.points) {
                    pt.x = (pt.x - m_scale_params.dw) / m_scale_params.ratio;
                    pt.y = (pt.y - m_scale_params.dh) / m_scale_params.ratio;
                }
            } else {
                float scale_x = static_cast<float>(image.cols) / m_config.input_width;
                float scale_y = static_cast<float>(image.rows) / m_config.input_height;
                det.box.x1 *= scale_x;
                det.box.y1 *= scale_y;
                det.box.x2 *= scale_x;
                det.box.y2 *= scale_y;

                for (auto& pt : det.landmarks.points) {
                    pt.x *= scale_x;
                    pt.y *= scale_y;
                }
            }
        }

        // 9. 清理资源
        ort->ReleaseValue(input_tensor_ort);
        ort->ReleaseValue(output_tensor);
        ort->ReleaseMemoryInfo(memory_info);

        return nms_results;
    } catch (const std::exception& e) {
        LOG("人脸检测错误: " + std::string(e.what()));
        throw;
    }
}

cv::Mat FaceDetector::preprocess(const cv::Mat& image) {
    Timer timer("图像预处理");
    cv::Mat processed;
    
    if (m_config.use_letterbox) {
        float scale = std::min(
            static_cast<float>(m_config.input_width) / image.cols,
            static_cast<float>(m_config.input_height) / image.rows
        );
        
        int new_width = static_cast<int>(image.cols * scale);
        int new_height = static_cast<int>(image.rows * scale);
        
        cv::resize(image, processed, cv::Size(new_width, new_height));
        
        cv::Mat padded(m_config.input_height, m_config.input_width, CV_8UC3, cv::Scalar(114, 114, 114));
        
        int dx = (m_config.input_width - new_width) / 2;
        int dy = (m_config.input_height - new_height) / 2;
        
        processed.copyTo(padded(cv::Rect(dx, dy, new_width, new_height)));
        processed = padded;
        
        m_scale_params.ratio = scale;
        m_scale_params.dw = dx;
        m_scale_params.dh = dy;
        m_scale_params.flag = true;
    } else {
        cv::resize(image, processed, cv::Size(m_config.input_width, m_config.input_height));
        m_scale_params.ratio = static_cast<float>(m_config.input_width) / image.cols;
        m_scale_params.dw = 0;
        m_scale_params.dh = 0;
        m_scale_params.flag = true;
    }
    
    processed.convertTo(processed, CV_32F, 1.0/255.0);
    cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);
    
    return processed;
}

} // namespace face_recognition
