#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <iostream>
#include <locale>
#include <codecvt>
#include <windows.h>
#include <string>

extern "C" __declspec(dllexport) int detect_faces(const char* imagePath, int* faces, int max_faces, unsigned char** face_data, int* face_data_sizes) {
    std::cout << "Entering detect_faces function" << std::endl;
    std::cout << "Loading image from: " << imagePath << std::endl;
    
    // 将UTF-8字符串转换为宽字符串
    int wideCharLength = MultiByteToWideChar(CP_UTF8, 0, imagePath, -1, NULL, 0);
    std::wstring wideImagePath(wideCharLength, 0);
    MultiByteToWideChar(CP_UTF8, 0, imagePath, -1, &wideImagePath[0], wideCharLength);
    
    // 将宽字符串转换为ANSI字符串
    int ansiLength = WideCharToMultiByte(CP_ACP, 0, wideImagePath.c_str(), -1, NULL, 0, NULL, NULL);
    std::string ansiImagePath(ansiLength, 0);
    WideCharToMultiByte(CP_ACP, 0, wideImagePath.c_str(), -1, &ansiImagePath[0], ansiLength, NULL, NULL);
    
    cv::Mat image = cv::imread(ansiImagePath);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << ansiImagePath << std::endl;
        return 0;
    }
    std::cout << "Successfully loaded image" << std::endl;

    // 获取当前执行路径
    char buffer[MAX_PATH];
    GetModuleFileNameA(NULL, buffer, MAX_PATH);
    std::string::size_type pos = std::string(buffer).find_last_of("\\/");
    std::string current_path = std::string(buffer).substr(0, pos);

    std::string protoPath = current_path + "\\deploy.prototxt";
    std::string modelPath = current_path + "\\res10_300x300_ssd_iter_140000.caffemodel";

    // 然后修改加载模型的代码
    cv::dnn::Net net;
    try {
        net = cv::dnn::readNetFromCaffe(protoPath, modelPath);
        std::cout << "Successfully loaded face detection model from " << protoPath << " and " << modelPath << std::endl;
    } catch (const cv::Exception& e) {
        std::cerr << "Failed to load face detection model: " << e.what() << std::endl;
        std::cerr << "Attempted to load from " << protoPath << " and " << modelPath << std::endl;
        return 0;
    }

    cv::Mat blob = cv::dnn::blobFromImage(image, 1.0, cv::Size(300, 300), cv::Scalar(104.0, 177.0, 123.0));
    net.setInput(blob);
    cv::Mat detections = net.forward();

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
    int num_faces_to_process = (std::min)(static_cast<int>(detected_faces.size()), max_faces);
    for (int i = 0; i < num_faces_to_process; i++) {
        int x = detected_faces[i].x;
        int y = detected_faces[i].y;
        int width = detected_faces[i].width;
        int height = detected_faces[i].height;

        faces[valid_faces*4] = x;
        faces[valid_faces*4+1] = y;
        faces[valid_faces*4+2] = width;
        faces[valid_faces*4+3] = height;

        // 裁剪人脸图像
        cv::Mat face_roi = image(detected_faces[i]);
        
        // 将裁剪后的图像编码为JPEG
        std::vector<uchar> buf;
        cv::imencode(".jpg", face_roi, buf);
        
        // 分配内存并复制图像数据
        unsigned char* data = new unsigned char[buf.size()];
        std::copy(buf.begin(), buf.end(), data);
        
        // 设置图像数据指针和大小
        face_data[valid_faces] = data;
        face_data_sizes[valid_faces] = static_cast<int>(buf.size());

        valid_faces++;
        if (valid_faces >= max_faces) {
            std::cout << "Reached maximum number of faces (" << max_faces << "). Stopping processing." << std::endl;
            break;
        }
    }

    std::cout << "Exiting detect_faces function, detected " << valid_faces << " faces" << std::endl;
    return valid_faces;
}