#pragma once

// Windows headers
#define NOMINMAX
#include <windows.h>

// OpenCV headers
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

// ONNX Runtime headers
#include <onnxruntime_c_api.h>

// STL headers
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <filesystem>
#include <chrono>
#include <mutex>
#include <unordered_map>
#include <queue>
#include <thread>
#include <functional>
#include <condition_variable>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <codecvt>
#include <locale>
#include <algorithm>

// Global namespace alias
namespace fs = std::filesystem;

// Forward declarations
namespace face_recognition {
    class FaceDetector;
    class ArcFaceExtractor;
    
    // Global ONNX Runtime variables declaration
    extern const OrtApi* g_ort;
    extern OrtEnv* g_ort_env;
}
