#include "pch.h"

namespace face_recognition {
    // 定义全局变量
    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    OrtEnv* g_ort_env = nullptr;
} // namespace face_recognition
