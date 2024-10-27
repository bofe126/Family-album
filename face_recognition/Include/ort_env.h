#pragma once

#include "pch.h"

namespace face_recognition {

class OrtEnvironment {
public:
    static OrtEnvironment& getInstance() {
        static OrtEnvironment instance;
        return instance;
    }

    const OrtApi* getApi() const { return m_api; }
    OrtEnv* getEnv() const { return m_env; }

    // 删除拷贝构造和赋值操作
    OrtEnvironment(const OrtEnvironment&) = delete;
    OrtEnvironment& operator=(const OrtEnvironment&) = delete;

private:
    OrtEnvironment() : m_api(OrtGetApiBase()->GetApi(ORT_API_VERSION)), m_env(nullptr) {
        OrtStatus* status = m_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "GlobalOrtEnv", &m_env);
        if (status != nullptr) {
            const char* error_message = m_api->GetErrorMessage(status);
            m_api->ReleaseStatus(status);
            throw std::runtime_error(std::string("Failed to create OrtEnv: ") + error_message);
        }
    }

    ~OrtEnvironment() {
        if (m_env != nullptr) {
            m_api->ReleaseEnv(m_env);
            m_env = nullptr;
        }
    }

    const OrtApi* m_api;
    OrtEnv* m_env;
};

} // namespace face_recognition
