#pragma once

#include "pch.h"
#include <onnxruntime_c_api.h>
#include <string>
#include <mutex>
#include <unordered_map>

namespace face_recognition {

class ModelCache {
public:
    static ModelCache& getInstance();
    OrtSession* getSession(const std::string& modelPath, const std::string& modelType);
    void addSession(const std::string& modelPath, const std::string& modelType, OrtSession* session);
    void clear();

private:
    ModelCache() = default;
    ModelCache(const ModelCache&) = delete;
    ModelCache& operator=(const ModelCache&) = delete;

    std::mutex m_mutex;
    std::unordered_map<std::string, OrtSession*> m_sessions;
};

} // namespace face_recognition
