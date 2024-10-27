#include "model_cache.h"
#include "utils.h"
#include "ort_env.h"

namespace face_recognition {

ModelCache& ModelCache::getInstance() {
    static ModelCache instance;
    return instance;
}

OrtSession* ModelCache::getSession(const std::string& modelPath, const std::string& modelType) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto key = modelPath + "_" + modelType;
    auto it = m_sessions.find(key);
    if (it != m_sessions.end()) {
        LOG("从缓存获取模型会话: " + modelType);
        return it->second;
    }
    return nullptr;
}

void ModelCache::addSession(const std::string& modelPath, const std::string& modelType, OrtSession* session) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto key = modelPath + "_" + modelType;
    m_sessions[key] = session;
    LOG("添加模型会话到缓存: " + modelType);
}

void ModelCache::clear() {
    std::lock_guard<std::mutex> lock(m_mutex);
    const OrtApi* ort = OrtEnvironment::getInstance().getApi();
    for (auto& pair : m_sessions) {
        if (pair.second) {
            ort->ReleaseSession(pair.second);
        }
    }
    m_sessions.clear();
    LOG("清理所有模型会话缓存");
}

} // namespace face_recognition
