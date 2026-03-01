#include "plugins/plugin_manager.h"
#include <filesystem>
#include <fstream>
#include <iostream>

namespace revia {

using json = nlohmann::json;

json PluginInfo::to_json() const {
    return {
        {"name", name}, {"version", version}, {"author", author},
        {"category", category}, {"capabilities", capabilities},
        {"enabled", enabled}, {"status", status}, {"last_error", last_error}
    };
}

void PluginManager::discover(const std::string& dir) {
    register_builtins();

    namespace fs = std::filesystem;
    if (!fs::exists(dir)) {
        fs::create_directories(dir);
        return;
    }
    for (auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() == ".json") {
            try {
                std::ifstream f(entry.path());
                json j;
                f >> j;
                PluginInfo p;
                p.name = j.value("name", entry.path().stem().string());
                p.version = j.value("version", "0.1.0");
                p.author = j.value("author", "Unknown");
                p.category = j.value("category", "tools");
                p.capabilities = j.value("capabilities", "");
                p.enabled = j.value("enabled", true);
                std::lock_guard lk(mtx_);
                plugins_.push_back(p);
            } catch (const std::exception& e) {
                std::cerr << "[Plugins] Failed to load " << entry.path() << ": " << e.what() << "\n";
            }
        }
    }
    std::cout << "[Plugins] " << plugins_.size() << " plugins registered.\n";
}

void PluginManager::register_builtins() {
    std::lock_guard lk(mtx_);
    plugins_ = {
        {"Whisper-STT",     "1.0.0", "REVIA", "stt",    "speech-to-text, VAD, batched",       true,  "Stub", ""},
        {"Qwen3-TTS",       "1.0.0", "REVIA", "tts",    "text-to-speech, streaming, cloning",  true,  "Stub", ""},
        {"Vision-CLIP",     "1.0.0", "REVIA", "vision",  "image classification, detection",     true,  "Stub", ""},
        {"ChromaDB-Memory", "1.0.0", "REVIA", "memory",  "vector store, RAG retrieval",         true,  "Stub", ""},
        {"LLM-Stub",        "1.0.0", "REVIA", "llm",     "text generation, streaming tokens",   true,  "OK",   ""},
        {"System-Tools",    "1.0.0", "REVIA", "tools",   "file ops, web search, commands",      false, "OK",   ""},
    };
}

std::vector<PluginInfo> PluginManager::list() const {
    std::lock_guard lk(mtx_);
    return plugins_;
}

bool PluginManager::enable(const std::string& name) {
    std::lock_guard lk(mtx_);
    for (auto& p : plugins_) {
        if (p.name == name) { p.enabled = true; p.status = "OK"; return true; }
    }
    return false;
}

bool PluginManager::disable(const std::string& name) {
    std::lock_guard lk(mtx_);
    for (auto& p : plugins_) {
        if (p.name == name) { p.enabled = false; return true; }
    }
    return false;
}

PluginInfo* PluginManager::find(const std::string& name) {
    std::lock_guard lk(mtx_);
    for (auto& p : plugins_) {
        if (p.name == name) return &p;
    }
    return nullptr;
}

json PluginManager::to_json() const {
    std::lock_guard lk(mtx_);
    json arr = json::array();
    for (auto& p : plugins_) arr.push_back(p.to_json());
    return arr;
}

} // namespace revia
