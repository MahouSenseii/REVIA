#pragma once
#include <string>
#include <vector>
#include <mutex>
#include <nlohmann/json.hpp>

namespace revia {

struct PluginInfo {
    std::string name;
    std::string version;
    std::string author;
    std::string category;
    std::string capabilities;
    bool enabled = true;
    std::string status = "OK";
    std::string last_error;

    nlohmann::json to_json() const;
};

class PluginManager {
public:
    void discover(const std::string& plugins_dir);
    std::vector<PluginInfo> list() const;
    bool enable(const std::string& name);
    bool disable(const std::string& name);
    PluginInfo* find(const std::string& name);
    nlohmann::json to_json() const;

private:
    mutable std::mutex mtx_;
    std::vector<PluginInfo> plugins_;
    void register_builtins();
};

} // namespace revia
