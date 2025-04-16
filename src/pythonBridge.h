#pragma once
#include <cstdio>
#include <string>
#include <stdexcept>
#include <vector>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>

class PythonBridge {
public:
    PythonBridge(const std::string& scriptPath)
    {
        namespace fs = std::filesystem;
        fs::path script = fs::absolute(scriptPath);
        fs::path python = script.parent_path() / "venv" / "Scripts" / "python.exe";
        
        std::string insideCmd = "\"" + python.string() + "\" \"" + script.string() + "\"";
        std::string cmd = "cmd.exe /C \"" + insideCmd + "\"";
        
        m_pipe = _popen(cmd.c_str(), "r");
        if (!m_pipe) {
            throw std::runtime_error("Failed to run Python script.");
        }
    }
    ~PythonBridge()
    {
        if (m_pipe) {
            _pclose(m_pipe);
        }
    }

    bool read(std::vector<cv::Point2f>& out) {
        char buffer[10000];
        if (!fgets(buffer, sizeof(buffer), m_pipe)) {
            return false;
        }

        out.clear();

        char* token = strtok(buffer, ",\n");
        while (token) {
            float x = std::stof(token);
            token = strtok(nullptr, ",\n");
            if (!token) {
                break;
            }
            float y = std::stof(token);
            out.emplace_back(x, y);
            token = strtok(nullptr, ",\n");
        }
        return out.size() == 478; // 478 is expected output size
    }
private:
    FILE* m_pipe = nullptr;
};