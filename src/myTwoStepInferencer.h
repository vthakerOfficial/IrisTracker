#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <array>
#include <cstdlib> // used to exit
#include <chrono> // time
// #include <thread>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

class Camera {
public:
    Camera(int camIndex = 0);

    cv::Mat getFrame();

    ~Camera();

private:
    cv::VideoCapture m_cam;
};

class Inferencer {
public:
    Inferencer(const std::wstring& modelPath, float confidence = .3f, bool bGPUBased = true);
    
    cv::Mat operator()(const cv::Mat& frame);
    
    struct LetterboxResult {
        cv::Mat image;
        int top;
        int left;
        float scale;
    };
    static cv::Mat twoStepInference(const cv::Mat& frame, Inferencer& modelA, Inferencer& modelB);
private:
    LetterboxResult letterbox(const cv::Mat& img);
    std::vector<std::array<float,6>> rmOverlappingBoxes(const std::vector<std::array<float,6>>& boxes, float inferenceThres = .3f);

public:
    std::vector<std::array<float,6>> lastInferenceBoxes;
    LetterboxResult lastLB;
private:    
    Ort::Env env;
    Ort::Session session;

    Ort::AllocatorWithDefaultOptions allocator;
    std::string inputName;
    std::string outputName;

    float confThres;
};