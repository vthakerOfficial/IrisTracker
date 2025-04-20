#pragma once
#include <opencv2/opencv.hpp>

class Camera {
public:
    Camera(int camindex = 0);
    ~Camera();
    cv::Mat getFrame();

private:
    cv::VideoCapture m_cam;
};