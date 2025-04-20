#include "camera.h"
#include <iostream>


Camera::Camera(int camIndex) {
    m_cam.open(camIndex);
    if (!m_cam.isOpened()) {
        std::cerr << "Unable to open camera." << std::endl;
        exit(EXIT_FAILURE);
    }
    std::cout << "Camera Started\n";
}

        
cv::Mat Camera::getFrame() {
    cv::Mat frame;
    m_cam >> frame;

    if (frame.empty()) {
        std::cerr << "[Camera] ERROR captured an empty frame\n";
        return frame;
    }

    return frame;
}
        
Camera::~Camera() {
    m_cam.release();
    std::cout << "Camera Stopped\n";
}
    