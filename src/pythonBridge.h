#pragma once

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include <vector>


namespace py = pybind11;

class PythonBridge {
public:
    PythonBridge();
    // ~PythonBridge();

    std::vector<cv::Point2f> runModelOn(const cv::Mat& img);
private:
    py::scoped_interpreter m_interpreter;
    py::module_ m_module;
};