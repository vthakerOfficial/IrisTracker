#include "pythonBridge.h"

#include <iostream>



using namespace std;

PythonBridge::PythonBridge()
    : m_interpreter{}, 
    m_module{py::module_::import("landmarksDetector")}
{}

vector<cv::Point2f> PythonBridge::runModelOn(const cv::Mat& img) {
    py::buffer_info bufInfo(
        img.data,
        sizeof(uint8_t),
        py::format_descriptor<uint8_t>::format(),
        3,
        {img.rows, img.cols, img.channels()} ,
        { img.step[0], img.step[1], sizeof(uint8_t) }
    );
    
    py::array_t<uint8_t> pyImg(bufInfo);

    py::array pts = m_module.attr("run_model_on")(pyImg);

    int n = (int)pts.shape(0);
    vector<cv::Point2f> outLandmarks(n);
    memcpy(outLandmarks.data(), pts.data(), n*sizeof(cv::Point2f));
    return outLandmarks;
}