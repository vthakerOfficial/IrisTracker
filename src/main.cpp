#include <iostream>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

int noitmain() {
    // checking opencv
    std::cout << CV_VERSION << std::endl;
    
    // checking onnxruntime
    std::cout << "ONNX Runtime version: " << OrtGetApiBase()->GetVersionString() << std::endl;
    return 0;
}