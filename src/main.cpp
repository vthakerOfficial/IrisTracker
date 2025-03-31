#include <iostream>
#include <onnxruntime_cxx_api.h>

int notmain() {
    
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "CudaTest");
    Ort::SessionOptions sessionOptions;
    OrtCUDAProviderOptions cudaOptions;
    cudaOptions.device_id = 0;
    sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
    std::cout << "CUDA WORKS!!!!\n";
    return 0;
}
