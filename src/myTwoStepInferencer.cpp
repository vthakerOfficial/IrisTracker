#include "myTwoStepInferencer.h"
using namespace std;

Camera::Camera(int camIndex) {
    m_cam.open(camIndex);
    if (!m_cam.isOpened()) {
        cerr << "Unable to open camera." << endl;
        exit(EXIT_FAILURE);
    }
    cout << "Camera Started\n";
}

cv::Mat Camera::getFrame() {
    cv::Mat frame;
    m_cam >> frame;

    if (frame.empty()) {
        cerr << "Error: captured an empty frame" << endl;
        exit(EXIT_FAILURE);
    }

    return frame;
}

Camera::~Camera() {
    m_cam.release();
    cout << "Camera Stopped\n";
}

Inferencer::Inferencer(const wstring& modelPath, float confidence, bool bGPUBased) 
    : env(ORT_LOGGING_LEVEL_WARNING, "onnxruntimeEnv"),
    confThres(confidence),
    session(
    env,
    modelPath.c_str(),
    [bGPUBased]() -> Ort::SessionOptions {
        Ort::SessionOptions sessionOptions;
        if (bGPUBased) {
            sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
            OrtCUDAProviderOptions cudaOptions;
            cudaOptions.device_id = 0;
            sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
        }
        return sessionOptions;
    }()
    )
{
    
    // setting names of model's inputs/outputs
    { // smart ptr so we make sure it immediatley frees memory
        Ort::AllocatedStringPtr inputNamePtr = session.GetInputNameAllocated(0, allocator);
        inputName = inputNamePtr.get();
    }
    {
        Ort::AllocatedStringPtr outputNamePtr = session.GetOutputNameAllocated(0, allocator);
        outputName = outputNamePtr.get();
    }

    cout << "Loaded model: " << inputName << endl;
}
cv::Mat Inferencer::operator()(const cv::Mat& frame) {
    LetterboxResult lb = letterbox(frame);
    lastLB = lb;
    // converting bgr to rgb (fitting for model)
    cv::Mat rgb;
    cv::cvtColor(lb.image, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0f / 255.0f);

    // converting to blob (which onnx model can recieve)
    cv::Mat blob = cv::dnn::blobFromImage(rgb, 1.0, {640, 640});

    // essentially just dimensions that inputtensor needs
    vector<int64_t> dims = {1, 3, 640, 640};

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault),
        reinterpret_cast<float*>(blob.data),
        blob.total(),
        dims.data(),
        dims.size()
    );

    const char* inNames[] = { inputName.c_str() };
    const char* outNames[] = { outputName.c_str() };

    auto outputs = session.Run(
        Ort::RunOptions(nullptr),
        inNames, 
        &inputTensor, 
        1, 
        outNames,
        1
    );

    // the following code draws rectangles around results
    auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    int C = static_cast<int>(shape[1]);
    int N = static_cast<int>(shape[2]);
    float* data = outputs[0].GetTensorMutableData<float>();

    // convert data into proper format [x,y,w,h,confidence,class]
    vector<array<float,6>> predictions;
    predictions.reserve(N);
    for(int i = 0; i < N; i++){
        array<float, 6> arr{};
        for(int j = 0; j < C; j++){
            arr[j] = data[j * N + i];
        }
        if(arr[4] >= confThres){
            float x1 = arr[0] - arr[2] / 2.f;
            float y1 = arr[1] - arr[3] / 2.f;
            float x2 = arr[0] + arr[2] / 2.f;
            float y2 = arr[1] + arr[3] / 2.f;
            predictions.push_back({x1, y1, x2, y2, arr[4], arr[5]});
        }
    }
    auto boxes = rmOverlappingBoxes(predictions);
    lastInferenceBoxes = boxes;
    // 9) Draw final bounding boxes
    cv::Mat resultantFrame = frame.clone();
    for(auto &b : boxes){
        // Undo letterbox scaling/padding
        float x1 = (b[0] - lastLB.left) / lastLB.scale;
        float y1 = (b[1] - lastLB.top)  / lastLB.scale;
        float x2 = (b[2] - lastLB.left) / lastLB.scale;
        float y2 = (b[3] - lastLB.top)  / lastLB.scale;

        cv::rectangle(
            resultantFrame,
            cv::Point((int)x1, (int)y1),
            cv::Point((int)x2, (int)y2),
            cv::Scalar(0, 255, 0),
            2
        );
    }
    return resultantFrame;
}

Inferencer::LetterboxResult Inferencer::letterbox(const cv::Mat& img) {
        int w = img.cols, h = img.rows;
        float scale = min(640.f/w, 640.f/h);
        int nw = static_cast<int>(w * scale), nh = static_cast<int>(h * scale);
        int pw = 640 - nw, ph = 640-nh;
        int top = ph/2, left = pw/2;


        cv::Mat resized;
        cv::resize(img, resized, {nw,nh});

        cv::Mat letterboxed;
        //  {114,114,114} is a constant value of our yolo model padding
        cv::copyMakeBorder(resized, letterboxed, top, ph-top, left, pw-left, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
        return { letterboxed, top, left, scale };

    }

vector<array<float,6>> Inferencer::rmOverlappingBoxes(const vector<array<float,6>>& boxes, float inferenceThres) {
        vector<array<float,6>> result;
        if (boxes.empty()) return result;
        vector<int> i(boxes.size());
        for (int j = 0; j < static_cast<int>(boxes.size()); j++) {
            i[j] = j;
        }
        // sortin based on conf scores
        sort(i.begin(), i.end(), [&](int a, int b) {
            return boxes[a][4] > boxes[b][4];
        });

        vector<bool> ignored(boxes.size(), false);

        for (size_t j = 0; j < i.size(); j++) {
            if (ignored[i[j]]) continue;

            result.push_back(boxes[i[j]]);

            for (size_t k = j+1; k < i.size(); k++) {
                if (ignored[i[k]]) continue;
                
                auto &A = boxes[i[j]];
                auto &B = boxes[i[k]];

                float x1 = max(A[0], B[0]);
                float y1 = max(A[1], B[1]);
                float x2 = min(A[2], B[2]);
                float y2 = min(A[3], B[3]);

                float intersection = max(0.f, x2-x1) * max(0.f, y2-y1);
                float areaA = (A[2] -A[0]) * (A[3] - A[1]);
                float areaB = (B[2] - B[0]) *(B[3] - B[1]);

                float conf = intersection / (areaA + areaB -intersection);
                if(conf > inferenceThres){
                    ignored[i[k]] = true;
                }
            }
        }
        return result;
    }

cv::Mat Inferencer::twoStepInference(const cv::Mat& frame, Inferencer& modelA, Inferencer& modelB, array<float, 4>* outDetections) {
    int outDetectionsIndex = 0;    
    cv::Mat resultA = modelA(frame);
        int numDetections = min(2, static_cast<int>(modelA.lastInferenceBoxes.size()));
        for (int i = 0; i < numDetections; i++) {
            auto aBox = modelA.lastInferenceBoxes[i];

            int ax1  = static_cast<int>((aBox[0] - modelA.lastLB.left) / modelA.lastLB.scale);
            int ay1  = static_cast<int>((aBox[1] - modelA.lastLB.top)  / modelA.lastLB.scale);
            int ax2 = static_cast<int>((aBox[2] - modelA.lastLB.left) / modelA.lastLB.scale);
            int ay2 = static_cast<int>((aBox[3] - modelA.lastLB.top)  / modelA.lastLB.scale);
            cv::Rect aRect(ax1, ay1, ax2-ax1, ay2-ay1);

            cv::Mat aCrop = frame(aRect);
            int origW = aCrop.cols;
            int origH = aCrop.rows;

            cv::Mat aCropResized;
            cv::resize(aCrop, aCropResized, cv::Size(640, 640));
            cv::Mat bResult = modelB(aCropResized);

            if (!modelB.lastInferenceBoxes.empty()) {
                auto& bBox = modelB.lastInferenceBoxes[0];
                int bx1  = static_cast<int>((bBox[0] - modelB.lastLB.left)/ modelB.lastLB.scale);
                int by1  = static_cast<int>((bBox[1] - modelB.lastLB.top)/ modelB.lastLB.scale);
                int bx2 = static_cast<int>((bBox[2] - modelB.lastLB.left)/ modelB.lastLB.scale);
                int by2 = static_cast<int>((bBox[3] - modelB.lastLB.top)/ modelB.lastLB.scale);
                
                
                float scaleX = static_cast<float>(origW) / 640.0f;
                float scaleY = static_cast<float>(origH) / 640.0f;

                
                int origbx1 = static_cast<int>(bx1 * scaleX);
                int origby1 = static_cast<int>(by1 * scaleY);
                int origbx2 = static_cast<int>(bx2 * scaleX);
                int origby2 = static_cast<int>(by2 * scaleY);

                // scaling it onto main window
                int finalbx1 = ax1 + origbx1;
                int finalby1 = ay1 + origby1;
                int finalbx2 = ax1 + origbx2;
                int finalby2 = ay1 + origby2;

                // returning midpoint coords
                if (outDetections != nullptr) {
                    (*outDetections)[outDetectionsIndex++] = (finalbx1 + finalbx2) / 2.0;
                    (*outDetections)[outDetectionsIndex++] = (finalby1 + finalby2) / 2.0;
                }

                cv::rectangle(resultA, cv::Point(finalbx1, finalby1), cv::Point(finalbx2, finalby2), cv::Scalar(0, 255, 255), 4);
            }
        }
        return resultA;
    }

