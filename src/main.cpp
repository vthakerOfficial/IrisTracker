#include <myTwoStepInferencer.h>
#include <overlayDot.h>

#include <windows.h> // for overlay 
#include <thread>
#include <chrono>


using namespace std;


class PredictLook {
public:
    PredictLook() 
        : eyeInferencer(m_eyePath), irisInferencer(m_irisPath)
    {}
public:
    void run(DotOverlay* dotOverlay = nullptr) {
        cv::Mat frame = cam.getFrame();
        array<float, 4> outDetections; // in format of [x1, y1, x2, y2...]
        cv::Mat result = Inferencer::twoStepInference(frame, eyeInferencer, irisInferencer, &outDetections);
        cv::imshow("Result", result);

        float x = outDetections[0];
        float y = outDetections[1];
        cout << "Iris xy: (" << x << ", " << y << ")";

        if (modelIrisScreen.empty() || dotOverlay == nullptr) {
            cout << endl;
            return;
        }

        vector<cv::Point2f> src, dst;
        src.emplace_back(x, y);
        cv::perspectiveTransform(src, dst, modelIrisScreen);
        float predictedScreenX = dst[0].x;
        float predictedScreenY = dst[0].y;

        int w = dotOverlay->getScreenWidth();
        int h = dotOverlay->getScreenHeight();
        if (predictedScreenX < 0) {
            predictedScreenX = 0;
        } 
        else if (predictedScreenX > w) {
            predictedScreenX = w;
        }
        if (predictedScreenY < 0) {
            predictedScreenY = 0;
        } 
        else if (predictedScreenY > h) {
            predictedScreenY = h;
        }


        cout << "\tScreen xy: (" << predictedScreenX << ", " <<  predictedScreenY << ")\n";

        dotOverlay->moveTo(predictedScreenX, predictedScreenY);
    }

    void calibrate(DotOverlay* dotOverlay) {
        int w = dotOverlay->getScreenWidth();
        int h = dotOverlay->getScreenHeight();
        
        vector<cv::Point2f> irisPts;
        std::vector<cv::Point2f> screenPts {
            { w / 10.0f, h / 10.0f },
            { w * 9.0f / 10.0f, h / 10.0f },
            { w / 10.0f, h * 9.0f / 10.0f },
            { w * 9.0f / 10.0f, h * 9.0f / 10.0f },
            { w / 2.0f , h / 2.0f }
        };

        for (cv::Point2f& pt : screenPts) {
            dotOverlay->moveTo(pt.x, pt.y);
            this_thread::sleep_for(std::chrono::seconds(3));
            cv::Mat frame = cam.getFrame();
            array<float, 4> outDetections; // in format of [x1, y1, x2, y2...]
            cv::Mat result = Inferencer::twoStepInference(frame, eyeInferencer, irisInferencer, &outDetections);
            
            float irisX = outDetections[0];
            float irisY = outDetections[1];

            irisPts.emplace_back(irisX, irisY);

            cout << "Calibration: Added coords iris(" << irisX << ", " << irisY
            << ") & screen(" << pt.x << ", " << pt.y << ")\n"; 
        }
        modelIrisScreen = cv::findHomography(irisPts, screenPts, cv::RANSAC);
        cout << "Success in creating model? " << !modelIrisScreen.empty() << endl;
    } 
private:
    wstring m_eyePath = L"C:/V_Dev/irisTracker/models/eyeModel.onnx";
    wstring m_irisPath = L"C:/V_Dev/irisTracker/models/irisModel.onnx";

    Camera cam;

    Inferencer eyeInferencer;
    Inferencer irisInferencer;

    // in form of y = mx + b, first two values are m and b for horizontal
    // second 2 are for vertical
    cv::Mat modelIrisScreen;


};





int main() {
    
    PredictLook predicter;
    
    DotOverlay* dotOverlay = new DotOverlay();
    dotOverlay->runDotOverlay();
    
    cout << "Calibration has begun\n";
    predicter.calibrate(dotOverlay);
    cout << "Calibration has completed succesffully\n";

    //fps
    auto t_start = chrono::high_resolution_clock::now();
    int frameCount = 0;
    cout << "in while true loop\n";
    while (true) {
        predicter.run(dotOverlay);

        // fps logic + q to break out of loop below
        frameCount++;
        auto t_now = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(t_now-t_start).count();
        if (duration >= 1000) { 
            // cout << "fps: " << frameCount << endl;
            frameCount = 0;
            t_start = t_now;

        }
        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q') { 
            break;
        }
    }

}