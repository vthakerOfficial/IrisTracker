#include <myTwoStepInferencer.h>
using namespace std;

// aain3l
vector<int> xVals;

class PredictLook {
public:
    PredictLook() 
        : eyeInferencer(m_eyePath), irisInferencer(m_irisPath)
    {}
public:
void run() {
        cv::Mat frame = cam.getFrame();
        array<float, 4> outDetections; // in format of [x1, y1, x2, y2...]
        cv::Mat result = Inferencer::twoStepInference(frame, eyeInferencer, irisInferencer, &outDetections);
        {
            // aain3l
            xVals.push_back(outDetections[0]);
        }
        int j =0;
        for (int i = 0; i < outDetections.size(); i++) {

            if (i % 2 != 0) {
                cout << outDetections[i] << ")\t";
            } else {
                cout << ++j << ": (" << outDetections[i] << ", ";
            }
        }
        cout << endl;
        cv::imshow("Result", result);
    }
private:
    wstring m_eyePath = L"C:/V_Dev/irisTracker/models/eyeModel.onnx";
    wstring m_irisPath = L"C:/V_Dev/irisTracker/models/irisModel.onnx";

    Camera cam;

    Inferencer eyeInferencer;
    Inferencer irisInferencer;
};





int main() {
    PredictLook predicter;

    // fps
    auto t_start = chrono::high_resolution_clock::now();
    int frameCount = 0;
    while (true) {
        predicter.run();

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
        {
            // logic for getting mean of last 5 vals aain3l
            if (key == 'c') { 
                cout << "cut------------------------\n";
                int sum = 0;
                if (xVals.size() < 5) {
                    cout << "cut too early, xVals did not hv 5+ elems\n";
                    break;
                }
                for (int i = 0; i < 5; i ++) {
                    int xVal = xVals[xVals.size() - i - 1];
                    sum += xVal;
                }
                cout << "mean: " << sum / 5.0f << endl;
                xVals.clear();
            }
        }
    }
}