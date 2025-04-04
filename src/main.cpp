#include <myTwoStepInferencer.h>
using namespace std;

static const wstring g_eyePath = L"C:/V_Dev/irisTracker/models/eyeModel.onnx";
static const wstring g_irisPath = L"C:/V_Dev/irisTracker/models/irisModel.onnx";
int main() {
    Camera cam;

    Inferencer eyeInferencer(g_eyePath);
    Inferencer irisInferencer(g_irisPath);
    // fps
    auto t_start = chrono::high_resolution_clock::now();
    int frameCount = 0;

    while (true) {
        cv::Mat frame = cam.getFrame();

        cv::Mat result = Inferencer::twoStepInference(frame, eyeInferencer, irisInferencer);
        cv::imshow("Result", result);

        frameCount++;
        auto t_now = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(t_now-t_start).count();
        if (duration >= 1000) { 
            cout << "fps: " << frameCount << endl;
            frameCount = 0;
            t_start = t_now;
        }
        if ((cv::waitKey(1) & 0xFF) == 'q') { 
            break;
        }
    }

    cout << "Script has finished operating\n";
    // cin.get();
}