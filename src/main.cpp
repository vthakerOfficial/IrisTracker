#include <myTwoStepInferencer.h>
#include <windows.h> // for overlay 
#include <overlayDot.h>
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

        if (linearModel.empty() || dotOverlay == nullptr) return;

        float x = outDetections[0];
        float y = outDetections[1];
        cout << "Iris xy: (" << x << ", " << y << ")\n";


        float hm = linearModel[0];
        float hb = linearModel[1];
        float vm = linearModel[2];
        float vb = linearModel[3];

        // let coordX = y = mx + b
        int finalX = hm * x + hb;
        int finalY = vm * y + vb;
        dotOverlay->moveTo(finalX, finalY);
    }

    void createLinearModel(DotOverlay* dotOverlay, int numPoints = 2) {
        // lamba where x vals (inputs) are iris coords and y vals (outputs) are screen coords
        auto updateVals = [this](Camera& cam, DotOverlay* dotOverlay, int& hx, int& vx, int hy, int vy) -> void {
            dotOverlay->moveTo(hy, vy);
            this_thread::sleep_for(std::chrono::seconds(3));
            cv::Mat frame = cam.getFrame();
            array<float, 4> outDetections; // in format of [x1, y1, x2, y2...]
            cv::Mat result = Inferencer::twoStepInference(frame, eyeInferencer, irisInferencer, &outDetections);
            hx = outDetections[0];
            vx = outDetections[1];
        };
        
        int w = dotOverlay->getScreenWidth();
        int h = dotOverlay->getScreenHeight();

        // let y = screen coords 
        // let x = iris coords
        int hy1 = w / 3;
        int vy1 = h / 3;
        int hx1, vx1;
        updateVals(cam, dotOverlay, hx1, vx1, hy1, vy1);

        int hy2 = w * 2 / 3;
        int vy2 = h * 2 / 3;
        int hx2, vx2;
        updateVals(cam, dotOverlay, hx2, vx2, hy2, vy2);


        // calculating y = mx + b; 
        // horizontal
        float hm = static_cast<float>(hy2 - hy1) / static_cast<float>(hx2-hx1);
        float hb = static_cast<float>(hy1) - static_cast<float>(hx1) * hm;

        // vertical
        float vm = static_cast<float>(vy2 - vy1) / static_cast<float>(vx2-vx1);
        float vb = static_cast<float>(vy1) - static_cast<float>(vx1) * vm;

        linearModel = { hm, hb, vm, vb};
        cout << "Horizontal equation: \n\ty = " << hm << "x + " << hb << endl;
        cout << "Vertical equation: \n\t y = " << vm << "x + " << vb << endl;
    } 
private:
    wstring m_eyePath = L"C:/V_Dev/irisTracker/models/eyeModel.onnx";
    wstring m_irisPath = L"C:/V_Dev/irisTracker/models/irisModel.onnx";

    Camera cam;

    Inferencer eyeInferencer;
    Inferencer irisInferencer;

    // in form of y = mx + b, first two values are m and b for horizontal
    // second 2 are for vertical
    array<float, 4> linearModel; 

};





int main() {
    
    PredictLook predicter;
    
    DotOverlay* dotOverlay = new DotOverlay();
    dotOverlay->runDotOverlay();

    cout << "Calibration has begun\n";
    predicter.createLinearModel(dotOverlay);
    cout << "Calibration has completed succesffully\n";

    // fps
    auto t_start = chrono::high_resolution_clock::now();
    int frameCount = 0;
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

            // random val
            // float rand1 = rand() / (float)(RAND_MAX);
            // float rand2 = rand() / (float)(RAND_MAX);
            // dotOverlay->moveTo(rand1 * dotOverlay->getScreenWidth(), rand2 * dotOverlay->getScreenHeight());
        }
        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q') { 
            break;
        }
    }
    delete dotOverlay;
}