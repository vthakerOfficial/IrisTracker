#include <myTwoStepInferencer.h>
#include <overlayDot.h>

#include <windows.h> // for overlay 
#include <thread>
#include <chrono>


using namespace std;


class PredictLook {
public:
    struct CalibrationHomographies {
        cv::Mat eyeLeft;
        cv::Mat eyeRight;
    };
public:
    PredictLook(Camera& cam) 
        : eyeInferencer(m_eyePath), irisInferencer(m_irisPath), cam(cam)
    {}
    void run(DotOverlay* dotOverlay = nullptr) {
        cv::Mat frame = cam.getFrame();
        if (frame.empty()) {
            cout << "Camera inaccessible\n";
            exit(EXIT_FAILURE);
        }

        array<float, 4> outDetections; // in format of [x1, y1, x2, y2...]
        cv::Mat result = Inferencer::twoStepInference(frame, eyeInferencer, irisInferencer, &outDetections);
        cv::imshow("Result", result);

        if (!dotOverlay) return;


        bool bBothEyesDetected = true;
        for (int i = 0; i < outDetections.size(); i++) {
            if (outDetections[i] < 0) {
                bBothEyesDetected = false;
                break;
            }
        }
        if (!bBothEyesDetected) return;



        vector<cv::Point2f> srcLeft, srcRight;
        if (outDetections[0] < outDetections[2]) {
            srcLeft.emplace_back(outDetections[0], outDetections[1]);
            srcRight.emplace_back(outDetections[2], outDetections[3]);
        } else {
            srcLeft.emplace_back(outDetections[2], outDetections[3]);
            srcRight.emplace_back(outDetections[0], outDetections[1]);

        }

        vector<cv::Point2f> dstLeft, dstRight;
        cv::perspectiveTransform(srcLeft, dstLeft, calibH.eyeLeft);
        cv::perspectiveTransform(srcRight, dstRight, calibH.eyeRight);
        
        cv::Point2f avg( (dstLeft[0].x + dstRight[0].x) / 2.0f, (dstLeft[0].y + dstRight[0].y) / 2.0f );

        dotOverlay->moveTo(avg.x, avg.y);

        cout << "Predicted Coord acc 2 Eye Left(" << dstLeft[0].x << ", " << dstLeft[0].y 
        << "), Right(" << dstRight[0].x << ", " << dstRight[0].y << "), Avg: "
        << avg.x << ", " << avg.y << ")\n";
    }

    void calibrate(DotOverlay* dotOverlay) {
        int w = dotOverlay->getScreenWidth();
        int h = dotOverlay->getScreenHeight();
        
        std::vector<cv::Point2f> screenPts {
            { w * .1f, h * .1f },
            { w * .5f, h * .1f },
            { w * .9f, h * .1f },
            { w * .1f, h * .5f },
            { w * .5f, h * .5f },
            { w * .9f, h * .5f },
            { w * .1f, h * .9f },
            { w * .5f, h * .9f},
            { w * .9f, h * .9f}
        };

        // adding this for visualization purposes
        // {
        //     cv::Mat frame = cam.getFrame();
        //     if (frame.empty()) {
        //         cout << "Camera inaccessible\n";
        //         exit(EXIT_FAILURE);
        //     }
        //     array<float,4> outDetections;
        //     cv::Mat result = Inferencer::twoStepInference(frame, eyeInferencer, irisInferencer, &outDetections);
        //     cv::imshow("Result", result);
        // }
        //

        vector<cv::Point2f> irisPtsLeft;
        vector<cv::Point2f> irisPtsRight;

        cout << "Follow red dot, blink on dot move. Keep head still.\n";
        for (cv::Point2f& pt : screenPts) {
            dotOverlay->moveTo(pt.x, pt.y);
            this_thread::sleep_for(std::chrono::seconds(1));
            
            const int numSamples = 5;
            float sumXLeft = 0;
            float sumYLeft = 0;

            float sumXRight = 0;
            float sumYRight = 0;

            for (int i = 0; i < numSamples; i++) {
                cv::Mat frame = cam.getFrame();
                array<float, 4> outDetections; // in format of [x1, y1, x2, y2...]
                cv::Mat result = Inferencer::twoStepInference(frame, eyeInferencer, irisInferencer, &outDetections);
                bool bSuccess = true;
                for (int j = 0; j < outDetections.size(); j++) {
                    if (outDetections[j] < 0) {
                        bSuccess = false;
                        break;
                    }
                }
                if (bSuccess) {
                    if (outDetections[0] < outDetections[2]) {
                        sumXLeft += outDetections[0];
                        sumYLeft += outDetections[1];
                        sumXRight += outDetections[2];
                        sumYRight += outDetections[3];
                    } else {
                        sumXLeft += outDetections[2];
                        sumYLeft += outDetections[3];
                        sumXRight += outDetections[0];
                        sumYRight += outDetections[1];
                    }
                } else {
                    cout << "Failed calibration. No iris detected\n";
                    exit(EXIT_FAILURE);
                }
                this_thread::sleep_for(chrono::milliseconds(50));
            }
            float avgXLeft = sumXLeft / numSamples;
            float avgYLeft = sumYLeft / numSamples;
            float avgXRight = sumXRight / numSamples;
            float avgYRight = sumYRight / numSamples;

            irisPtsLeft.emplace_back(avgXLeft, avgYLeft);
            irisPtsRight.emplace_back(avgXRight, avgYRight);

            cout << "Calibration: At screen(" << pt.x << ", " << pt.y << "),\tIris Avg Left(" << avgXLeft << ", " << avgYLeft <<
            ")  Right(" << avgXRight << ", " << avgYRight << ")\n";
        }
        
        calibH.eyeLeft = cv::findHomography(irisPtsLeft, screenPts, cv::RANSAC);
        calibH.eyeRight = cv::findHomography(irisPtsRight, screenPts, cv::RANSAC);

        if (calibH.eyeLeft.empty() || calibH.eyeRight.empty()) {
            cout << "Failed Homography during calibration\n";
            exit(EXIT_FAILURE);
        }
    } 
private:
    wstring m_eyePath = L"C:/V_Dev/irisTracker/models/eyeModel.onnx";
    wstring m_irisPath = L"C:/V_Dev/irisTracker/models/irisModel.onnx";

    Camera cam;

    Inferencer eyeInferencer;
    Inferencer irisInferencer;

    CalibrationHomographies calibH;
};


int main() {
    Camera cam;
    PredictLook predicter(cam);

    bool bCalibrate = true;
    DotOverlay* dotOverlay = nullptr;
    if (bCalibrate) {
        dotOverlay = new DotOverlay();
        dotOverlay->runDotOverlay();

        predicter.calibrate(dotOverlay);

        cout << "Calibration succeeded\n";
    }

    while (true) {
        predicter.run(dotOverlay);

        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q') { 
            break;
        }
    }
    if (dotOverlay) {
        delete dotOverlay;
    }
}