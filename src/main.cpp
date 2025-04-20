#include "overlayDot.h"

#include <windows.h> // for overlay 
#include <thread>
#include <chrono>
#include "pythonBridge.h"
#include "Camera.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <array>
#include <vector>

using namespace std;




class PredictLook {
public:
    struct CalibrationHomographies {
        cv::Mat eyeLeft;
        cv::Mat eyeRight;
    };
public:
    PredictLook(Camera& cam) 
        : cam(cam), pyBridge("../../src/PythonHelper/face_model_server.py")
    {}
    bool run(DotOverlay* dotOverlay = nullptr) {
        array<cv::Point2f, 2> outDetections;
        if (!runPyModel(outDetections)) {
            return false;
        }
        cv::Point2f leftEye = outDetections[0];
        cv::Point2f rightEye = outDetections[1]; 
        cout << "Left Eye: (" << leftEye.x << ", " << leftEye.y << ")\tRight Eye: (" << rightEye.x << ", " << rightEye.y << ")\n";
        cv::Mat frame = cam.getFrame();
        cv::circle(frame, leftEye, 5, cv::Scalar(0, 255, 0), -1);
        cv::circle(frame, rightEye, 5, cv::Scalar(0, 255, 0), -1);
        cv::imshow("Iris Detection", frame);
        
        
        if (!dotOverlay) return true;

        vector<cv::Point2f> srcLeft{ leftEye };
        vector<cv::Point2f> srcRight{ rightEye };

        vector<cv::Point2f> dstLeft, dstRight;
        cv::perspectiveTransform(srcLeft, dstLeft, calibH.eyeLeft);
        cv::perspectiveTransform(srcRight, dstRight, calibH.eyeRight);
        
        cv::Point2f avg( (dstLeft[0].x + dstRight[0].x) / 2.0f, (dstLeft[0].y + dstRight[0].y) / 2.0f );

        dotOverlay->moveTo(avg.x, avg.y);
        return true;
        // cout << "Predicted Coord acc 2 Eye Left(" << dstLeft[0].x << ", " << dstLeft[0].y 
        // << "), Right(" << dstRight[0].x << ", " << dstRight[0].y << "), Avg: "
        // << avg.x << ", " << avg.y << ")\n";
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

        vector<cv::Point2f> irisPtsLeft;
        vector<cv::Point2f> irisPtsRight;

        cout << "Follow red dot, blink on dot move. Keep head still.\n";
        for (cv::Point2f& pt : screenPts) {
            dotOverlay->moveTo(pt.x, pt.y);
            this_thread::sleep_for(std::chrono::seconds(1));
            
            const int numSamples = 5;
            cv::Point2f sumLeft(0, 0);
            cv::Point2f sumRight(0, 0);

            for (int i = 0; i < numSamples; i++) {
                array<cv::Point2f, 2> outDetections;
                bool bSuccess = runPyModel(outDetections);
                if (bSuccess) {
                    sumLeft += outDetections[0];
                    sumRight += outDetections[1];
                } else {
                    cout << "Failed calibration. No iris detected\n";
                    exit(EXIT_FAILURE);
                }
                this_thread::sleep_for(chrono::milliseconds(50));
            }
            cv::Point2f avgLeft = sumLeft / numSamples;
            cv::Point2f avgRight = sumRight / numSamples;
            

            irisPtsLeft.push_back(avgLeft);
            irisPtsRight.push_back(avgRight);

            cout << "Calibration: At screen(" << pt.x << ", " << pt.y << "),\tIris Avg Left(" << avgLeft.x << ", " << avgLeft.y <<
            ")  Right(" << avgRight.x << ", " << avgRight.y << ")\n";
        }
        
        calibH.eyeLeft = cv::findHomography(irisPtsLeft, screenPts, cv::RANSAC);
        calibH.eyeRight = cv::findHomography(irisPtsRight, screenPts, cv::RANSAC);

        if (calibH.eyeLeft.empty() || calibH.eyeRight.empty()) {
            cout << "[Calibration] failed homography\n";
            exit(EXIT_FAILURE);
        }
    } 
private:
    bool runPyModel(array<cv::Point2f, 2>& outDetections) {
        cv::Mat frame = cam.getFrame();
        if (frame.empty()) {
            cout << "[runPyModel] cam frame was empty\n";
            return false;
        }

        pyBridge.sendFrame(frame);
        std::vector<cv::Point2f> landmarks;
        if(!pyBridge.read(landmarks)) {
            return false;
        }

        auto mean5Pts = [&](int start) {
            cv::Point2f sum(0, 0);
            for (int i = 0; i < 5; i++) {
                sum+=landmarks[start + i];
            }
            return sum / 5.0f;
        };

        cv::Point2f leftEye = mean5Pts(468); // last 10 points contain iris results (so 5 for left, 5 for right)
        cv::Point2f rightEye = mean5Pts(473); 
        outDetections[0] = leftEye;
        outDetections[1] = rightEye;
        cout << "Left Eye: (" << leftEye.x << ", " << leftEye.y << ")\tRight Eye: (" << rightEye.x << ", " << rightEye.y << ")\n";
        return true;
    }
private:
    Camera cam;
    CalibrationHomographies calibH;

    PythonBridge pyBridge;
};

int main() {
    Camera cam(0);
    PredictLook predicter(cam);

    bool bCalibrate = false;
    DotOverlay* dotOverlay = nullptr;
    if (bCalibrate) {
        dotOverlay = new DotOverlay();
        dotOverlay->run();

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