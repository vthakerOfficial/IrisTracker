#include <iostream>
#include <vector>
#include <string>
#include <array>
#include <cstdlib> // used to exit
#include <chrono> // (to loop every x ms)
#include <thread>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

using namespace std;

class Camera {
private:
    cv::VideoCapture m_cam;
public:
    Camera() {
        m_cam.open(0);
        if (!m_cam.isOpened()) {
            cerr << "Unable to open camera." << endl;
            exit(EXIT_FAILURE);
        }
        cout << "Camera Started\n";
    }

    cv::Mat getFrame() {
        cv::Mat result;
        m_cam >> result;

        if (result.empty()) {
            cerr << "Error: captured an empty frame" << endl;
            exit(EXIT_FAILURE);
        }

        return result;
    }

    ~Camera() {
        m_cam.release();
        cout << "Camera Stopped\n";
    }
};

class CameraInference {
private:
    Camera m_cam;
    string m_eyePath, m_irisPath;
    cv::dnn::Net m_eyeModel;
public:
    CameraInference(const string& eyePath, const string& irisPath) 
        : m_eyePath(eyePath), m_irisPath(irisPath)
    {
        // loading neural network model
        m_eyeModel = cv::dnn::readNetFromONNX(m_eyePath);
        if (m_eyeModel.empty()) {
            cerr << "Error: Failed to load eye model from " << m_eyePath << endl;
            exit(EXIT_FAILURE);
        }
    }
    void test() {
        displayRunInference(m_eyeModel, m_cam.getFrame(), cv::Size(640, 640), .3);
    }
    // will return a vector of vectors containing four values: centerX, centerY, width, and height
    // of the resultant rectangle.The values are between 0-1 and will need to be 
    // multiplied by imgW and imgH to be converted into actual rectangle
                        
    static vector<array<float, 4>> runInference(cv::dnn::Net &net, const cv::Mat &img, const cv::Size &modelShape, float conf) {
        
        cv::Mat blob = cv::dnn::blobFromImage(
            img,
            1.0,
            modelShape
        );
        
        net.setInput(blob); 
        vector<cv::Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());
        
        vector<array<float, 4>> results;
        // loop thru to get only > confidence threshold
        cout << "NumOuts: " << outputs.size() << endl;
        for (cv::Mat output : outputs) {
            for (int i = 0; i < output.rows; i++) {
                float* data = output.ptr<float>(i);

                float centerX = data[0];
                float centerY = data[1];
                float width = data[2];
                float height = data[3];

                // verifying if exceeding conf level
                float objectness = data[4];
                
                int classId = -1;
                float max = 0;
                for (int j = 5; j < output.cols; j++) {
                    float classProb = data[j];
                    if (classProb > max) {
                        max = classProb;
                        classId = j - 5;
                    }
                }

                float resultantConfidence = objectness * max;

                if (resultantConfidence > conf) {
                    array<float,4> result = {centerX, centerY, width, height};
                    // result.push_back(
                    // resultCol.push_back(centerX);
                    // resultCol.push_back(centerY);
                    // resultCol.push_back(width);
                    // resultCol.push_back(height);
                    results.push_back(result);
                }

            }
        }
        return results;
    }

    static void displayRunInference(cv::dnn::Net &net, const cv::Mat &img, const cv::Size &modelShape, float conf) {
        vector<array<float, 4>> outputs = runInference(net, img, modelShape, conf);
        cout << "found: " << outputs.size() << " results";
        int imgW = img.cols;
        int imgH = img.rows;
        
        cv::Mat imgCopy = img.clone();
        for (const array<float, 4> output : outputs) {
            float centerX = output[0];
            float centerY = output[1];
            float width = output[2];
            float height = output[3];

            int rectCenterX = static_cast<int>(centerX * imgW);
            int rectCenterY = static_cast<int>(centerY * imgH);
            int rectWidth = static_cast<int>(width * imgW);
            int rectHeight = static_cast<int>(height * imgH);

            // topleft coords
            int topLeftX = rectCenterX - rectWidth/2;
            int topLeftY = rectCenterX - rectHeight/2;
            int bottomRightX = rectCenterX + rectWidth/2;
            int bottomRightY = rectCenterY + rectWidth/2;

            cv::rectangle(
                imgCopy, 
                cv::Point(topLeftX, topLeftY), 
                cv::Point(bottomRightX, bottomRightY),
                cv::Scalar(0, 255, 0), 
                5
            );
        }
        
        cv::imshow("Vid feed", imgCopy);

        if (cv::waitKey(500) == 27) {
            exit(EXIT_SUCCESS);
        }
    }

};

static const string g_eyePath = "C:/V's Dev/irisTracker/models/eyeModel.onnx";
static const string g_irisPath = "C:/V's Dev/irisTracker/models/irisModel.onnx";
int main() {
    CameraInference inference(g_eyePath, g_irisPath);
    while (true) {

        inference.test();
        
    }

    cout << "Script has finished operating\n";
    // cin.get();
}