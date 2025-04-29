IrisTracker project: The Goal is to control a computer mouse by calculating where your eyes are looking. 
Using two yolov12 models, first detects eyes, second detects the iris. Using these models together resulted in slow performance in Python.
This project is my attempt at switching all the logic to C++ and getting more performance.

**Update1**:
We have succeeded in creating a working calibration. The script can now predict user gaze on screen post-calibration, 
yet is inaccurate due to our YOLOv12 iris detection model's inaccuracy.
Will need to tweak the iris model if accuracy is to improve, or use a different method of finding pupil location.

**Update2**
Found a better model, Google's face landmarks model, which is proving to be much more accurate. In fact, my tests have shown that the model results enable me to calculate relative face distance. This is critical if I want my gaze prediction to compensate for head movements. Unfortunately, it is natively built for Python. I am currently working on creating a bridge between Python and C++ so Python can run model inference and send results to C++.
