IrisTracker project: Goal is to control computer mouse by calculating where your eyes are looking at. 
Using two yolov12 models, first detects eyes, second detects iris. Using these models together resulted in slow performance in python,
so this project is my attempt at switching all the logic to C++ and getting more performance.

**Update1**:
We have succeeded in creating a working calibration. The script can now predict user gaze on screen post-calibration, 
yet is a inaccurate due to our yolov12 iris detection model's inaccuracy.
Will need to tweak iris model if accuracy is to improve or use a different method of finding pupil location.

**Update2**
Found a better model, Google's face landmarks model which is proving to be much more accurate. Unfortunately, it is natively built for python. So currently working on creating a bridge between Python and C++ so Python can run model inference and send results to C++.
