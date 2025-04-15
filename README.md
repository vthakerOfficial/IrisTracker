IrisTracker project: Goal is to control computer mouse by calculating where your eyes are looking at. 
Using two yolov12 models, first detects eyes, second detects iris. Using these models together resulted in slow performance in python,
so this project is my attempt at switching all the logic to C++ and getting more performance.

**Update**:
We have succeeded in creating a working calibration. The script can now predict user gaze on screen post-calibration, 
yet is a inaccurate due to our yolov12 iris detection model's inaccuracy.
Will need to tweak iris model if accuracy is to improve or use a different method of finding pupil location.
