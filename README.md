IrisTracker project: Goal is to control computer mouse by calculating where your eyes are looking at. 
Using two yolov12 models, first detects eyes, second detects iris. Using these models together resulted in slow performance in python,
so this project is my attempt at switching all the logic to C++ and getting more performance.
