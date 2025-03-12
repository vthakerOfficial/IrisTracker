cmake_minimum_required(VERSION 3.10)

project(IrisTracker)

set(OpenCV_DIR "C:/V's Dev/opencv/build")
find_package(OpenCV REQUIRED)
include_directories(${OPENCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} src/main.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})