cmake_minimum_required(VERSION 3.10)

project(IrisTracker)

set(OpenCV_DIR "C:/V's Dev/opencv/build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} include)

set(SOURCES
    src/main.cpp
    src/myTwoStepInferencer.cpp
)
add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

set(OpenCV_DLL_DIR "C:/V's Dev/opencv/build/x64/vc16/bin")
file(GLOB OPENCV_DLLS "${OpenCV_DLL_DIR}/*.dll")

add_custom_command(TARGET ${PROJECT_NAME}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${OPENCV_DLLS}
        $<TARGET_FILE_DIR:${PROJECT_NAME}>
)