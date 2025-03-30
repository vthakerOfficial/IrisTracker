cmake_minimum_required(VERSION 3.10)
project(IrisTracker)

# set OpenCV directory paths
set(OpenCV_DIR "C:/V_Dev/dependencies/opencv/build")
set(OpenCV_INCLUDE_DIR "C:/V_Dev/dependencies/opencv/build/include")
set(OpenCV_LIB_DIR "C:/V_Dev/dependencies/opencv/build/x64/vc16/lib")
set(OpenCV_BIN_DIR "C:/V_Dev/dependencies/opencv/build/x64/vc16/bin")

# find OpenCV package
find_package(OpenCV REQUIRED PATHS ${OpenCV_DIR})

# include directories from OpenCV
include_directories(${OpenCV_INCLUDE_DIR})

# create executables
set(SOURCES
    src/main.cpp
    src/myTwoStepInferencer.cpp
)
add_executable(${PROJECT_NAME} ${SOURCES})

# link OpenCV libraries
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# copy opencv dlls to output directory
add_custom_command(
    TARGET 
    ${PROJECT_NAME} 
    POST_BUILD 
    COMMAND 
    ${CMAKE_COMMAND} 
    -E copy_directory 
    ${OpenCV_BIN_DIR} 
    $<TARGET_FILE_DIR:${PROJECT_NAME}>
)

# ALL ONNX STUFF BELOW 

# set ONNX Runtime directory paths
set(ONNXRUNTIME_DIR "C:/V_Dev/dependencies/onnxruntime/onnxruntime-win-x64-gpu-1.21.0")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_DIR}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_DIR}/lib")

# include directories from ONNX Runtime
include_directories(${ONNXRUNTIME_INCLUDE_DIR})

# link ONNX Runtime libraries
target_link_libraries(${PROJECT_NAME} "${ONNXRUNTIME_LIB_DIR}/onnxruntime.lib")

# copy ONNX Runtime dlls to output directory
add_custom_command(
    TARGET 
    ${PROJECT_NAME} 
    POST_BUILD 
    COMMAND 
    ${CMAKE_COMMAND} 
    -E copy_directory
    ${ONNXRUNTIME_LIB_DIR} 
    $<TARGET_FILE_DIR:${PROJECT_NAME}>
)