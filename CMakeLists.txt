cmake_minimum_required(VERSION 3.28)
project(IrisTracker)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set OpenCV directory paths
set(OpenCV_DIR "C:/V_Dev/dependencies/opencv/build")
set(OpenCV_INCLUDE_DIR "C:/V_Dev/dependencies/opencv/build/include")
set(OpenCV_LIB_DIR "C:/V_Dev/dependencies/opencv/build/x64/vc16/lib")
set(OpenCV_BIN_DIR "C:/V_Dev/dependencies/opencv/build/x64/vc16/bin")

# find OpenCV package
find_package(OpenCV REQUIRED PATHS ${OpenCV_DIR})


# create executables
set(SOURCES
    src/main.cpp
    src/myTwoStepInferencer.cpp
    src/overlayDot.cpp
)
add_executable(${PROJECT_NAME} ${SOURCES})

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
set(ONNXRUNTIME_DIR "C:/V_Dev/dependencies/onnxruntime/onnxruntime-win-x64-gpu-1.21.0(1)")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_DIR}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_DIR}/lib")

# link ONNX Runtime & OpenCV libraries
target_link_libraries(${PROJECT_NAME} 
    ${OpenCV_LIBS}
    "${ONNXRUNTIME_LIB_DIR}/onnxruntime.lib"
)

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


# need to add cuda dlls to use onnxruntime with gpu 
set(CUDA_BIN_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin")
file(GLOB CUDA_DLLS "${CUDA_BIN_DIR}/*.dll")
foreach(dll ${CUDA_DLLS})
    add_custom_command(
        TARGET ${PROJECT_NAME}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${dll}"
                $<TARGET_FILE_DIR:${PROJECT_NAME}>
    )
endforeach()
# need to add CUDNN dll files also so that onnxruntime gpu works
set(CUDNN_BIN_DIR "C:/Program Files/NVIDIA/CUDNN/v9.8/bin/12.8")
add_custom_command(
    TARGET 
    ${PROJECT_NAME} 
    POST_BUILD 
    COMMAND 
    ${CMAKE_COMMAND} 
    -E copy_directory
    ${CUDNN_BIN_DIR} 
    $<TARGET_FILE_DIR:${PROJECT_NAME}>
)


# instructions to use gpu onnx, first get onnxruntime gpu build off of website update path above accorindgly
# then go to https://developer.nvidia.com/cudnn, choose download cudnn library, install it, and run .exe
# update path ^ 
# thats it go to code and set bGPUBased = true



# allowing my headers through this cmd
target_include_directories(${PROJECT_NAME} PRIVATE
    "${OpenCV_INCLUDE_DIR}"
    "${ONNXRUNTIME_INCLUDE_DIR}"
    "${CMAKE_SOURCE_DIR}/src" # any ofmy own dirs
)

# prev build timE = [driver] Build completed: 00:00:20.761
