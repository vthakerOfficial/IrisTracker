cmake_minimum_required(VERSION 3.28)
project(IrisTracker)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# pybind
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/src/venv/Lib/site-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 CONFIG REQUIRED)
find_package(Python3 COMPONENTS Development REQUIRED)

# opencv
set(OpenCV_DIR "C:/V_Dev/dependencies/opencv/build")
set(OpenCV_INCLUDE_DIR "${OpenCV_DIR}/include")
set(OpenCV_LIB_DIR "${OpenCV_DIR}/x64/vc16/lib")
set(OpenCV_BIN_DIR "${OpenCV_DIR}/x64/vc16/bin")
find_package(OpenCV REQUIRED PATHS ${OpenCV_DIR})

# rest of the file is adding src files and applying ^ (opencv + pybind) files where needed
set(SOURCES
    src/main.cpp
    src/overlayDot.cpp
    src/camera.cpp
    src/pythonBridge.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE
    "${OpenCV_INCLUDE_DIR}"
    "${CMAKE_SOURCE_DIR}/src"
)


target_link_libraries(${PROJECT_NAME} 
    ${OpenCV_LIBS}
    pybind11::embed
)

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
