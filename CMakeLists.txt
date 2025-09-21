cmake_minimum_required(VERSION 3.16)
project(MyTargetDetector)

# 设置 C++ 标准为 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 自动查找并配置 OpenCV
# Kubuntu 22.04 通常可以通过 'sudo apt install libopencv-dev' 安装
find_package(OpenCV REQUIRED)

# 将 include 目录添加到头文件搜索路径
include_directories(include)

# 添加可执行文件及其源文件
add_executable(detector_app
    src/main.cpp
    src/modules/camera.cpp
    src/detectors/yolo_detector.cpp
)

# 链接 OpenCV 库
target_link_libraries(detector_app PRIVATE ${OpenCV_LIBS})

# 输出提示信息
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "Include directories: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Libraries: ${OpenCV_LIBS}")
