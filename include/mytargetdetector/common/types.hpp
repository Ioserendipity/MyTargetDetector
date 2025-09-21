#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace my_detector {

// 定义一个更通用的检测结果结构体
struct Detection {
    cv::Rect box;       // 边界框
    float confidence;   // 置信度
    int class_id;       // 类别ID
};

} // namespace my_detector
