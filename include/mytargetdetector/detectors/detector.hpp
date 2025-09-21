#pragma once

#include "mytargetdetector/common/types.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

namespace my_detector {

// 检测器接口基类
class Detector {
public:
    // 纯虚函数，输入一帧图像，返回检测到的所有目标
    virtual std::vector<Detection> detect(const cv::Mat& frame) = 0;

    // 虚析构函数，确保子类能被正确销毁
    virtual ~Detector() = default;
};

} // namespace my_detector
