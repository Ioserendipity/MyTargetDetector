#pragma once

#include "mytargetdetector/detectors/detector.hpp"
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>

namespace my_detector {

class YoloDetector : public Detector {
public:
    YoloDetector(const std::string& model_path, float conf_threshold, float nms_threshold);

    std::vector<Detection> detect(const cv::Mat& frame) override;

private:
    cv::dnn::Net net_;
    float conf_threshold_;
    float nms_threshold_;
    int input_width_;
    int input_height_;
};

} // namespace my_detector
