#pragma once

#include <opencv2/opencv.hpp>
#include <string>

namespace my_detector {

// 相机接口基类
class Camera {
public:
    virtual bool open() = 0;
    virtual bool is_open() const = 0;
    virtual bool get_frame(cv::Mat& frame) = 0;
    virtual ~Camera() = default; // 使用 default 即可
};

// USB 摄像头（为未来使用预留）
class USBCamera : public Camera {
public:
    explicit USBCamera(int device_index);
    bool open() override;
    bool is_open() const override;
    bool get_frame(cv::Mat& frame) override;
private:
    cv::VideoCapture cap_;
    int device_index_;
};

// 视频文件读取
class VideoCamera : public Camera {
public:
    explicit VideoCamera(const std::string& video_path);
    bool open() override;
    bool is_open() const override;
    bool get_frame(cv::Mat& frame) override;
private:
    cv::VideoCapture cap_;
    std::string video_path_;
};

} // namespace my_detector
