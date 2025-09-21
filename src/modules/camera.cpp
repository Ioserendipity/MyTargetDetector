#include "mytargetdetector/modules/camera.hpp"
#include <iostream>

namespace my_detector {

// --- USBCamera 实现 ---
USBCamera::USBCamera(int device_index) : device_index_(device_index) {}

bool USBCamera::open() {
    cap_.open(device_index_);
    if (!cap_.isOpened()) {
        std::cerr << "ERROR: Failed to open USB camera at index " << device_index_ << std::endl;
        return false;
    }
    std::cout << "[USBCamera] Camera opened at index " << device_index_ << "." << std::endl;
    return true;
}

bool USBCamera::is_open() const {
    return cap_.isOpened();
}

bool USBCamera::get_frame(cv::Mat& frame) {
    if (!is_open()) return false;
    return cap_.read(frame);
}

// --- VideoCamera 实现 ---
VideoCamera::VideoCamera(const std::string& video_path) : video_path_(video_path) {}

bool VideoCamera::open() {
    cap_.open(video_path_);
    if (!cap_.isOpened()) {
        std::cerr << "ERROR: Failed to open video file at " << video_path_ << std::endl;
        return false;
    }
    std::cout << "[VideoCamera] Video opened at " << video_path_ << "." << std::endl;
    return true;
}

bool VideoCamera::is_open() const {
    return cap_.isOpened();
}

bool VideoCamera::get_frame(cv::Mat& frame) {
    if (!is_open()) {
        frame.release();
        return false;
    }
    
    // 如果读取失败，意味着视频播放完毕
    if (!cap_.read(frame)) {
        std::cout << "[VideoCamera] Video finished. Loop is disabled, stopping." << std::endl;
        frame.release(); // 确保返回空帧
        return false;
    }
    
    return true;
}

} // namespace my_detector
