#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "mytargetdetector/modules/camera.hpp"
#include "mytargetdetector/detectors/yolo_detector.hpp"
#include "mytargetdetector/common/constants.hpp"

// 绘图函数：在图像上绘制检测结果
void draw_detections(cv::Mat& frame, const std::vector<my_detector::Detection>& detections) {
    for (const auto& det : detections) {
        // 绘制边界框
        cv::rectangle(frame, det.box, cv::Scalar(0, 255, 0), 2);
        
        // 准备标签文本
        std::string label = "Unknown";
        if (det.class_id >= 0 && det.class_id < my_detector::COCO_CLASSES.size()) {
            label = my_detector::COCO_CLASSES[det.class_id];
        }
        label += ": " + cv::format("%.2f", det.confidence);
        
        // 在边界框上方绘制标签背景和文本
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, nullptr);
        cv::Rect text_bg(det.box.x, det.box.y - text_size.height - 5, text_size.width, text_size.height + 5);
        cv::rectangle(frame, text_bg, cv::Scalar(0, 255, 0), -1);
        cv::putText(frame, label, cv::Point(det.box.x, det.box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
    }
}

int main() {
    // --- 相机选择与配置 ---
    // 您的项目可以轻松在此处切换视频源和摄像头源
    std::unique_ptr<my_detector::Camera> camera;
    
    // ** 选项1: 使用视频文件 (默认) **
    // 您可以下载一些测试视频放到 data/videos/ 目录下
    // 例如：https://www.pexels.com/video/video-of-a-busy-street-3209828/
    std::string video_path = "data/videos/traffic.mp4"; // 请确保这个文件存在
    camera = std::make_unique<my_detector::VideoCamera>(video_path);

    // ** 选项2: 使用USB摄像头 (未来接口已保留) **
    // 如果要切换到机器人上的摄像头，取消下面的注释，并注释掉上面的视频选项
    // 0 通常代表 /dev/video0
    // int camera_index = 0;
    // camera = std::make_unique<my_detector::USBCamera>(camera_index);
    
    if (!camera || !camera->open()) { 
        std::cerr << "错误: 无法打开相机或视频源。" << std::endl;
        std::cerr << "请检查 'data/videos/traffic.mp4' 文件是否存在，或者摄像头是否连接正常。" << std::endl;
        return -1; 
    }
    std::cout << "[INFO] 相机源已成功打开。" << std::endl;

    // --- 加载 YOLOv5n 模型 ---
    std::string model_path = "assets/models/yolov5n.onnx";
    // 调整置信度和NMS阈值以平衡准确率和召回率
    // 0.45f 的置信度对于 yolov5n 是一个不错的起点
    auto detector = std::make_unique<my_detector::YoloDetector>(model_path, 0.45f, 0.5f);

    std::cout << "==========================================" << std::endl;
    std::cout << "[INFO] 使用检测器: YOLOv5n" << std::endl;
    std::cout << "       模型路径: " << model_path << std::endl;
    std::cout << "==========================================" << std::endl;

    cv::Mat frame;
    while (true) {
        if (!camera->get_frame(frame) || frame.empty()) {
            std::cout << "[INFO] 视频流结束或无法获取帧。" << std::endl;
            break; 
        }

        // 执行检测
        auto detections = detector->detect(frame);
        
        // 在帧上绘制结果
        draw_detections(frame, detections);

        cv::imshow("YOLOv5 Detection Result", frame);
        
        // 按 'q' 键或关闭窗口退出
        if (cv::waitKey(1) == 'q' || cv::getWindowProperty("YOLOv5 Detection Result", cv::WND_PROP_VISIBLE) < 1) {
            break;
        }
    }
    
    cv::destroyAllWindows();
    std::cout << "[INFO] 程序已结束。" << std::endl;
    return 0;
}
